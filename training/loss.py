# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
from torch_utils import persistence
from torch_utils.git_commit_hash import get_git_commit_hash_from_marker
import torch.distributed as dist

#----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

#----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------

class DenoiserError:
    def __init__(self, log_path='%s_denoiser_l2_norm_squared.log'%get_git_commit_hash_from_marker(), max_sigma=75.0):
        self.log_path = log_path
        self.max_sigma = max_sigma
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        if self.rank == 0:
            os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
            self.logger = dnnlib.util.Logger(file=self.log_path, file_mode='a', should_flush=True)

    def __call__(self, net, images, labels, augment_pipe=None):  # augment_pipe unused
        batch_size = images.shape[0]

        #no gradient accumulation, this is just the measurement
        with torch.no_grad():
            # Sample sigma from Uniform(0, max_sigma)
            sigma = torch.rand([batch_size, 1, 1, 1], device=images.device) * self.max_sigma

            # Generate noisy inputs
            noise = torch.randn_like(images) * sigma
            noisy_input = images + noise

            # Run denoiser
            prediction = net(noisy_input, sigma, labels)

            # Compute L2 error per image
            per_image_error = ((prediction - images) ** 2).view(batch_size, -1).sum(dim=1)  # shape: [B]
            sigma_flat = sigma.view(batch_size)

        # Gather across GPUs
        all_errors = [torch.zeros_like(per_image_error) for _ in range(self.world_size)]
        all_sigma = [torch.zeros_like(sigma_flat) for _ in range(self.world_size)]

        if self.world_size > 1:
            dist.all_gather(all_errors, per_image_error)
            dist.all_gather(all_sigma, sigma_flat)
        else:
            all_errors = [per_image_error]
            all_sigma = [sigma_flat]

        # Log from rank 0
        if self.rank == 0:
            errors_cat = torch.cat(all_errors).cpu().tolist()
            sigmas_cat = torch.cat(all_sigma).cpu().tolist()

            log_line = " ".join(f"{e:.6f}:{s:.6f}" for e, s in zip(errors_cat, sigmas_cat))
            self.logger.print(log_line)

        # Return dummy zero loss, so the net wouldn't get updated
        return torch.zeros_like(per_image_error, requires_grad=True)  # [B]