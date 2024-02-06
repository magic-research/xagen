# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Loss functions."""

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d
from training.dual_discriminator import filtered_resizing

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(
            self, device, G, D, augment_pipe=None, r1_gamma=10, style_mixing_prob=0, pl_weight=0, pl_batch_shrink=2, 
            pl_decay=0.01, pl_no_weight_grad=False, blur_raw_target=True, blur_init_sigma=0, blur_fade_kimg=0, r1_gamma_init=0, r1_gamma_fade_kimg=0, 
            neural_rendering_resolution_initial=64, neural_rendering_resolution_final=None, neural_rendering_resolution_fade_kimg=0, 
            gpc_reg_fade_kimg=1000, gpc_reg_prob=None, dual_discrimination=False, filter_mode='antialiased',eik_weight=1e-3, face_disc_weight=1.0, hand_disc_weight=1.0,
            minsurf_weight=5e-2, minsurf_weight_hand=1e-4, smplx_reg_weight=0.1, deform_reg_weight=10.0, deform_reg_weight_face=1e2, deform_reg_weight_hand=1e2):

        super().__init__()
        self.device             = device
        self.G                  = G
        self.D                  = D
        self.augment_pipe       = augment_pipe
        self.r1_gamma           = r1_gamma
        self.style_mixing_prob  = style_mixing_prob
        self.pl_weight          = pl_weight
        self.pl_batch_shrink    = pl_batch_shrink
        self.pl_decay           = pl_decay
        self.pl_no_weight_grad  = pl_no_weight_grad
        self.pl_mean            = torch.zeros([], device=device)
        self.blur_init_sigma    = blur_init_sigma
        self.blur_fade_kimg     = blur_fade_kimg
        self.r1_gamma_init      = r1_gamma_init
        self.r1_gamma_fade_kimg = r1_gamma_fade_kimg
        self.neural_rendering_resolution_initial = neural_rendering_resolution_initial
        self.neural_rendering_resolution_final = neural_rendering_resolution_final
        self.neural_rendering_resolution_fade_kimg = neural_rendering_resolution_fade_kimg
        self.gpc_reg_fade_kimg = gpc_reg_fade_kimg
        self.gpc_reg_prob = gpc_reg_prob
        self.dual_discrimination = dual_discrimination
        self.filter_mode = filter_mode
        self.resample_filter = upfirdn2d.setup_filter([1,3,3,1], device=device)
        self.blur_raw_target = True
        self.cam_dim = 25
        # avatargen args
        self.eik_weight = eik_weight
        self.face_disc_weight = face_disc_weight
        self.hand_disc_weight = hand_disc_weight
        self.minsurf_weight = minsurf_weight
        self.minsurf_weight_hand = minsurf_weight_hand
        self.smplx_reg_weight = smplx_reg_weight
        self.deform_reg_weight = deform_reg_weight
        self.deform_reg_weight_face = deform_reg_weight_face
        self.deform_reg_weight_hand = deform_reg_weight_hand
        assert self.gpc_reg_prob is None or (0 <= self.gpc_reg_prob <= 1)

    def run_G(self, z, c, c_face, c_hand, swapping_prob, neural_rendering_resolution, update_emas=False, return_eikonal=False, smplx_reg=False):
        if swapping_prob is not None and swapping_prob > 0:
            indices = torch.rand((c.shape[0], 1), device=c.device) < swapping_prob
            c_cam = c[:, :self.cam_dim].clone()  # 16 + 4
            c_swapped = torch.roll(c_cam, 1, 0)
            c_cam_perm = torch.where(indices,  c_swapped, c_cam)
            
            # c_cam_face = c_face[:, :self.cam_dim].clone()  # 16 + 4
            # c_swapped_face = torch.roll(c_cam_face, 1, 0)
            # c_cam_perm_face = torch.where(indices, c_swapped_face, c_cam_face)
            
            # c_cam_hand = c_hand[:, :self.cam_dim].clone()  # 16 + 4
            # c_swapped_hand = torch.roll(c_cam_hand, 1, 0)
            # c_cam_perm_hand = torch.where(indices, c_swapped_hand, c_cam_hand)
        else:
            c_cam_perm = c[:, :self.cam_dim]
            # c_cam_perm_face = c_face[:, :self.cam_dim]
            # c_cam_perm_hand = c_hand[:, :self.cam_dim]

        c_smplx_perm = c[:, self.cam_dim:]

        # TODO: check if we need to asign a value for z_geo?
        # ws_face = self.G.mapping_face(z, c_cam_perm_face, update_emas=update_emas)
        # ws_hand = self.G.mapping_hand(z, c_cam_perm_hand, update_emas=update_emas)
        ws = self.G.mapping(z, c_cam_perm, update_emas=update_emas)
        ws_geo = self.G.mapping_geo(z, c_smplx_perm, update_emas=update_emas)
        # ws_face = ws[:, :1].repeat(1, self.G.backbone_face.num_ws, 1)
        # ws_hand = ws[:, :1].repeat(1, self.G.backbone_hand.num_ws, 1)

        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.G.mapping_body(torch.randn_like(z), c[:, :self.cam_dim], update_emas=False)[:, cutoff:]

        gen_output = self.G.synthesis(
            ws, ws_geo, c, c_face, c_hand, neural_rendering_resolution=neural_rendering_resolution, 
            update_emas=update_emas, return_eikonal=return_eikonal, smplx_reg=smplx_reg)
        return gen_output, ws

    def run_D(self, img, c, c_face, c_hand, blur_sigma=0, blur_sigma_raw=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img['image'].device).div(blur_sigma).square().neg().exp2()
                img['image'] = upfirdn2d.filter2d(img['image'], f / f.sum())
                if self.G.part_disc:
                    img['image_part_face'] = upfirdn2d.filter2d(img['image_part_face'], f / f.sum())
                    img['image_part_right_hand'] = upfirdn2d.filter2d(img['image_part_right_hand'], f / f.sum())
                    img['image_part_left_hand'] = upfirdn2d.filter2d(img['image_part_left_hand'], f / f.sum())

        if self.augment_pipe is not None:
            augmented_pair = self.augment_pipe(
                torch.cat([img['image'],
                torch.nn.functional.interpolate(img['image_raw'], size=img['image'].shape[2:], mode='bilinear', antialias=True)],
                dim=1))
            img['image'] = augmented_pair[:, :img['image'].shape[1]]
            img['image_raw'] = torch.nn.functional.interpolate(augmented_pair[:, img['image'].shape[1]:], size=img['image_raw'].shape[2:], mode='bilinear', antialias=True)

        logits = self.D(img, c, c_face, c_hand, update_emas=update_emas)
        return logits

    def accumulate_gradients(self, phase, real_img, real_img_face, real_img_hand, real_c, real_c_face, real_c_hand, real_vis_face, real_vis_hand, gen_z, gen_c, gen_c_face, gen_c_hand, gain, cur_nimg):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        if self.G.rendering_kwargs.get('density_reg', 0) == 0:
            phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0
        r1_gamma = self.r1_gamma
        r1_gamma_face = 5.0 # TODO: set as argument

        alpha = min(cur_nimg / (self.gpc_reg_fade_kimg * 1e3), 1) if self.gpc_reg_fade_kimg > 0 else 1
        swapping_prob = (1 - alpha) * 1 + alpha * self.gpc_reg_prob if self.gpc_reg_prob is not None else None

        if self.neural_rendering_resolution_final is not None:
            alpha = min(cur_nimg / (self.neural_rendering_resolution_fade_kimg * 1e3), 1)
            neural_rendering_resolution = int(np.rint(self.neural_rendering_resolution_initial * (1 - alpha) + self.neural_rendering_resolution_final * alpha)) // 2 * 2
        else:
            neural_rendering_resolution = self.neural_rendering_resolution_initial

        real_img_raw = filtered_resizing(real_img, size=[neural_rendering_resolution, neural_rendering_resolution//2], f=self.resample_filter, filter_mode=self.filter_mode)
        real_img_face_raw = filtered_resizing(real_img_face, size=[self.G.neural_rendering_resolution_face, self.G.neural_rendering_resolution_face], f=self.resample_filter, filter_mode=self.filter_mode)
        real_img_hand_raw = filtered_resizing(real_img_hand, size=[self.G.neural_rendering_resolution_hand, self.G.neural_rendering_resolution_hand], f=self.resample_filter, filter_mode=self.filter_mode)

        real_vis_face = real_vis_face.unsqueeze(1).to(real_img.dtype)
        real_vis_hand = real_vis_hand.unsqueeze(1).to(real_img.dtype)
        
        if self.blur_raw_target:
            blur_size = np.floor(blur_sigma * 3)
            if blur_size > 0:
                f = torch.arange(-blur_size, blur_size + 1, device=real_img_raw.device).div(blur_sigma).square().neg().exp2()
                real_img_raw = upfirdn2d.filter2d(real_img_raw, f / f.sum())
                f = torch.arange(-blur_size, blur_size + 1, device=real_img_face_raw.device).div(blur_sigma).square().neg().exp2()
                real_img_face_raw = upfirdn2d.filter2d(real_img_face_raw, f / f.sum())
                f = torch.arange(-blur_size, blur_size + 1, device=real_img_hand_raw.device).div(blur_sigma).square().neg().exp2()
                real_img_hand_raw = upfirdn2d.filter2d(real_img_hand_raw, f / f.sum())

        real_img = {
            'image': real_img, 
            'image_raw': real_img_raw, 
            'image_face': real_img_face, 
            'image_raw_face': real_img_face_raw, 
            'image_hand': real_img_hand,
            'image_raw_hand': real_img_hand_raw
        }
        if self.G.part_disc:
            canonical_mapping_kwargs = self.G.get_canonical_mapping_joints(real_c)
            image_part_face, image_part_left_hand, image_part_right_hand = self.G.get_part(real_img['image'], real_c, canonical_mapping_kwargs)
            real_img['image_part_face'] = image_part_face
            real_img['image_part_left_hand'] = image_part_left_hand
            real_img['image_part_right_hand'] = image_part_right_hand
        # real_img['image'].shape: [N, 3, 512, 256] 
        # real_img['image_raw'].shape: [N, 3, 128, 128]
        # Gmain: Maximize logits for generated images.
        if phase in ['Gmain', 'Gboth']:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, gen_c_face, gen_c_hand, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution, return_eikonal=True, smplx_reg=True)
                gen_logits = self.run_D(gen_img, gen_c, gen_c_face, gen_c_hand, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/body/fake', gen_logits['body'])
                training_stats.report('Loss/signs/body/fake', gen_logits['body'].sign())
                training_stats.report('Loss/scores/face/fake', gen_logits['face'])
                training_stats.report('Loss/signs/face/fake', gen_logits['face'].sign())
                training_stats.report('Loss/scores/hand/fake', gen_logits['hand'])
                training_stats.report('Loss/signs/hand/fake', gen_logits['hand'].sign())
                loss_Gmain_body = torch.nn.functional.softplus(-gen_logits['body'])
                training_stats.report('Loss/G/body/loss', loss_Gmain_body)
                training_stats.report('Loss/G/body/beta', self.G.decoder.get_beta())
                loss_Gmain_face = torch.nn.functional.softplus(-gen_logits['face'])
                training_stats.report('Loss/G/face/loss', loss_Gmain_face)
                # training_stats.report('Loss/G/face/beta', self.G.decoder_face.get_beta())
                loss_Gmain_hand = torch.nn.functional.softplus(-gen_logits['hand'])
                training_stats.report('Loss/G/hand/loss', loss_Gmain_hand)
                # training_stats.report('Loss/G/hand/beta', self.G.decoder_hand.get_beta())
                
                # Eikonal 
                # TODO: add loss weights to hyper-parameters
                if gen_img['eikonal'] is not None:
                    loss_eikonal = gen_img['eikonal'].mean().mul(self.eik_weight)
                    training_stats.report('Loss/G/body/eikonal_loss', loss_eikonal)
                # Minsurf loss
                if gen_img['coarse_sdf'] is not None:
                    loss_minsurf = torch.exp(-100 * gen_img['coarse_sdf'].abs()).mean().mul(self.minsurf_weight)
                    training_stats.report('Loss/G/body/minsurf_loss', loss_minsurf)
                    
                if gen_img['eikonal_face'] is not None:
                    loss_eikonal_face = gen_img['eikonal_face'].mean().mul(self.eik_weight)
                    training_stats.report('Loss/G/face/eikonal_loss', loss_eikonal_face)
                    
                if gen_img['eikonal_hand'] is not None:
                    loss_eikonal_hand = gen_img['eikonal_hand'].mean().mul(self.eik_weight)
                    training_stats.report('Loss/G/hand/eikonal_loss', loss_eikonal_hand)
                
                if gen_img['coarse_sdf_face'] is not None:
                    loss_minsurf_face = torch.exp(-100 * gen_img['coarse_sdf_face'].abs()).mean().mul(self.minsurf_weight)
                    training_stats.report('Loss/G/face/minsurf_loss', loss_minsurf_face)

                if gen_img['coarse_sdf_hand'] is not None:
                    loss_minsurf_hand = torch.exp(-100 * gen_img['coarse_sdf_hand'].abs()).mean().mul(self.minsurf_weight_hand)
                    training_stats.report('Loss/G/hand/minsurf_loss', loss_minsurf_hand)
                
                if gen_img['overlap_features_fine'] is not None:
                    loss_con_rgb = torch.nn.functional.mse_loss(gen_img['overlap_features_fine']['overlap_face_rgb'], gen_img['overlap_features_fine']['overlap_body_rgb'])
                    training_stats.report('Loss/G/con_rgb_loss', loss_con_rgb)
                    
                    loss_con_sdf = torch.nn.functional.mse_loss(gen_img['overlap_features_fine']['overlap_face_sdf'], gen_img['overlap_features_fine']['overlap_body_sdf'])
                    training_stats.report('Loss/G/con_sdf_loss', loss_con_sdf)
                    
                    # zero_level_set = (gen_img['overlap_features_fine']['overlap_face_sdf'].abs()<1e-2)
                    # loss_zero_surf = (gen_img['overlap_features_fine']['overlap_body_sdf'][zero_level_set]).norm()    
                    # training_stats.report('Loss/G/zero_surf_loss', loss_zero_surf)              
                    
                # SMPL REG
                if gen_img['smplx_sdf'] is not None:
                    sdf_reg_weights = gen_img['sdf_reg_weights']
                    loss_smplreg = (gen_img['smplx_sdf'].abs() * sdf_reg_weights).mean().mul(self.smplx_reg_weight)
                    training_stats.report('Loss/G/smplreg_loss', loss_smplreg)
                # DEFORM REG
                if gen_img['deformation_reg'] is not None:
                    loss_deformreg = (gen_img['deformation_reg']['body'].abs()).mean().mul(self.deform_reg_weight) + \
                                     (gen_img['deformation_reg']['face'].abs()).mean().mul(self.deform_reg_weight_face) + \
                                     (gen_img['deformation_reg']['hand'].abs()).mean().mul(self.deform_reg_weight_hand)
                    training_stats.report('Loss/G/deformreg_loss', loss_deformreg)

            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss = loss_Gmain_body.mean().mul(gain) + \
                       (loss_Gmain_face * real_vis_face).mean().mul(gain).mul(self.face_disc_weight) + \
                       (loss_Gmain_hand * real_vis_hand).mean().mul(gain).mul(self.hand_disc_weight)
                if gen_img['eikonal'] is not None:
                    loss = loss + loss_eikonal
                if gen_img['coarse_sdf'] is not None:
                    loss = loss + loss_minsurf
                # if gen_img['eikonal_face'] is not None:
                #     loss = loss + loss_eikonal_face
                if gen_img['coarse_sdf_face'] is not None:
                    loss = loss + loss_minsurf_face
                # if gen_img['eikonal_hand'] is not None:
                #     loss = loss + loss_eikonal_hand
                if gen_img['coarse_sdf_hand'] is not None:
                    loss = loss + loss_minsurf_hand
                # if gen_img['overlap_features_fine'] is not None:
                #     loss = loss + loss_con_rgb
                #     loss = loss + loss_con_sdf
                #     # loss = loss + loss_zero_surf
                if gen_img['smplx_sdf'] is not None:
                    loss = loss + loss_smplreg
                if gen_img['deformation_reg'] is not None:
                    loss = loss + loss_deformreg

                loss.backward()

        # Density Regularization
        if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs.get('density_reg', 0) > 0 and self.G.rendering_kwargs['reg_type'] == 'l1':
            raise NotImplementedError()
            if swapping_prob is not None:
                rand_idx = torch.rand([], device=gen_c.device) < swapping_prob
                # c_swapped = torch.roll(gen_c[:, :self.cam_dim].clone(), 1, 0)
                # c_gen_conditioning = torch.where(rand_idx, c_swapped, gen_c[:, :25])
                c_swapped_face = torch.roll(gen_c_face[:, :self.cam_dim].clone(), 1, 0)
                c_gen_conditioning_face = torch.where(rand_idx, c_swapped_face, gen_c_face[:, :25])
            else:
                # c_gen_conditioning = torch.zeros_like(gen_c[:, :self.cam_dim])
                c_gen_conditioning_face = torch.zeros_like(gen_c_face[:, :self.cam_dim])

            c_gen_cam = gen_c[:, :self.cam_dim]
            c_gen_smpl = gen_c[:, self.cam_dim:]
            ws_face = self.G.mapping_face(gen_z, c_gen_conditioning_face, update_emas=False)

            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G.mapping_body(torch.randn_like(z), c_gen_cam, update_emas=False)[:, cutoff:]

            initial_coordinates = torch.rand((ws_face.shape[0], 1000, 3), device=ws_face.device) * 2 - 1
            perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * self.G.rendering_kwargs['density_reg_p_dist']
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws_face, gen_c_face, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            TVloss = torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed) * self.G.rendering_kwargs['density_reg']
            training_stats.report('Loss/G/face/reg', TVloss)
            TVloss.mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dboth']:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, gen_c_face, gen_c_hand, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution, update_emas=True, return_eikonal=False, smplx_reg=False)
                gen_logits = self.run_D(gen_img, gen_c, gen_c_face, gen_c_hand, blur_sigma=blur_sigma, update_emas=True)
                training_stats.report('Loss/scores/body/fake', gen_logits['body'])
                training_stats.report('Loss/signs/body/fake', gen_logits['body'].sign())
                training_stats.report('Loss/scores/face/fake', gen_logits['face'])
                training_stats.report('Loss/signs/face/fake', gen_logits['face'].sign())
                training_stats.report('Loss/scores/hand/fake', gen_logits['hand'])
                training_stats.report('Loss/signs/hand/fake', gen_logits['hand'].sign())
                loss_Dgen_body = torch.nn.functional.softplus(gen_logits['body'])
                loss_Dgen_face = torch.nn.functional.softplus(gen_logits['face'])
                loss_Dgen_hand = torch.nn.functional.softplus(gen_logits['hand'])
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen_body.mean().mul(gain).backward()
                (loss_Dgen_face * real_vis_face).mean().mul(gain).backward()
                (loss_Dgen_hand * real_vis_hand).mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp_image = real_img['image'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_img_tmp_image_raw = real_img['image_raw'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_img_tmp_image_face = real_img['image_face'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_img_tmp_image_raw_face = real_img['image_raw_face'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_img_tmp_image_hand = real_img['image_hand'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_img_tmp_image_raw_hand = real_img['image_raw_hand'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_img_tmp = {'image': real_img_tmp_image, 'image_raw': real_img_tmp_image_raw, 
                                'image_face': real_img_tmp_image_face, 'image_raw_face': real_img_tmp_image_raw_face,
                                'image_hand': real_img_tmp_image_hand, 'image_raw_hand': real_img_tmp_image_raw_hand}
                if self.G.part_disc:
                    real_img_tmp_image_part_face = real_img['image_part_face'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                    real_img_tmp_image_part_left_hand = real_img['image_part_left_hand'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                    real_img_tmp_image_part_right_hand = real_img['image_part_right_hand'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                    real_img_tmp['image_part_face'] = real_img_tmp_image_part_face
                    real_img_tmp['image_part_left_hand'] = real_img_tmp_image_part_left_hand
                    real_img_tmp['image_part_right_hand'] = real_img_tmp_image_part_right_hand
                real_logits = self.run_D(real_img_tmp, real_c, real_c_face, real_c_hand, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/body/real', real_logits['body'])
                training_stats.report('Loss/signs/body/real', real_logits['body'].sign())
                training_stats.report('Loss/scores/face/real', real_logits['face'])
                training_stats.report('Loss/signs/face/real', real_logits['face'].sign())
                training_stats.report('Loss/scores/hand/real', real_logits['hand'])
                training_stats.report('Loss/signs/hand/real', real_logits['hand'].sign())

                loss_Dreal_body = 0
                loss_Dreal_face = 0
                loss_Dreal_hand = 0
                if phase in ['Dmain', 'Dboth']:
                    loss_Dreal_body = torch.nn.functional.softplus(-real_logits['body'])
                    training_stats.report('Loss/D/body/loss', loss_Dgen_body + loss_Dreal_body)
                    loss_Dreal_face = torch.nn.functional.softplus(-real_logits['face']) * real_vis_face
                    training_stats.report('Loss/D/face/loss', loss_Dgen_face + loss_Dreal_face)
                    loss_Dreal_hand = torch.nn.functional.softplus(-real_logits['hand']) * real_vis_hand
                    training_stats.report('Loss/D/hand/loss', loss_Dgen_hand + loss_Dreal_hand)

                loss_Dr1 = 0
                if phase in ['Dreg', 'Dboth']:
                    if self.G.part_disc:
                        with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                            r1_grads_body = torch.autograd.grad(outputs=[real_logits['body'].sum()], inputs=[real_img_tmp['image'], real_img_tmp['image_raw'], real_img_tmp['image_part_face'], real_img_tmp['image_part_right_hand'], real_img_tmp['image_part_left_hand']], create_graph=True, only_inputs=True)
                            r1_grads_face = torch.autograd.grad(outputs=[real_logits['face'].sum()], inputs=[real_img_tmp['image_face'], real_img_tmp['image_raw_face']], create_graph=True, only_inputs=True)
                            r1_grads_hand = torch.autograd.grad(outputs=[real_logits['hand'].sum()], inputs=[real_img_tmp['image_hand'], real_img_tmp['image_raw_hand']], create_graph=True, only_inputs=True)
                            r1_grads_image = r1_grads_body[0]
                            r1_grads_image_raw = r1_grads_body[1]
                            r1_grads_image_part_face = r1_grads_body[2]
                            r1_grads_image_part_right_hand = r1_grads_body[3]
                            r1_grads_image_part_left_hand = r1_grads_body[4]
                            r1_grads_image_face = r1_grads_face[0]
                            r1_grads_image_raw_face = r1_grads_face[1]
                            r1_grads_image_hand = r1_grads_hand[0]
                            r1_grads_image_raw_hand = r1_grads_hand[1]
                        r1_penalty_body = r1_grads_image.square().sum([1,2,3]) + r1_grads_image_raw.square().sum([1,2,3]) + r1_grads_image_part_face.square().sum([1,2,3]) + r1_grads_image_part_right_hand.square().sum([1,2,3]) + r1_grads_image_part_left_hand.square().sum([1,2,3])
                        r1_penalty_face = r1_grads_image_face.square().sum([1,2,3]) + r1_grads_image_raw_face.square().sum([1,2,3])
                        r1_penalty_hand = r1_grads_image_hand.square().sum([1,2,3]) + r1_grads_image_raw_hand.square().sum([1,2,3])
                    else:
                        if self.dual_discrimination:
                            with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                                r1_grads_body = torch.autograd.grad(outputs=[real_logits['body'].sum()], inputs=[real_img_tmp['image'], real_img_tmp['image_raw']], create_graph=True, only_inputs=True)
                                r1_grads_face = torch.autograd.grad(outputs=[real_logits['face'].sum()], inputs=[real_img_tmp['image_face'], real_img_tmp['image_raw_face']], create_graph=True, only_inputs=True)
                                r1_grads_hand = torch.autograd.grad(outputs=[real_logits['hand'].sum()], inputs=[real_img_tmp['image_hand'], real_img_tmp['image_raw_hand']], create_graph=True, only_inputs=True)
                                r1_grads_image = r1_grads_body[0]
                                r1_grads_image_raw = r1_grads_body[1]
                                r1_grads_image_face = r1_grads_face[0]
                                r1_grads_image_raw_face = r1_grads_face[1]
                                r1_grads_image_hand = r1_grads_hand[0]
                                r1_grads_image_raw_hand = r1_grads_hand[1]
                            r1_penalty_body = r1_grads_image.square().sum([1,2,3]) + r1_grads_image_raw.square().sum([1,2,3])
                            r1_penalty_face = r1_grads_image_face.square().sum([1,2,3]) + r1_grads_image_raw_face.square().sum([1,2,3])
                            r1_penalty_hand = r1_grads_image_hand.square().sum([1,2,3]) + r1_grads_image_raw_hand.square().sum([1,2,3])
                        else: # single discrimination
                            with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                                r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp['image']], create_graph=True, only_inputs=True)
                                r1_grads_image = r1_grads[0]
                            r1_penalty = r1_grads_image.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty_body * (r1_gamma / 2) + r1_penalty_face * real_vis_face.squeeze(1) * (r1_gamma / 2) + r1_penalty_hand * real_vis_hand.squeeze(1) * (r1_gamma / 2)
                    training_stats.report('Loss/body/r1_penalty', r1_penalty_body)
                    training_stats.report('Loss/face/r1_penalty', r1_penalty_face)
                    training_stats.report('Loss/hand/r1_penalty', r1_penalty_hand)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal_body + loss_Dreal_face + loss_Dreal_hand + loss_Dr1).mean().mul(gain).backward()

#----------------------------------------------------------------------------
