# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Discriminator architectures from the paper
"Efficient Geometry-aware 3D Generative Adversarial Networks"."""

import numpy as np
import torch
from torch_utils import persistence
from torch_utils.ops import upfirdn2d
from training.networks_stylegan2 import DiscriminatorBlock, MappingNetwork, DiscriminatorEpilogue, FullyConnectedLayer
from training.face.dual_discriminator import DualDiscriminator as DualDiscriminatorFace
from training.hand.dual_discriminator import DualDiscriminator as DualDiscriminatorHand

@persistence.persistent_class
class SingleDiscriminator(torch.nn.Module):
    def __init__(self,
        c_dim,                          # Conditioning label (C) dimensionality.
        img_resolution,                 # Input resolution.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_fp16_res        = 4,        # Use FP16 for the N highest resolutions.
        conv_clamp          = 256,      # Clamp the output of convolution layers to +-X, None = disable clamping.
        cmap_dim            = None,     # Dimensionality of mapped conditioning label, None = default.
        sr_upsample_factor  = 1,        # Ignored for SingleDiscriminator
        block_kwargs        = {},       # Arguments for DiscriminatorBlock.
        mapping_kwargs      = {},       # Arguments for MappingNetwork.
        epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.
    ):
        super().__init__()
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        if cmap_dim is None:
            cmap_dim = channels_dict[4]
        if c_dim == 0:
            cmap_dim = 0

        common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs, **common_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers
        if c_dim > 0:
            self.mapping = MappingNetwork(z_dim=0, c_dim=c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None, **mapping_kwargs)
        self.b4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=cmap_dim, resolution=4, **epilogue_kwargs, **common_kwargs)

    def forward(self, img, c, update_emas=False, **block_kwargs):
        img = img['image']

        _ = update_emas # unused
        x = None
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img = block(x, img, **block_kwargs)

        cmap = None
        if self.c_dim > 0:
            cmap = self.mapping(None, c)
        x = self.b4(x, img, cmap)
        return x

    def extra_repr(self):
        return f'c_dim={self.c_dim:d}, img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d}'

#----------------------------------------------------------------------------

def filtered_resizing(image_orig_tensor, size, f, filter_mode='antialiased'):
    if isinstance(size, list):
        H, W = size
    else:
        H = size
        W = size // 2
    
    if filter_mode == 'antialiased':
        ada_filtered_64 = torch.nn.functional.interpolate(image_orig_tensor, size=(H, W), mode='bilinear', align_corners=False) #, antialias=True) -- Not available in PyTorch 1.10.0
    elif filter_mode == 'classic':
        ada_filtered_64 = upfirdn2d.upsample2d(image_orig_tensor, f, up=2)
        ada_filtered_64 = torch.nn.functional.interpolate(ada_filtered_64, size=(H * 2 + 2, W * 2 + 2), mode='bilinear', align_corners=False)
        ada_filtered_64 = upfirdn2d.downsample2d(ada_filtered_64, f, down=2, flip_filter=True, padding=-1)
    elif filter_mode == 'none':
        ada_filtered_64 = torch.nn.functional.interpolate(image_orig_tensor, size=(H, W), mode='bilinear', align_corners=False)
    elif type(filter_mode) == float:
        assert 0 < filter_mode < 1

        filtered = torch.nn.functional.interpolate(image_orig_tensor, size=(H, W), mode='bilinear', align_corners=False) #, antialias=True) -- Not available in PyTorch 1.10.0
        aliased  = torch.nn.functional.interpolate(image_orig_tensor, size=(H, W), mode='bilinear', align_corners=False) #, antialias=False) -- Not available in PyTorch 1.10.0
        ada_filtered_64 = (1 - filter_mode) * aliased + (filter_mode) * filtered
        
    return ada_filtered_64

#----------------------------------------------------------------------------

@persistence.persistent_class
class DualDiscriminator(torch.nn.Module):
    def __init__(self,
        c_dim,                          # Conditioning label (C) dimensionality.
        img_resolution,                 # Input resolution.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_fp16_res        = 4,        # Use FP16 for the N highest resolutions.
        conv_clamp          = 256,      # Clamp the output of convolution layers to +-X, None = disable clamping.
        cmap_dim            = None,     # Dimensionality of mapped conditioning label, None = default.
        disc_c_noise        = 0,        # Corrupt camera parameters with X std dev of noise before disc. pose conditioning.
        block_kwargs        = {},       # Arguments for DiscriminatorBlock.
        mapping_kwargs      = {},       # Arguments for MappingNetwork.
        epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.
        **kwargs
    ):
        super().__init__()
        img_channels *= 2

        self.c_dim = c_dim = 25
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution[0]))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        if cmap_dim is None:
            cmap_dim = channels_dict[4]
        if c_dim == 0:
            cmap_dim = 0

        common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution[0] else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs, **common_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers
        if c_dim > 0:
            self.mapping = MappingNetwork(z_dim=0, c_dim=c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None, **mapping_kwargs)
        self.b4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=cmap_dim, resolution=4, **epilogue_kwargs, **common_kwargs)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter([1,3,3,1]))
        self.disc_c_noise = disc_c_noise

    def forward(self, image, image_raw, c, update_emas=False, **block_kwargs):
        image_raw = filtered_resizing(image_raw, size=image.shape[-2], f=self.resample_filter)
        img = torch.cat([image, image_raw], 1)

        _ = update_emas # unused
        x = None
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img = block(x, img, **block_kwargs)

        cmap = None
        if self.c_dim > 0:
            if self.disc_c_noise > 0: c += torch.randn_like(c) * c.std(0) * self.disc_c_noise
            cmap = self.mapping(None, c[:, :self.c_dim])
        x = self.b4(x, img, cmap)
        return x

    def extra_repr(self):
        return f'c_dim={self.c_dim:d}, img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d}'
    
#----------------------------------------------------------------------------

@persistence.persistent_class
class JointDualDiscriminator(torch.nn.Module):
    def __init__(self,
        c_dim,                          # Conditioning label (C) dimensionality.
        img_resolution,                 # Input resolution.
        img_channels,                   # Number of input color channels.
        **kwargs
    ):
        super().__init__()
        # self.body_discriminator = MultiDiscriminator(c_dim, img_resolution, img_channels, **kwargs)
        self.body_discriminator = DualDiscriminator(c_dim, img_resolution, img_channels, **kwargs)
        self.face_discriminator = DualDiscriminatorFace(c_dim, 256, img_channels, **kwargs)
        self.hand_discriminator = DualDiscriminatorHand(c_dim, 256, img_channels, **kwargs)
        
        # Load body discriminator backbone.
        backbone_body_pkl = kwargs.get('backbone_body_pkl', None)
        if backbone_body_pkl is not None:
            print(f'Loading body discriminator from "{backbone_body_pkl}"...')
            state_dict = torch.load(backbone_body_pkl)['D']
            self.body_discriminator.load_state_dict(state_dict)
        # else:
        #     raise RuntimeError("Pretrained body backbone not found")
        
        # Load face discriminator backbone.
        backbone_face_pkl = kwargs.get('backbone_face_pkl', None)
        if backbone_face_pkl is not None:
            print(f'Loading face discriminator from "{backbone_face_pkl}"...')
            state_dict = torch.load(backbone_face_pkl)['D']
            self.face_discriminator.load_state_dict(state_dict)
        # else:
        #     raise RuntimeError("Pretrained face backbone not found")
        
        # Load hand discriminator backbone.
        backbone_hand_pkl = kwargs.get('backbone_hand_pkl', None)
        if backbone_hand_pkl is not None:
            print(f'Loading hand discriminator from "{backbone_hand_pkl}"...')
            state_dict = torch.load(backbone_hand_pkl)['D']
            self.hand_discriminator.load_state_dict(state_dict)
        # else:
        #     raise RuntimeError("Pretrained hand backbone not found")
        
    def forward(self, img, c_body, c_face, c_hand, update_emas=False, **block_kwargs):
        x_body, x_face, x_hand = None, None, None
        # x_body = self.body_discriminator(img['image'], img['image_raw'], img['image_part_face'], img['image_part_right_hand'], img['image_part_left_hand'], c_body, update_emas=update_emas, **block_kwargs)
        x_body = self.body_discriminator(img['image'], img['image_raw'], c_body, update_emas=update_emas, **block_kwargs)
        x_face = self.face_discriminator(img['image_face'], img['image_raw_face'], c_face, update_emas=update_emas, **block_kwargs)
        x_hand = self.hand_discriminator(img['image_hand'], img['image_raw_hand'], c_hand, update_emas=update_emas, **block_kwargs)
        logits = {'body':x_body, 'face':x_face, 'hand':x_hand}
        return logits
    
    def get_parameters(self):
        '''
        Freeze selected submodules
        '''
        # TODO: set different learning rates for each submodule
        frozen_modules = []
        for n, m in self.named_parameters():
            if n.split(".")[0] in frozen_modules:
                continue
            yield m


#----------------------------------------------------------------------------


@persistence.persistent_class
class MultiDiscriminator(torch.nn.Module):
    def __init__(self,
        c_dim,                          # Conditioning label (C) dimensionality.
        img_resolution,                 # Input resolution.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_fp16_res        = 4,        # Use FP16 for the N highest resolutions.
        conv_clamp          = 256,      # Clamp the output of convolution layers to +-X, None = disable clamping.
        cmap_dim            = None,     # Dimensionality of mapped conditioning label, None = default.
        disc_c_noise        = 0,        # Corrupt camera parameters with X std dev of noise before disc. pose conditioning.
        block_kwargs        = {},       # Arguments for DiscriminatorBlock.
        mapping_kwargs      = {},       # Arguments for MappingNetwork.
        epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.
        weight_full_body    = 6e-1, 
        weight_part_face    = 2e-1,
        weight_part_right_hand = 1e-1,
        weight_part_left_hand = 1e-1, 
        **kwargs
    ):
        super().__init__()
        img_channels *= 2
        c_dim = 25
        smplx_dim = 409

        self.c_dim = c_dim
        self.smplx_dim = smplx_dim
        self.img_resolution = img_resolution[0]
        self.img_resolution_log2 = int(np.log2(img_resolution[0]))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        # part disc
        self.p_img_resolution = img_resolution[0]//8
        self.p_img_channels = img_channels // 2

        if cmap_dim is None:
            cmap_dim = channels_dict[4]
        if c_dim == 0:
            cmap_dim = 0

        common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)
        common_kwargs_p = dict(img_channels=self.p_img_channels, architecture=architecture, conv_clamp=conv_clamp)
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution[0] else 0
            pf_in_channels = in_channels if res < self.p_img_resolution else 0
            ph_in_channels = in_channels if res < self.p_img_resolution // 2 else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)

            # build part disc
            if res <= self.p_img_resolution:
                p_block = DiscriminatorBlock(pf_in_channels, tmp_channels, out_channels, resolution=res,
                    first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs, **common_kwargs_p)
                setattr(self, f'pbf{res}', p_block)
                
            if res <= self.p_img_resolution // 2:
                p_block = DiscriminatorBlock(ph_in_channels, tmp_channels, out_channels, resolution=res,
                    first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs, **common_kwargs_p)
                setattr(self, f'pblh{res}', p_block)
                p_block = DiscriminatorBlock(ph_in_channels, tmp_channels, out_channels, resolution=res,
                    first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs, **common_kwargs_p)
                setattr(self, f'pbrh{res}', p_block)

            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs, **common_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers
        if c_dim > 0:
            self.mapping = MappingNetwork(z_dim=0, c_dim=c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None, **mapping_kwargs)
        if smplx_dim > 0:
            self.mapping_smplx = MappingNetwork(z_dim=0, c_dim=smplx_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None, **mapping_kwargs)
            self.mapping_smplx_face = MappingNetwork(z_dim=0, c_dim=53, w_dim=128, num_ws=None, w_avg_beta=None, **mapping_kwargs)
            self.mapping_smplx_right_hand = MappingNetwork(z_dim=0, c_dim=45, w_dim=128, num_ws=None, w_avg_beta=None, **mapping_kwargs)
            self.mapping_smplx_left_hand = MappingNetwork(z_dim=0, c_dim=45, w_dim=128, num_ws=None, w_avg_beta=None, **mapping_kwargs)
        self.pbf4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=128, resolution=4, square=True, **epilogue_kwargs, **common_kwargs)
        self.pblh4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=128, resolution=4, square=True, **epilogue_kwargs, **common_kwargs)
        self.pbrh4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=128, resolution=4, square=True, **epilogue_kwargs, **common_kwargs)
        self.b4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=cmap_dim, resolution=4, **epilogue_kwargs, **common_kwargs)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter([1,3,3,1]))
        self.disc_c_noise = disc_c_noise
        
        self.weight_full_body = weight_full_body
        self.weight_part_face = weight_part_face
        self.weight_part_right_hand = weight_part_right_hand
        self.weight_part_left_hand = weight_part_left_hand
        
        assert (weight_full_body + weight_part_face + weight_part_right_hand + weight_part_left_hand) == 1
        
        # # Load body discriminator backbone.
        # if backbone_body_pkl is not None:
        #     print(f'Loading body discriminator from "{backbone_body_pkl}"...')
        #     state_dict = torch.load(backbone_body_pkl)['D']
        #     self.load_state_dict(state_dict)
        # # else:
        # #     raise RuntimeError("Pretrained body backbone not found")

    def forward(self, image, image_raw, image_part_face, image_part_right_hand, image_part_left_hand, c, update_emas=False, **block_kwargs):
        image_raw = filtered_resizing(image_raw, size=self.img_resolution, f=self.resample_filter)
        image_part_face = filtered_resizing(image_part_face, size=[self.p_img_resolution, self.p_img_resolution], f=self.resample_filter)
        image_part_right_hand = filtered_resizing(image_part_right_hand, size=[self.p_img_resolution//2, self.p_img_resolution//2], f=self.resample_filter)
        image_part_left_hand = filtered_resizing(image_part_left_hand, size=[self.p_img_resolution//2, self.p_img_resolution//2], f=self.resample_filter)
        img = torch.cat([image, image_raw], 1)

        x, x_part_face, x_part_left_hand, x_part_right_hand = None, None, None, None
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img = block(x, img, **block_kwargs)
            if res <= self.p_img_resolution:
                p_block = getattr(self, f'pbf{res}')
                x_part_face, image_part_face = p_block(x_part_face, image_part_face, **block_kwargs)
            if res <= self.p_img_resolution // 2:
                plh_block = getattr(self, f'pblh{res}')
                x_part_left_hand, image_part_left_hand = plh_block(x_part_left_hand, image_part_left_hand, **block_kwargs)
                prh_block = getattr(self, f'pbrh{res}')
                x_part_right_hand, image_part_right_hand = prh_block(x_part_right_hand, image_part_right_hand, **block_kwargs)
        
        cam = c[:, :self.c_dim]
        smplx = c[:, self.c_dim:(self.c_dim+self.smplx_dim)]

        cmap = None
        if self.c_dim > 0:
            if self.disc_c_noise > 0: cam += torch.randn_like(cam) * cam.std(0) * self.disc_c_noise
            cam_map = self.mapping(None, cam)
            cmap = cam_map
        if self.smplx_dim > 0:
            # if self.disc_c_noise > 0: cam += torch.randn_like(cam) * cam.std(0) * self.disc_c_noise
            smplx_map = self.mapping_smplx(None, smplx)
            cmap = 0.5 * (cam_map + smplx_map)
            
            # NOTE: face smplx: expression, jaw pose
            #       right hand smplx: right hand pose
            #       left hand smplx: left hand pose
            expression = smplx[:, 200:250]
            jaw_pose = smplx[:, 316:319]
            left_hand_pose = smplx[:, 319:364]
            right_hand_pose = smplx[:, 364:]
            cmap_face = self.mapping_smplx_face(None, torch.cat([expression, jaw_pose], dim=1))
            cmap_right_hand = self.mapping_smplx_right_hand(None, right_hand_pose)
            cmap_left_hand = self.mapping_smplx_left_hand(None, left_hand_pose)

        x = self.b4(x, img, cmap)
        x_part_face = self.pbf4(x_part_face, image_part_face, cmap_face)
        x_part_right_hand = self.pbrh4(x_part_right_hand, image_part_right_hand, cmap_right_hand)
        x_part_left_hand = self.pblh4(x_part_left_hand, image_part_left_hand, cmap_left_hand)
        
        out = self.weight_full_body * x + self.weight_part_face * x_part_face + self.weight_part_right_hand * x_part_right_hand + self.weight_part_left_hand * x_part_left_hand
        
        return out

    def extra_repr(self):
        return f'c_dim={self.c_dim:d}, img_resolution={self.img_resolution[0]:d}, img_channels={self.img_channels:d}'