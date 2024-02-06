# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import re, cv2
from typing import List, Optional, Tuple, Union
import math

import click
import dnnlib
import PIL
import numpy as np
import torch
import math

import legacy
from torch_utils import misc
from training.volumetric_rendering.utils import (
    generate_camera_params, align_volume, extract_mesh_with_marching_cubes,
    xyz2mesh, create_cameras, create_mesh_renderer, create_mesh_normal_renderer, create_depth_mesh_renderer,
    add_textures, get_canonical_pose
)

from training.triplane_visualize import TriPlaneGenerator
from pytorch3d.structures import Meshes

#----------------------------------------------------------------------------
def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    return samples.unsqueeze(0), voxel_origin, voxel_size

def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')

#----------------------------------------------------------------------------

def make_transform(translate: Tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', required=True)
@click.option('--poseids', type=parse_range, help='List of pose ids (e.g., \'0,1,4-6\')', default=[])
@click.option('--data',help='dataset path (to load camera labels)', metavar='[ZIP|DIR]', type=str, default=None)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--translate', help='Translate XY-coordinate (e.g. \'0.3,1\')', type=parse_vec2, default='0,0', show_default=True, metavar='VEC2')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--save-geometry', help='Save mesh and normal map', type=bool, default=False, show_default=True)
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, required=False, metavar='BOOL', default=True, show_default=True)
@click.option('--sample_mult', 'sampling_multiplier', type=float, help='Multiplier for depth sampling in volume rendering', default=1, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=0, show_default=True)
@click.option('--zsamples_dir',     help='Where to load z_samples.npz', required=False, default=None, metavar='DIR')
@click.option('--img_size',     help='Where to load z_samples.npz', required=False, default=512)
@click.option('--repose_seed',  type=int,default=0, help='seed for repose smpl label')
@click.option('--fps',          help='Frames per second of final video', default=30, show_default=True)
def generate_images(
    network_pkl: str,
    seeds: List[int],
    poseids: List[int],
    data: str,
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    translate: Tuple[float,float],
    class_idx: Optional[int],
    save_geometry: bool,
    reload_modules: bool,
    sampling_multiplier: float,
    truncation_cutoff: int,
    zsamples_dir: str,
    img_size: int,
    repose_seed: int,
    fps: int,
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate an image using pre-trained AFHQv2 model ("Ours" in Figure 1, left).
    python gen_images.py --outdir=out --trunc=1 --seeds=2 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl

    \b
    # Generate uncurated images with truncation using the MetFaces-U dataset
    python gen_images.py --outdir=out --trunc=0.7 --seeds=600-605 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-metfacesu-1024x1024.pkl
    """
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    if True:
        print("Reloading Modules!")
        init_kwargs = G.init_kwargs
        init_kwargs['rendering_kwargs']['depth_noise'] = False
        init_kwargs['rendering_kwargs']['depth_resolution'] = 64
        init_kwargs['rendering_kwargs']['depth_resolution_importance'] = 48
        G_new = TriPlaneGenerator(*G.init_args, **init_kwargs).eval().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(G, G_new, require_all=True)
        G_new.neural_rendering_resolution = G.neural_rendering_resolution
        G_new.neural_rendering_resolution_face = G.neural_rendering_resolution_face
        G_new.neural_rendering_resolution_hand = G.neural_rendering_resolution_hand
        G_new.rendering_kwargs = G.rendering_kwargs
        G = G_new
        print(G_new.neural_rendering_resolution, G_new.neural_rendering_resolution_face, G_new.neural_rendering_resolution_hand)

    G.rendering_kwargs['depth_resolution'] = int(G.rendering_kwargs['depth_resolution'] * sampling_multiplier)
    G.rendering_kwargs['depth_resolution_importance'] = int(G.rendering_kwargs['depth_resolution_importance'] * sampling_multiplier)
    if truncation_cutoff == 0:
        truncation_psi = 1.0 # truncation cutoff of 0 means no truncation anyways
    if truncation_psi == 1.0:
        truncation_cutoff = 14 # no truncation so doesn't matter where we cutoff

    if data is not None:
        dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset_infer.SMPLLabeledDataset', path=data, use_labels=True, max_size=None, xflip=False)
        dataset = dnnlib.util.construct_class_by_name(**dataset_kwargs) # Subclass of training.dataset.Dataset.
    else:
        dataset = None

    if dataset is not None:
        labels = dataset._get_raw_labels()
        rand_idx = np.random.RandomState(0).choice(range(len(labels)), 128)
        cs = torch.from_numpy(labels[rand_idx]).to(device)
        labels_face = dataset._get_raw_labels_face()
        cs_face = torch.from_numpy(labels_face[rand_idx]).to(device)
        labels_hand = dataset._get_raw_labels_hand()
        cs_hand = torch.from_numpy(labels_hand[rand_idx]).to(device)

    os.makedirs(f"{outdir}", exist_ok=True)

    M = torch.eye(4, device=device)
    M[1, 1] *= -1
    M[2, 2] *= -1

    # more samples per ray for better view/repose consistency
    render_params = {'depth_resolution': 64, 'depth_resolution_importance': 48}
    g_render_params = {'depth_resolution': 256, 'depth_resolution_importance': 0} #470

    # Generate images.
    seed_idx = 85
    for _, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        rots = [i for i in range(0, 360, 30)]
        for idx, rotate in enumerate(rots):
            
            t = rotate/360+0.5
            azim = math.pi * 1 * np.cos(t * 1 * math.pi)
            elev = math.pi * 0 * np.sin(t * 1 * math.pi)
            fov = 12
            trajectory = torch.from_numpy(np.array([azim, elev, fov])).float().unsqueeze(0).to(device)
            sample_cameras, sample_focals, sample_near, sample_far, _ = generate_camera_params(G.neural_rendering_resolution, device, locations=trajectory[:, :2], fov=trajectory[:, 2:])
            sample_cameras = sample_cameras[:, :3, :3]
            
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)

            c_body = cs[seed_idx: seed_idx+1]
            c_face = cs_face[seed_idx: seed_idx+1]
            c_hand = cs_hand[seed_idx: seed_idx+1]
            G.rendering_kwargs.update(render_params)
            ws = G.mapping(z, c_body[:, :25], truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
            
            M = torch.eye(4, device=device)
            M[1, 1] *= -1
            M[2, 2] *= -1
            cam2world_pose = torch.linalg.inv(c_body[:, :16].reshape(4, 4).to(device))@M
            Rz = torch.eye(4, device=device)
            Rz[0, 0] = torch.cos(torch.tensor(rotate)/180*torch.pi)
            Rz[0, 2] = torch.sin(torch.tensor(rotate)/180*torch.pi)
            Rz[2, 0] = -torch.sin(torch.tensor(rotate)/180*torch.pi)
            Rz[2, 2] = torch.cos(torch.tensor(rotate)/180*torch.pi)
            cam2world_pose = torch.linalg.inv(cam2world_pose@Rz@M).reshape(-1, 16)
            rot_c_body = c_body.clone()
            rot_c_body[:, :16] = cam2world_pose
            c_body = rot_c_body            
            
            output = G.synthesis(ws, None, c_body, c_face, c_hand)
            img = output['image']
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/rgb_seed{seed:04d}_{idx:04d}.png')

            G.rendering_kwargs.update(g_render_params)

            H = 400
            W = 200
            canonical_mapping_kwargs_body = G.get_canonical_mapping_info(c_body, mode="body")
            ray_origins_body, ray_directions_body = G.ray_sampler(c_body[:, :16].view(-1, 4, 4), c_body[:, 16:25].view(-1, 3, 3), [H, W])
            cam_dist = torch.norm(ray_origins_body, dim=-1, keepdim=True)
            near = cam_dist[:, :1, 0] - 0.75
            far = cam_dist[:, :1, 0] + 0.75
            near = near.unsqueeze(-1) * torch.ones_like(ray_directions_body[..., :1]).float()
            far = far.unsqueeze(-1) * torch.ones_like(ray_directions_body[..., :1]).float()

            ray_start = torch.ones_like(near) * canonical_mapping_kwargs_body['obs_bbox_cam'][:, :1, 2:]
            ray_end = torch.ones_like(far) * canonical_mapping_kwargs_body['obs_bbox_cam'][:, 1:, 2:]
            
            depths_coarse = G.renderer_body.sample_stratified(ray_origins_body, ray_start, ray_end, G.rendering_kwargs['depth_resolution'], False, G.rendering_kwargs['disparity_space_sampling'])
            
            batch_size, num_rays, samples_per_ray, _ = depths_coarse.shape
            coordinates = (ray_origins_body.unsqueeze(-2) + depths_coarse * ray_directions_body.unsqueeze(-2)).reshape(batch_size, -1, 3)
            directions = ray_directions_body.unsqueeze(-2).expand(-1, -1, samples_per_ray, -1).reshape(batch_size, -1, 3)
            
            sigmas = torch.zeros((coordinates.shape[0], coordinates.shape[1], 1), device=device)

            batch_gen = 4*4
            for i in range(batch_gen):
                with torch.no_grad():
                    output = G.sample_mixed(coordinates[:, i::batch_gen], directions[:, i::batch_gen], ws, c_body, c_face, c_hand, canonical_mapping_kwargs_body)
                    sigmas[:, i::batch_gen] = output['sdf']

            frostum_aligned_sdf = align_volume(sigmas.reshape(1, H, W, g_render_params['depth_resolution'],1))

            marching_cubes_mesh = extract_mesh_with_marching_cubes(frostum_aligned_sdf, level=0)
            G.rendering_kwargs.update(render_params)
            mc_mesh = Meshes(
                verts=[torch.from_numpy(np.asarray(marching_cubes_mesh.vertices)).to(torch.float32).to(device)],
                faces=[torch.from_numpy(np.asarray(marching_cubes_mesh.faces)).to(torch.float32).to(device)],
                textures=None,
                verts_normals=[torch.from_numpy(np.copy(np.asarray(marching_cubes_mesh.vertex_normals))).to(torch.float32).to(device)],)
            mc_mesh = add_textures(mc_mesh)
            render_R = np.eye(3)
            render_R[0,0] *= -1
            render_R[2,2] *= -1
            # render_R = np.matmul(sample_cameras.cpu().numpy()[0].T,render_R)
            cameras = create_cameras(R=render_R[None], T=np.array([[0, 0, 1]]), fov=12, dist=1, device=device)

            def gen_img(a_color, d_color, s_color, l_location):
                light_kwargs = {
                    "ambient_color": a_color,
                    "diffuse_color": d_color,
                    "specular_color": s_color,
                }
                renderer = create_depth_mesh_renderer(
                    cameras, image_size=img_size, light_location=l_location, 
                    device=device, **light_kwargs)
                mc_img = renderer(mc_mesh)[0][..., :3].permute(0, 3, 1, 2)
                mc_img = (mc_img + 1) * (255 / 2)
                mc_img = mc_img.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()

                return mc_img
            
            """
            ambient_color=((0.5, 0.5, 0.5),),
            diffuse_color=((0.3, 0.3, 0.3),),
            specular_color=((0.2, 0.2, 0.2),),
            location=((-0.5, 1, 5),),
            """
            
            ambient_color=((0.3, 0.3, 0.3),)
            diffuse_color=((0.3, 0.3, 0.3),)
            specular_color=((0.4, 0.4, 0.4),)
            loc = np.matmul(torch.eye(3).unsqueeze(0).cpu().numpy()[0].T,np.array([0,0,5]))
            l_location = (totuple(loc),)
            geo_img = gen_img(ambient_color, diffuse_color, specular_color, l_location)
            geo_img = PIL.Image.fromarray(geo_img, 'RGB')
            geo_img = geo_img.resize((512, 1024))
            geo_img.save(f"{outdir}/geo_{seed:04d}_{idx:04d}.png")


def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------