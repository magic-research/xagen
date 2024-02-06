import numpy as np
import torch
import joblib
from torch_utils import misc
from smplx.SMPLX import SMPLX, LEFT_HAND_INDEX, RIGHT_HAND_INDEX
from training.triplane import face_vertices
from kaolin.ops.mesh import check_sign
from kaolin.metrics.trianglemesh import point_to_mesh_distance


canonical_body_bbox = torch.tensor([[-0.9554, -1.5118, -0.2120], [ 0.9554, 0.7406, 0.2011]]).reshape(1,2,3).float().repeat(1, 1, 1)
canonical_face_bbox = torch.tensor([[-0.2, 0.032, -0.18], [0.2, 0.4833, 0.18]]).reshape(1, 2, 3).float().repeat(1, 1, 1)
canonical_hand_bbox = torch.tensor([[-0.9554, -0.0285, -0.212], [-0.5797,  0.0921,  0.1172]]).reshape(1, 2, 3).float().repeat(1, 1, 1)


@misc.profiled_function
def cal_sdf_batch(verts, faces, triangles, points, device):
    # verts [B, N_vert, 3]
    # faces [B, N_face, 3]
    # triangles [B, N_face, 3, 3]
    # points [B, N_point, 3]
    
    Bsize = points.shape[0]
    residues, _, _ = point_to_mesh_distance(points.contiguous(), triangles)
    pts_dist = torch.sqrt(residues.contiguous())
    pts_signs = -2.0 * (check_sign(verts.cuda(), faces[0].cuda(), points.cuda()).float() - 0.5).to(device) # negative outside
    pts_sdf = (pts_dist * pts_signs).unsqueeze(-1)

    return pts_sdf.view(Bsize, -1, 1)


def export(part="body"):
    RESOLUTION = 128
    device = torch.device("cuda:0")
    body_model = SMPLX().to(device)
    # FLAME_INDEX = np.load("smplx/assets/SMPL-X__FLAME_vertex_ids.npy").tolist()
    # RIGHT_HAND_INDEX = joblib.load('smplx/assets/MANO_SMPLX_vertex_ids.pkl')['right_hand']
    # Define Da-Pose
    pose_canonical = body_model.get_canonical_pose(use_da_pose=True)
    lbs_weights = body_model.lbs_weights
    faces = body_model.faces_tensor
    body_model_params_canonical = {'full_pose': pose_canonical.unsqueeze(0).repeat(1, 1, 1, 1)}
    verts_canonical, _, _, verts_transform_canonical, shape_offsets_canonical, pose_offsets_canonical = body_model(
                                    **body_model_params_canonical, return_transform=True, return_offsets=True)
    triangles = face_vertices(verts_canonical, faces, device)
    # Query SMPL SDF
    indices = torch.linspace(0.5, RESOLUTION - 0.5, RESOLUTION)
    coords_i, coords_j, coords_k = torch.meshgrid([indices, indices, indices], indexing='ij')
    coords = torch.stack([coords_i, coords_j, coords_k], dim=-1).view(-1, 3)
    if part == "body":
        canonical_bbox = canonical_body_bbox
    elif part == "face":
        canonical_bbox = canonical_face_bbox
    elif part == "hand":
        canonical_bbox = canonical_hand_bbox
    normalized_coords = ((coords / RESOLUTION) - 0.5) * (canonical_bbox[:, 1:]-canonical_bbox[:, :1]).squeeze(1) + canonical_bbox.mean(1)
    normalized_coords = normalized_coords.unsqueeze(0).to(device)
    
    sdf = cal_sdf_batch(
        verts_canonical,
        faces.unsqueeze(0).repeat(verts_canonical.size(0), 1, 1), 
        triangles, 
        normalized_coords,
        device)

    sdf = sdf[0].view(RESOLUTION, RESOLUTION, RESOLUTION, 1).permute(3, 2, 1, 0)
    print(f"Exporting voxel grid to smplx/assets/smplx_canonical_{part}_sdf.pkl")
    assert part=="body"
    joblib.dump(sdf.cpu(), f"smplx/assets/smplx_canonical_{part}_sdf.pkl")

if __name__ == "__main__":
    export(part="body")
