NUM_GPUS=${1:-8}
BATCH=${2:-16}

ps aux|grep "multiprocessing"|awk '{print $2}'|xargs kill -9

# Print env variable info
export METIS_WORKER_0_HOST=localhost
echo "master ip: ${METIS_WORKER_0_HOST}"

# NCCL debugging flag on
export OMP_NUM_THREADS=2


PORTS=(`echo $METIS_WORKER_0_PORT | tr ',' ' '`)
# PORT=${PORTS[0]}
PORT=$((12000 + $RANDOM % 20000))
echo "current port: ${PORT}"

DATASET_PATH=deepfashion-pixie-512
METHOD=mpv_smplx
OUTDIR=checkpoints
MBSTD_GROUP=2
DLR=2e-3
GLR=2.5e-3
GAMMA=10
RENDER_SIZE=224
FACE_DISC_WEIGHT=0.25
HAND_DISC_WEIGHT=0.75
EIK_WEIGHT=1e-3
MINSURF_WEIGHT=5e-3
SIGMOID_BETA=3.0
NUM_STEPS=20
NUM_STEPS_HIER=20
DEFORMATION=False
CANONITAL_REG=False
SMPLX_REG_WEIGHT=1.0
BODY_SDF_FROM_OBS=False
REPRESENTATION=plane  # plane, volume, hplane, density
USE_PART_DISC=False  # if use face part disc
USE_POSE_COND=False
SMPLX_REG_FULL_SPACE=True
RAW_SDF=False
NO_SDF_PRIOR=False
DEFORM_REG_WEIGHT=50.0
POSITIONAL_ENCODING=False
SNAP=50
EMBED_FACE_COND=False
CALC_COARSE_EIKONAL=True

DATASET_PATH=/path_to_the_sampled_data/$DATASET_PATH

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=${NUM_GPUS} \
  --master_addr=${METIS_WORKER_0_HOST} \
  --master_port=${PORT} \
    dist_train.py \
    --cfg=$METHOD \
    --data=$DATASET_PATH \
    --gpus=$NUM_GPUS \
    --gen_pose_cond=1 \
    --gamma=$GAMMA \
    --batch=$BATCH \
    --mbstd-group=$MBSTD_GROUP \
    --outdir=$OUTDIR \
    --mirror=0 \
    --density_reg=0 \
    --rebalance=1 \
    --dlr=$DLR \
    --glr=$GLR \
    --face_disc_weight=$FACE_DISC_WEIGHT \
    --hand_disc_weight=$HAND_DISC_WEIGHT \
    --eik_weight=$EIK_WEIGHT \
    --minsurf_weight=$MINSURF_WEIGHT \
    --sigmoid_beta=$SIGMOID_BETA \
    --depth_resolution=$NUM_STEPS \
    --use_deformation=$DEFORMATION \
    --canonical_reg=$CANONITAL_REG \
    --smplx_reg_weight=$SMPLX_REG_WEIGHT \
    --depth_resolution_importance=$NUM_STEPS_HIER \
    --body_sdf_from_obs=$BODY_SDF_FROM_OBS \
    --representation=$REPRESENTATION \
    --part_disc=$USE_PART_DISC \
    --calc_eikonal_coarse=$CALC_COARSE_EIKONAL \
    --use_pose_cond=$USE_POSE_COND \
    --smplx_reg_full_space=$SMPLX_REG_FULL_SPACE \
    --no_sdf_prior=$NO_SDF_PRIOR \
    --raw_sdf=$RAW_SDF \
    --snap=$SNAP \
    --neural_rendering_resolution_initial=64 \
    --neural_rendering_resolution_final=$RENDER_SIZE \
    --deform_reg_weight=$DEFORM_REG_WEIGHT \
    --positional_encoding=$POSITIONAL_ENCODING \
    --embed_face_cond=$EMBED_FACE_COND
    
