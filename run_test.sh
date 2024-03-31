export FLAME_PATH=/home/liam-schoneveld/gaussian_avatars_wrk/flame_model/assets/flame
# export FLAME_PATH=/home/user/code/nerf/flame/assets/flame

DATA_PATH=$1
OUTFOLDER=$2
GPUID=$3

BASIS_NUM=46

CUDA_VISIBLE_DEVICES=$GPUID python run_nerfblendshape.py\
    --data_folder $DATA_PATH \
    --workspace $OUTFOLDER \
    --fp16 --tcnn  --cuda_ray --basis_num $BASIS_NUM   --add_mean  --use_lpips   --mode test --to_mem \
    --neck_pose_to_expr --eye_pose_to_expr
