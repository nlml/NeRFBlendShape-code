export FLAME_PATH=/home/liam-schoneveld/gaussian_avatars_wrk/flame_model/assets/flame
DATA_PATH=./dataset

NAME=$1
GPUID=$2
BASIS_NUM=46

python create_nersemble_transforms_json.py --dataset_path $DATA_PATH/$NAME
TEST_START=$(cat /tmp/len_val)

# echo "BASIS_NUM = $BASIS_NUM"
echo "TEST_START = $TEST_START"

python get_max.py --path $DATA_PATH/$NAME/transforms.json --test_start $TEST_START --num $BASIS_NUM

CUDA_VISIBLE_DEVICES=$GPUID python run_nerfblendshape.py\
    --img_idpath $DATA_PATH/$NAME/transforms.json \
    --exp_idpath $DATA_PATH/$NAME/transforms.json \
    --pose_idpath $DATA_PATH/$NAME/transforms.json \
    --intr_idpath $DATA_PATH/$NAME/transforms.json \
    --workspace trial_nerfblendshape_$NAME\
    --test_start $TEST_START\
    --fp16 --tcnn  --cuda_ray --basis_num $BASIS_NUM   --add_mean  --use_lpips   --mode train --to_mem
