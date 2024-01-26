export FLAME_PATH=/home/user/code/nerf/flame/assets/flame

DATA_PATH=$1
OUTFOLDER=$2
GPUID=$3

BASIS_NUM=46

python create_nersemble_transforms_json.py --dataset_path $DATA_PATH
TEST_START=$(cat /tmp/len_val)

# echo "BASIS_NUM = $BASIS_NUM"
echo "TEST_START = $TEST_START"

python get_max.py --path $DATA_PATH/transforms_nb_train.json --test_start $TEST_START --num $BASIS_NUM

CUDA_VISIBLE_DEVICES=$GPUID python run_nerfblendshape.py\
    --data_folder $DATA_PATH
    --workspace /data/nerfblendshape/trial_nerfblendshape/$OUTFOLDER\
    --fp16 --tcnn  --cuda_ray --basis_num $BASIS_NUM   --add_mean  --use_lpips   --mode train --to_mem
