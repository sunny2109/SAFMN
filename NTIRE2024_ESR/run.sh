# Evaluation on LSDIR_DIV2K_valid datasets:
CUDA_VISIBLE_DEVICES=0 python test_demo.py \
    --data_dir [path to your data dir] \
    --save_dir [path to your save dir] \
    --model_id 0

# When Test datasets are included 
# CUDA_VISIBLE_DEVICES=0 python test_demo.py \
#     --data_dir ../ \
#     --save_dir ../results \
#     --include_test \
#     --model_id 0
