export CUDA_VISIBLE_DEVICES=0

python train.py \
    --epochs 10 \
    --batch_size 4 \
    --lr 1e-5 \
    --seed 1234 \
    --dataset_dir ./dataset/AI_challenge_data \
    --output_dir ./output_directory \
    --early_stop 5 \
    --scheduler linear