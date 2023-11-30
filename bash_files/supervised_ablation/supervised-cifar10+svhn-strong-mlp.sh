python3 main_continual.py \
    --dataset cifar10 \
    --datasets cifar10 svhn \
    --encoder resnet18 \
    --data_dir $DATA_DIR \
    --split_strategy class \
    --max_epochs 100 \
    --num_tasks 2 \
    --task_idx 0 \
    --gpus 0 \
    --num_workers 2 \
    --precision 16 \
    --optimizer sgd \
    --scheduler warmup_cosine \
    --lr 0.1 \
    --classifier_lr 0.1 \
    --weight_decay 1e-4 \
    --batch_size 64 \
    --min_scale 0.6 \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.2 \
    --hue 0.1 \
    --gaussian_prob 0.0 0.0 \
    --solarization_prob 0.0 0.0 \
    --grayscale_prob 0.0 \
    --name supervised-2t-cifar10+svhn-strong-mlp-seed:$SEED \
    --wandb \
    --save_checkpoint \
    --method supervised \
    --mlp-projector mlpp \
    --task_aware_knn \
    --seed $SEED
