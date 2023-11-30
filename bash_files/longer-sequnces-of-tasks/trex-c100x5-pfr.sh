python3 main_continual.py \
    --dataset cifar100 \
    --encoder resnet18 \
    --data_dir $DATA_DIR \
    --split_strategy class \
    --max_epochs 100 \
    --num_tasks 5 \
    --task_idx 0 \
    --gpus 0 \
    --num_workers 4 \
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
    --check_val_every_n_epoch 20 \
    --name trex-c100x5-pfr-lamb:$DSTL_LAMB-seed:$SEED \
    --wandb \
    --save_checkpoint \
    --method trex \
    --mlp-projector trex \
    --task_aware_knn \
    --distiller pfr \
    --distill_lamb $DSTL_LAMB \
    --distill_proj_hidden_dim 256 \
    --seed $SEED
