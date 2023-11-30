python3 main_continual.py \
    --dataset cifar10 \
    --encoder resnet18 \
    --data_dir $DATA_DIR \
    --split_strategy class \
    --task_idx 0 \
    --max_epochs 500 \
    --num_tasks 5 \
    --gpus 0 \
    --precision 16 \
    --optimizer sgd \
    --scheduler warmup_cosine \
    --lr 0.4 \
    --classifier_lr 0.1 \
    --weight_decay 1e-5 \
    --batch_size 256 \
    --num_workers 4 \
    --brightness 0.8 \
    --contrast 0.8 \
    --saturation 0.8 \
    --hue 0.2 \
    --gaussian_prob 0.0 0.0 \
    --name supcon-trex-proj-c10x5-cassle-seed:$SEED \
    --project cassle \
    --wandb \
    --save_checkpoint \
    --method simclr \
    --supervised \
    --temperature 0.2 \
    --proj_hidden_dim 2048 \
    --output_dim 256 \
    --mlp-projector trex \
    --distiller contrastive \
    --task_aware_knn \
    --seed $SEED
