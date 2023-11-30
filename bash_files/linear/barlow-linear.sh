python3 main_linear.py \
    --dataset imagenet100 \
    --encoder resnet18 \
    --data_dir $DATA_DIR \
    --train_dir IN-100/train \
    --val_dir IN-100/val \
    --split_strategy class \
    --num_tasks 1 \
    --max_epochs 100 \
    --gpus 0 \
    --precision 16 \
    --optimizer sgd \
    --scheduler step \
    --lr 1.0 \
    --lr_decay_steps 60 80 \
    --weight_decay 0 \
    --batch_size 256 \
    --enable_knn_eval \
    --num_workers 8 \
    --dali \
    --name $EXP_NAME \
    --pretrained_feature_extractor $PRETRAINED_FEATURE_EXTRACTOR \
    --project cassle \
    --wandb \
    --save_checkpoint