import os

for alpha in [-0.5, 0, 0.1, 0.2, 0.5, 1, 5]:
    os.system(f"python pruning_train.py ./data/imagenet -a resnet18 \
                --save_dir ./alpha/{alpha}/resnet18-rate-0.7-mutual --prune_rate 0.3  --workers 8 \
                --alpha {alpha} --two-crop  --loss-type ce+kl")
