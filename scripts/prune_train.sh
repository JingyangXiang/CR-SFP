# resnet18+ce+kl
python pruning_train.py ./dataset/imagenet -a resnet18 --save_dir ./two_crop/resnet18-rate-0.3-mutual \
        --prune-rate 0.3  --workers 8 --alpha 1 --two-crop --prune-criterion l2 --loss-type ce+kl

# resnet34+ce+kl
python pruning_train.py ./dataset/imagenet -a resnet34 --save_dir ./two_crop/resnet34-rate-0.3-mutual \
        --prune-rate 0.3  --workers 8 --alpha 1 --two-crop --prune-criterion l2 --loss-type ce+kl

# resnet50+ce+kl
python pruning_train.py ./dataset/imagenet -a resnet50 --save_dir ./two_crop/resnet50-rate-0.3-mutual \
        --prune-rate 0.3  --workers 8 --alpha 1 --two-crop --prune-criterion l2 --loss-type ce+kl

# mobilenetv1+ce+kl
python pruning_train.py ./dataset/imagenet -a mobilenet_v1 --save_dir ./two_crop/mobilenet_v1-rate-0.34375-mutual \
        --prune-rate 0.34375  --workers 8 --alpha 1 --two-crop --prune-criterion l2 --loss-type ce+kl

# resnet18+ce+cos
python pruning_train.py ./dataset/imagenet -a resnet18 --save_dir ./two_crop/resnet18-rate-0.3-mutual \
        --prune-rate 0.3  --workers 8 --alpha 1 --two-crop --prune-criterion l2 --loss-type ce+cos

# resnet18+ce
python pruning_train.py ./dataset/imagenet -a resnet18 --save_dir ./two_crop/resnet18-rate-0.3-mutual \
        --prune-rate 0.3  --workers 8 --alpha 0. --two-crop --prune-criterion l2 --loss-type ce