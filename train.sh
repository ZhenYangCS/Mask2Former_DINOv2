python train_net.py --num-gpus 8 \
  --config-file configs/coco/instance-segmentation/dinov2/dinov2_vit_base_bs.yaml \
  --resume


# 使用num-gpus控制GPU的数量

# hfai python train_net.py --num-gpus 3 --config-file configs/coco/instance-segmentation/dinov2/dinov2_vit_base_bs.yaml --resume -- --nodes 3 --name test
# hfai bash train.sh -- --nodes 8 --name test