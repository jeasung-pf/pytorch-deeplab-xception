CUDA_VISIBLE_DEVICES=0,1,2,3;
python serve.py \
  --backbone resnet \
  --lr 0.007 \
  --workers 4 \
  --epochs 50 \
  --batch-size 16 \
  --gpu-ids 0,1,2,3 \
  --resume /dltraining/checkpoints/pascal/deeplab-resnet/experiment_7/ \
  --checkname deeplab-resnet \
  --eval-interval 1 \
  --dataset pascal
