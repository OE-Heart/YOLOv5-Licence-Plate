export CUDA_VISIBLE_DEVICES=6,7

python train.py \
--img 640 \
--batch 16 \
--epochs 300 \
--data data/ccpd.yaml \
--cfg models/yolov5s.yaml \
--weights weights/yolov5s.pt \
--hyp data/hyps/hyp.scratch-low.yaml \
--optimizer AdamW \
--evolve