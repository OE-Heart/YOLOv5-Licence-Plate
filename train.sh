export CUDA_VISIBLE_DEVICES=2,3,4,5

python train.py \
--img 640 \
--batch 16 \
--epochs 15 \
--data data/ccpd.yaml \
--cfg models/yolov5s.yaml \
--weights weights/yolov5s.pt \
--hyp data/hyps/hyp.test.yaml \
--optimizer SGD \
--save-period 1