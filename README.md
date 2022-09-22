# AdvSemiSeg-MA
1、Environment configuration
ubuntu,pytorch=1.0,python=3.6,python-opencv >=3.4.0
refer to：https://github.com/hfslyc/AdvSemiSeg


2、Steps to reproduce：
model training：
python train.py --snapshot-dir snapshots \
                --partial-data 0.125 \
                --num-steps 20000 \
                --lambda-adv-pred 0.01 \
                --lambda-semi 0.1 --semi-start 5000 --mask-T 0.2
Model prediction：
python evaluate_voc.py --restore-from snapshots/VOC_20000.pth \
                       --save-dir results

3、Dataset download：
「AdvSemiSeg-MA」https://www.aliyundrive.com/s/Lqhms7FWqws 提取码: 54ff
