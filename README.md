# TH1:
## python train.py --epoch 10 --mode pretrain --saveas "path/to/saveas" --data "path/to/dataset" --lr0 0.005 --batchsize 8 --img_size 512 512 --numclass 1 --weight_decay 0.5 --loss BCEDice_loss --checkpoint "path/to/checkpoint" 
# TH2:
## python train.py --epoch 10 --mode train --saveas "path/to/saveas" --data "path/to/dataset" --lr0 0.005 --batchsize 8 --img_size 512 512 --numclass 1 --weight_decay 0.5 --loss BCEDice_loss
