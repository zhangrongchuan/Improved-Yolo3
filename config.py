import torch

#DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE=torch.device("cpu")

WEIGHT_RESTORE_PATH="weight.pkl"
WEIGHT_SAVE_PATH="weight.pkl"
LOG_PATH="logs"

TRAIN_SET_PATH="train.txt"
VAL_SET_PATH="val.txt"
TEST_SET_PATH="val.txt"

PRE_TRAINED=False #False

AP_IOU_THRESHOLD=0.5 #AP50=0.5  AP75=0.75
CONF_THRESHOLD_L_OBJ=0.60
CONF_THRESHOLD_S_OBJ=0.96
NMS_THRESHOLD_L_OBJ=0.50
NMS_THRESHOLD_S_OBJ=0.00

BATCH_SIZE=4
IMAGE_SIZE=256
NUM_EPOCH=150
CURRENT_STEPS=0 #0
INITIAL_EPOCH=5 #5

TRANSFORMS=''
