import torch
from torchvision import transforms
DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#DEVICE=torch.device("cpu")

WEIGHT_RESTORE_PATH="w/weight2.pth"#"../autodl-tmp/weight79.pth" #"w/para.pth"
WEIGHT_SAVE_PATH="../autodl-tmp/weight"
LOG_PATH="logs"

TRAIN_SET_PATH = "train.txt"
TRAIN_SET_VAL_PATH = "trainval.txt"
VAL_SET_PATH   = "val.txt"
TEST_SET_PATH  = "test.txt"

PRE_TRAINED=True #True

AP_IOU_THRESHOLD=0.5 #AP50=0.5  AP75=0.75
CONF_THRESHOLD_L_OBJ=0.90
CONF_THRESHOLD_S_OBJ=0.96
NMS_THRESHOLD_L_OBJ=0.50
NMS_THRESHOLD_S_OBJ=0.00

BATCH_SIZE=8
IMAGE_SIZE=256
NUM_EPOCH=100
CURRENT_STEPS=0 #0
VAL_STEP=0 #0
INITIAL_EPOCH=0 #5

transform=transforms.Compose(
[transforms.Resize([256,256]),
transforms.GaussianBlur(kernel_size=3),
#transforms.RandomHorizontalFlip(p=0.5),
transforms.ColorJitter(brightness=1,contrast=1,saturation=1,hue=0.5),
#transforms.RandomRotation(90),
transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
])
