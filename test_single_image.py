#detect single image
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import seagull_dataset
from utils import filter_pred_bbox
from PIL import ImageDraw
from model import DarkNet
import torch
import config

def detect_single_image():
    convert_PIL=transforms.ToPILImage()
    test_set=seagull_dataset("train.txt",get_truth=True)
    test_loader=DataLoader(dataset=test_set,batch_size=1,shuffle=False)
    model=DarkNet().to(config.DEVICE)
    model.load_state_dict(torch.load(config.WEIGHT_RESTORE_PATH))
    model.train().to(config.DEVICE)
    model.eval().to(config.DEVICE)
    for data in test_loader:
        image,_,_,truth=data
        image=image.to(config.DEVICE)
        f1,f2=model(image)
        result=filter_pred_bbox(f1,f2)[0]

        image=torch.reshape(image,[3,256,256])
        image=convert_PIL(image)
        draw=ImageDraw.Draw(image)
        
        for i in result:
            box=[i[0],i[1],i[2],i[3]]
            draw.rectangle(box,outline="orange",width=2)
            print(i)

        for _ , i in enumerate(truth[0]):
            box=[i[0],i[1],i[2],i[3]]
            draw.rectangle(box,outline="green",width=2)
            #print(box)
            
        image.show()
        break

if __name__=="__main__":
    detect_single_image()
