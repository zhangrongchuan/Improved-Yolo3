from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from torch import Tensor
from PIL import Image
import config
import torch
class seagull_dataset(Dataset):
    def __init__(self,train_path,get_truth=False,transform=False):
        image_path=[]
        self.transform=transform
        self.get_truth=get_truth
        f = open(train_path,"r")
        for line in f:
            image_path.append(line)
        f.close()
        self.image_path=image_path
    
    def __getitem__(self, index):

        all_obj=[]
        get_truth=[]
        line=self.image_path[index]
        obj=[]
        line=[i for i in line.split(" ")]
#---------------------------------------transforms---------------------------------------------
        single_image=Image.open(line[0])
        image=transforms.ToTensor()(single_image).to(config.DEVICE)
#----------------------------------------------------------------------------------------------
        single_label=line[3:]
        single_label.append(single_label.pop()[:-1])
        object_number=int(len(single_label)/5)
#----------------------------------------------------------------------------------------------
        x=torch.linspace(0,64-1,64).repeat(64,1).t().repeat(1,1).view(64,64).type(Tensor).to(config.DEVICE)
        y=torch.linspace(0,64-1,64).repeat(64,1).repeat(1,1).view(64,64).type(Tensor).to(config.DEVICE)
        downSampling_4=torch.zeros([64,64,5]).to(config.DEVICE)
        downSampling_4[...,0]=x
        downSampling_4[...,1]=y

        x=torch.linspace(0,16-1,16).repeat(16,1).t().repeat(1,1).view(16,16).type(Tensor).to(config.DEVICE)
        y=torch.linspace(0,16-1,16).repeat(16,1).repeat(1,1).view(16,16).type(Tensor).to(config.DEVICE)
        downSampling_16=torch.zeros([16,16,5]).to(config.DEVICE)
        downSampling_16[...,0]=x
        downSampling_16[...,1]=y

        for _ in range(object_number):
            y2=int(single_label.pop())
            x2=int(single_label.pop())
            y1=int(single_label.pop())
            x1=int(single_label.pop())
            single_label.pop()
            obj.append([x1,y1,x2,y2])
            x=(x1+x2)/2
            y=(y1+y2)/2
            w=x2-x1
            h=y2-y1
            
            if w*h<256:
                downSampling_4[int(x/4),int(y/4),0]=x/4 
                downSampling_4[int(x/4),int(y/4),1]=y/4
                downSampling_4[int(x/4),int(y/4),2]=w/4
                downSampling_4[int(x/4),int(y/4),3]=h/4
                downSampling_4[int(x/4),int(y/4),4]=1
            else:
                downSampling_16[int(x/16),int(y/16),0]=x/16
                downSampling_16[int(x/16),int(y/16),1]=y/16
                downSampling_16[int(x/16),int(y/16),2]=w/16
                downSampling_16[int(x/16),int(y/16),3]=h/16
                downSampling_16[int(x/16),int(y/16),4]=1       
                
        all_obj=torch.tensor(obj)
        label1=downSampling_4#大scale预测小物体
        label2=downSampling_16#中scale预测中物体

        self.all_obj=all_obj

        if self.get_truth:
            return image, label1, label2, all_obj
        else:
            if self.transform:
                image=config.transform(image)
            return image, label1, label2
    def __len__(self):
        return len(self.image_path)
dataset=seagull_dataset("val.txt",get_truth=True)
dataset=DataLoader(dataset=dataset,batch_size=2)
for i in dataset:
    break
