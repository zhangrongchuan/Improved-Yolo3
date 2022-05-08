from torch.utils.data import Dataset
from torchvision import transforms
from torch import Tensor
from PIL import Image
import numpy as np
import torch
class seagull_dataset(Dataset):
    def __init__(self,train_path,get_truth=False,transforms=transforms.ToTensor()):
        image=[]
        label1=[]
        label2=[]
        all_obj=[]
        f = open(train_path,"r")
        for line in f:
            obj=[]
            line=[i for i in line.split(" ")]
#---------------------------------------transforms---------------------------------------------
            single_image=Image.open(line[0])
            single_image=transforms(single_image)
            image.append(single_image)
#----------------------------------------------------------------------------------------------
            single_label=line[3:]
            single_label.append(single_label.pop()[:-1])
            object_number=int(len(single_label)/5)
#----------------------------------------------------------------------------------------------
            x=torch.linspace(0,64-1,64).repeat(64,1).t().repeat(1,1).view(64,64).type(Tensor)
            y=torch.linspace(0,64-1,64).repeat(64,1).repeat(1,1).view(64,64).type(Tensor)
            downSampling_4=np.zeros([64,64,5])
            downSampling_4[...,0]=x
            downSampling_4[...,1]=y

            x=torch.linspace(0,16-1,16).repeat(16,1).t().repeat(1,1).view(16,16).type(Tensor)
            y=torch.linspace(0,16-1,16).repeat(16,1).repeat(1,1).view(16,16).type(Tensor)
            downSampling_16=np.zeros([16,16,5])
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
                    
            obj=torch.tensor(obj)
            all_obj.append(obj)
            label1.append(downSampling_4)
            label2.append(downSampling_16)

        f.close()
        self.image=image
        self.label1=label1#大scale预测小物体
        self.label2=label2#中scale预测中物体

        self.all_obj=all_obj
        self.get_truth=get_truth
    
    def __getitem__(self, index):
        if self.get_truth:
            return self.image[index], self.label1[index], self.label2[index], self.all_obj[index]
        else:
            return self.image[index], self.label1[index], self.label2[index]
    def __len__(self):
        return len(self.image)