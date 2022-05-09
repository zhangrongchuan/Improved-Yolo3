from torch import nn
from torch import Tensor
import config
from torchvision.ops import nms
import torch
class Decode(nn.Module):
    def __init__(self):
        super(Decode,self).__init__()
        self.box_par=5
    def decode_box(self,down_sample,input):
        if(down_sample==4):
            anchor=[5/4,5/4]
        elif(down_sample==16):
            anchor=[34/16,26/16]

        output=[]
        self.batch_size=config.BATCH_SIZE
        input_height = input.shape[2]
        input_width  = input.shape[3]
        scale=config.IMAGE_SIZE/input_height
             
        for _, input in enumerate(input):
            prediction = input.view(self.box_par,input_height,input_width).permute(1,2,0).contiguous()
            # #先验证框的中心位置调整
            x=torch.sigmoid(prediction[...,0])
            y=torch.sigmoid(prediction[...,1])
            w=torch.exp(prediction[...,2])*anchor[0]
            h=torch.exp(prediction[...,3])*anchor[1]
            p=torch.sigmoid(prediction[...,4])

            #计算调整以后先验框的位置
            grid_y=torch.linspace(0,input_height-1,input_width).repeat(input_height,1).repeat(1,1).view(input_height,input_width).type(Tensor).to(config.DEVICE)
            grid_x=torch.linspace(0,input_width-1,input_height).repeat(input_height,1).t().repeat(1,1).view(input_height,input_width).type(Tensor).to(config.DEVICE)

            pred_boxs=Tensor(prediction.shape)
            pred_boxs[...,0]= (grid_x + x.data)*scale
            pred_boxs[...,1]= (grid_y + y.data)*scale
            
            pred_boxs[...,2]= w.data*scale
            pred_boxs[...,3]= h.data*scale
            pred_boxs[...,4]= p
            output.append(pred_boxs)
        return output

    def remove_low_confident(self,input,threshold=0.80):
        output=[]
        for i, input in enumerate(input):
            input_mask=(input[...,4]>threshold).squeeze()
            out=input[input_mask]
            output.append(out)
        return output
        
    def concate_three_feature(self,feature1,feature2):
         batch_size=len(feature1)
         output=[]
         for i in range(batch_size):
             output.append(torch.cat([feature1[i],feature2[i]],0))
         return output

    def change_position_format(self,input):
        output=[]
        for i, input in enumerate(input):
            prediction=torch.zeros(input[...,:5].shape)
            prediction[:,0]=input[:,0]-input[:,2]/2
            prediction[:,1]=input[:,1]-input[:,3]/2
            prediction[:,2]=input[:,0]+input[:,2]/2
            prediction[:,3]=input[:,1]+input[:,3]/2
            prediction[:,4]=input[:,4]
            c=prediction.detach().numpy()
            output.append(prediction)
        return output
            
    def non_maximum_suppression(self,input,threshold=0):
        output=[]
        for i, input in enumerate(input):
            keep=nms(input[...,:4],input[...,4],threshold)
            output.append(input[keep])
        return output
        
