from torch import Tensor
from utils import MSELoss
import torch
import config

def loss1(down_sample,predict,target):

    batch_size = predict.shape[0]
    if(down_sample==4):
        anchor=[5/4,5/4]
    elif(down_sample==16):
        anchor=[34/16,26/16]
    
    height=int(256/down_sample)
    width =int(256/down_sample)
    y=torch.linspace(0,height-1,height).repeat(height,1).repeat(1,1).view(height,height).type(Tensor).to(config.DEVICE)
    x=torch.linspace(0,height-1,height).repeat(height,1).t().repeat(1,1).view(height,height).type(Tensor).to(config.DEVICE)

    predict = predict.view(batch_size,5,height,width).permute(0,2,3,1).contiguous()
    total_Loss=0
    target=target.float()

    x1,x2,x3,x4,x5,x6=0,0,0,0,0,0

    for i in range(batch_size):
#---------------------筛选出obj和noobj--------------------------
        no_obj_mask=(target[i,...,4]==0).squeeze()
        obj_mask=(target[i,...,4]==1).squeeze()
        obj_num=torch.sum(obj_mask)
        no_obj_num=torch.sum(no_obj_mask)
#---------------------如果没有obj--------------------------
        if obj_num==0:
            predict_noobj = predict[i][no_obj_mask]
            target_noobj  = target[i][no_obj_mask]
            predict_noobj[...,4]=torch.sigmoid(predict_noobj[...,4]).float()
            loss_c_noobj=MSELoss(predict_noobj[...,4],target_noobj[...,4])
            total_Loss+=loss_c_noobj.sum()/no_obj_num*2
#---------------------如果有obj--------------------------
        else:
            target[i,...,0]-=x.float()
            target[i,...,1]-=y.float()
            target_obj=target[i][obj_mask]
            target_obj[...,2]=torch.log(target_obj[...,2]/anchor[0])
            target_obj[...,3]=torch.log(target_obj[...,3]/anchor[1])
            #--------------------------正样本xywh的loss--------------------------------
            predict_obj=predict[i][obj_mask]
            predict_obj[...,0]=torch.sigmoid(predict_obj[...,0]).float()
            predict_obj[...,1]=torch.sigmoid(predict_obj[...,1]).float()
            predict_obj[...,2]=predict_obj[...,2].float()
            predict_obj[...,3]=predict_obj[...,3].float()
            predict_obj[...,4]=torch.sigmoid(predict_obj[...,4]).float()
            
            loss_x=MSELoss(predict_obj[...,0],target_obj[...,0])
            loss_y=MSELoss(predict_obj[...,1],target_obj[...,1])
            loss_w=MSELoss(predict_obj[...,2],target_obj[...,2])
            loss_h=MSELoss(predict_obj[...,3],target_obj[...,3])
            loss_c=MSELoss(predict_obj[...,4],target_obj[...,4])
            #-------------------------负样本的loss-----------------------------------
            predict_noobj = predict[i][no_obj_mask]
            target_noobj = target[i][no_obj_mask]
            predict_noobj[...,4]=torch.sigmoid(predict_noobj[...,4]).float()

            loss_c_noobj=MSELoss(predict_noobj[...,4],target_noobj[...,4])

            #--------------------------------------------------------------------------------------------------
            total_Loss+=(loss_x.sum() + loss_y.sum() + (loss_c.sum()/obj_num + loss_c_noobj.sum()/no_obj_num)*2 + loss_w.sum() + loss_h.sum())
        #---------------------------------------------------------------------------------------------------
            x1+=loss_x.sum()
            x2+=loss_y.sum()
            x3+=loss_w.sum()
            x4+=loss_h.sum()
            x5+=loss_c.sum()/obj_num
        x6+=loss_c_noobj.sum()/no_obj_num
    
    return total_Loss,x1,x2,x3,x4,x5,x6

def loss2(down_sample,predict,target):

    '''
    loss_a的lr从0.00001开始/0.000001
    其他的从0.001/0.0001开始
    '''
    batch_size = predict.shape[0]
    if(down_sample==4):
        anchor=[5/4,5/4]
    elif(down_sample==16):
        anchor=[34/16,26/16]
    
    height=int(256/down_sample)
    width =int(256/down_sample)
    y=torch.linspace(0,height-1,height).repeat(height,1).repeat(1,1).view(height,height).type(Tensor).to(config.DEVICE)
    x=torch.linspace(0,height-1,height).repeat(height,1).t().repeat(1,1).view(height,height).type(Tensor).to(config.DEVICE)

    predict = predict.view(batch_size,5,height,width).permute(0,2,3,1).contiguous()
    total_Loss=0
    target=target.float()

    x1,x2,x3,x4,x5,x6=0,0,0,0,0,0

    for i in range(batch_size):

        no_obj_mask=(target[i,...,4]==0).squeeze()
        obj_mask=(target[i,...,4]==1).squeeze()
        obj_num=torch.sum(obj_mask)
        no_obj_num=torch.sum(no_obj_mask)

        if obj_num==0:
            predict_noobj = predict[i][no_obj_mask]
            predict_noobj[...,4]=torch.sigmoid(predict_noobj[...,4]).float()
            loss_c_noobj=-torch.log(1-predict_noobj[...,4])
            total_Loss+=loss_c_noobj.sum()/no_obj_num
        else:
            target[i,...,0]-=x.float()
            target[i,...,1]-=y.float()
            target_obj=target[i][obj_mask]
            target_obj[...,2]=torch.log(target_obj[...,2]/anchor[0])
            target_obj[...,3]=torch.log(target_obj[...,3]/anchor[1])
            #--------------------------正样本xywh的loss--------------------------------
            predict_obj=predict[i][obj_mask]
            predict_obj[...,0]=torch.sigmoid(predict_obj[...,0]).float()
            predict_obj[...,1]=torch.sigmoid(predict_obj[...,1]).float()
            predict_obj[...,2]=predict_obj[...,2].float()
            predict_obj[...,3]=predict_obj[...,3].float()
            predict_obj[...,4]=torch.sigmoid(predict_obj[...,4]).float()
            
            loss_x=MSELoss(predict_obj[...,0],target_obj[...,0])
            loss_y=MSELoss(predict_obj[...,1],target_obj[...,1])
            loss_w=MSELoss(predict_obj[...,2],target_obj[...,2])
            loss_h=MSELoss(predict_obj[...,3],target_obj[...,3])
            loss_c=-torch.log(predict_obj[...,4])
            #-------------------------负样本的loss-----------------------------------
            predict_noobj = predict[i][no_obj_mask]
            predict_noobj[...,4]=torch.sigmoid(predict_noobj[...,4]).float()

            loss_c_noobj=-torch.log(1-predict_noobj[...,4])

            #--------------------------------------------------------------------------------------------------
            total_Loss+=(loss_x.sum() + loss_y.sum() + loss_c.sum()*2/obj_num + loss_c_noobj.sum()/no_obj_num + loss_w.sum() + loss_h.sum())
        #---------------------------------------------------------------------------------------------------
            x1+=loss_x.sum()
            x2+=loss_y.sum()
            x3+=loss_w.sum()
            x4+=loss_h.sum()
            x5+=loss_c.sum()/obj_num
        x6+=loss_c_noobj.sum()/no_obj_num
    
    return total_Loss,x1,x2,x3,x4,x5,x6

def loss(f1,l1,f2,l2,pretrained,epoch):
    if not pretrained and epoch<config.INITIAL_EPOCH:
        Total_Loss1,x1,y1,w1,h1,c_obj1,c_noobj1=loss1(4,f1,l1)
        Total_Loss2,x2,y2,w2,h2,c_obj2,c_noobj2=loss1(16,f2,l2)
        Total_Loss,x,y,w,h,c_obj,c_noobj=Total_Loss1+Total_Loss2,x1+x2,y1+y2,w1+w2,h1+h2,c_obj1+c_obj2,c_noobj1+c_noobj2
    else:
        Total_Loss1,x1,y1,w1,h1,c_obj1,c_noobj1=loss2(4,f1,l1)
        Total_Loss2,x2,y2,w2,h2,c_obj2,c_noobj2=loss2(16,f2,l2)
        Total_Loss,x,y,w,h,c_obj,c_noobj=Total_Loss1+Total_Loss2,x1+x2,y1+y2,w1+w2,h1+h2,c_obj1+c_obj2,c_noobj1+c_noobj2
        
    return Total_Loss,x.data,y.data,w.data,h.data,c_obj.data,c_noobj.data
