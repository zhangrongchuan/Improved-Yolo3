from torch.utils.data import DataLoader
from dataset import seagull_dataset
from decode import Decode
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import config
import torch
import time

def MSELoss(pred,target):
    return nn.MSELoss(size_average=True)(pred,target)
    
def clip_by_tensor(input,min=1e-7):
    input=input.float()
    result=(input>=min).float()*input+(input<=min).float()*min
    return result
    
def BCELoss(pred,target):
    pred=clip_by_tensor(pred)
    return nn.BCELoss(size_average=True)(pred,target)
    # return -(target * torch.log(pred) + (1.0 - target) * torch.log(1.0 - pred))

def CrossEntropyLoss(pred,target):
    return nn.CrossEntropyLoss(size_average=True)(pred,target)

def filter_pred_bbox(feature1,feature2):
    #用于decode神经网络的输出，使得我们可以得到预测的目标的(x1,y1,x2,y2,confident),所以shape=(n,5)
    Decoder=Decode()
    fea1=Decoder.decode_box(4,feature1)
    fea2=Decoder.decode_box(16,feature2)
    f1_=Decoder.remove_low_confident(fea1,threshold=config.CONF_THRESHOLD_S_OBJ)
    f2_=Decoder.remove_low_confident(fea2,threshold=config.CONF_THRESHOLD_L_OBJ)
    result1=Decoder.change_position_format(f1_)
    result2=Decoder.change_position_format(f2_)
    result1=Decoder.non_maximum_suppression(result1,threshold=config.NMS_THRESHOLD_S_OBJ)
    result2=Decoder.non_maximum_suppression(result2,threshold=config.NMS_THRESHOLD_L_OBJ)
    result=Decoder.concate_three_feature(result1,result2)
    result=Decoder.non_maximum_suppression(result,threshold=0.40)

    return result

def iou(rec1,rec2):
    #计算两个bbox的iou
    w=min(rec1[2],rec2[2])-max(rec1[0],rec2[0])
    h=min(rec1[3],rec2[3])-max(rec1[1],rec2[1])
    intersection=w*h
    area1=(rec1[3]-rec1[1])*(rec1[2]-rec1[0])
    area2=(rec2[3]-rec2[1])*(rec2[2]-rec2[0])
    if w<=0 or h<=0:
        return 0
    else:
        iou=(intersection)/(area2+area1-intersection+1e-7)
        return iou

def judge_correct(res, target, threshold):
    '''
    param
        res:经过nms后所有预测出来的结果(x1,y1,x2,y2,conf)
        target:每张图片所有target的位置(x1,y1,x2,y2)
    return
        tp_label:一个list,分别为confidence,iou,是否是tp(1/0),因此shape=(n*3)
        len(target):每张图片有多少个目标,用于计算整个数据集有多少个target
    '''
    target=target.detach().numpy()[0]
    res=res.detach().numpy()

    tp_label=[ [] for i in range(len(res))] #conf,iou,tp
    for index, i in enumerate(res):
        iou_per_groundtruth=[]#存放每一个pred和所有label的iou
        for j in target:
            Iou=iou(i,j)
            iou_per_groundtruth.append(Iou)
        if max(iou_per_groundtruth)>threshold:
            tp_label[index]=[i[4],max(iou_per_groundtruth),1]
        else:
            tp_label[index]=[i[4],max(iou_per_groundtruth),0]

    return tp_label, len(target)

def get_pr_curve(total_pt_label, total_target_count):
    '''
    这个func用于根据算好的真个数据集的True_Positive(tp)图(conf,iou,tp(1/0)),
    和整个testset的total_target_count来计算pr曲线
    param:
        total_pt_label:整个数据集的tp图(conf,iou,tp(1/0))
        total_target_count:整个数据集的target的总数
    return:
        pr_curve:根据pt计算出来的pr曲线图,shape=(n,3),分别为(conf,precision,recall),并且是按照conf从大到小来排序的
    '''
    '''
    pr曲线的规律:
    当某个bbox预测正确: recall=tp/total,因此recall增加, precision也增加(如果100%就不变)
    当某个bbox预测错误: recall=tp/total,因此recall不变, precision减小
    '''
    total_pt_label=torch.tensor(total_pt_label)
    total_pt_label,_=torch.sort(total_pt_label,dim=0,descending=True)
    pr_curve=[np.array([1,1,0],dtype="float32")]
    correct_num=0
    for i , pred in enumerate(total_pt_label):
        correct_num+=pred[2]
        pred[1],pred[2]= correct_num/(i+1), correct_num/total_target_count  #precision, recall
        pr_curve.append(pred.detach().numpy())
    pr_curve.append(np.array([0,0,1],dtype="float32"))
    
    return pr_curve

def calculate_ap_value(pr_curve):
    '''
    根据pr曲线来计算ap值
    param:
        pr_curve:从func get_pr_curve()中返回的pr曲线图
    return:
        ap的值
    '''
    temp_recall=0
    ap=0
    pr_curve=torch.Tensor(pr_curve)
    #找出precision的最大值
    while(not temp_recall==1):
        index=torch.max(pr_curve,dim=0)[1][1]
        precision,recall=pr_curve[index,1], pr_curve[index,2]
        ap+=precision*(recall-temp_recall)
        temp_recall=recall
        pr_curve=pr_curve[(index+1):]
    
    return ap

def evaluate(data_loader,model,writer,epoch,type_of_eval,threshold=config.AP_IOU_THRESHOLD):
    '''
    type_of_eval:评估的类型,0:train,1:val,2:test
    '''
    data_loader=tqdm(data_loader)
    data_loader.set_description("Evaluate")
    
    total_target_count=0
    total_pt_label=[]

    for data in data_loader:
        image,_,_,truth=data
        image=image.to(config.DEVICE)
        f1,f2=model(image)
        result=filter_pred_bbox(f1,f2)[0]
        
        label, target_num=judge_correct(result,truth,threshold=threshold)
        total_pt_label+=label
        total_target_count+=target_num

        data_loader.update(1)
        time.sleep(0.1)
    pr_curve=get_pr_curve(total_pt_label,total_target_count)
    ap=calculate_ap_value(pr_curve)
    if type_of_eval==0:
        print("The ap of the trainset is "+str(ap)[7:-1])
        writer.add_scalar(tag="ap",scalar_value=ap,global_step=epoch)
    elif type_of_eval==1:
        print("The ap of the valset is "+str(ap)[7:-1])
        writer.add_scalar(tag="test_ap",scalar_value=ap,global_step=epoch)
    elif type_of_eval==2:
        print("The ap of the testset is "+str(ap)[7:-1])
    return ap

def get_dataloader(train_path,val_path,test_path):

    train_set=seagull_dataset(train_path)
    train_loader=DataLoader(dataset=train_set,batch_size=config.BATCH_SIZE,shuffle=True)

    train_set_eval=seagull_dataset(train_path,get_truth=True)
    train_eval_loader=DataLoader(dataset=train_set_eval,batch_size=1,shuffle=True)

    val_set=seagull_dataset(val_path,get_truth=True)
    val_loader=DataLoader(dataset=val_set,batch_size=1,shuffle=True)

    test_set=seagull_dataset(test_path,get_truth=True)
    test_loader=DataLoader(dataset=test_set,batch_size=1,shuffle=True)

    return train_loader,train_eval_loader,val_loader, test_loader
    

def write_logs(writer,x,y,w,h,c_obj,c_noobj,Total_Loss,step):

    writer.add_scalar(tag="loss_x",scalar_value=x, global_step=step)
    writer.add_scalar(tag="loss_y",scalar_value=y, global_step=step)
    writer.add_scalar(tag="loss_w",scalar_value=w, global_step=step)
    writer.add_scalar(tag="loss_h",scalar_value=h, global_step=step)
    writer.add_scalar(tag="loss_c_obj",scalar_value=c_obj, global_step=step)
    writer.add_scalar(tag="loss_c_noobj",scalar_value=c_noobj, global_step=step)
    writer.add_scalar(tag="Total_Loss",scalar_value=Total_Loss,global_step=step)

def get_lr(epoch,pretrained):
    if not pretrained and epoch<config.INITIAL_EPOCH:#5
        lr=1e-04 #0.0001
    elif epoch<15:#10
        lr=1e-05 #0.00001
    elif epoch<120:#90
        lr=1e-06 #0.000001
    elif epoch<150:#40
        lr=1e-07 #0.0000001
    else:#5
        lr=1e-08
    
    return lr
