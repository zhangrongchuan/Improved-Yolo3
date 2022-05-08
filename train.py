from utils import write_logs, evaluate, get_lr,get_dataloader
from torch.utils.tensorboard import SummaryWriter
from model import DarkNet
from tqdm import tqdm
from loss import loss
import torch.optim
import config
import time

def main():

    train_loader, train_val_loader, val_loader, test_loader=get_dataloader(config.TRAIN_SET_PATH,config.VAL_SET_PATH,config.TEST_SET_PATH)
    if config.PRE_TRAINED:
        model=torch.load(config.WEIGHT_RESTORE_PATH).to(config.DEVICE)
    else:
        model=DarkNet().to(config.DEVICE)

    model.train().to(config.DEVICE)
    model.eval().to(config.DEVICE)
    writer=SummaryWriter(config.LOG_PATH)

    for epoch in range(config.NUM_EPOCH):
        train_loader=tqdm(train_loader)
        train_loader.set_description("epoch "+str(epoch+1)+" train")
        optimizer=torch.optim.Adam(model.parameters(),lr=get_lr(epoch,config.PRE_TRAINED))
        #optimizer=torch.optim.Adam(model.parameters(),lr=0.000001)
        for data in train_loader:
            image,l1,l2=data
            image,l1,l2=image.to(config.DEVICE),l1.to(config.DEVICE),l2.to(config.DEVICE)
            f1,f2=model(image)

            Total_Loss,x,y,w,h,c_obj,c_noobj=loss(f1,l1,f2,l2,config.PRE_TRAINED,epoch)

            write_logs(writer,x,y,w,h,c_obj,c_noobj,Total_Loss,config.CURRENT_STEPS)
            config.CURRENT_STEPS+=1
            train_loader.set_postfix(Total_loss=Total_Loss.data,loss_x=x,loss_y=y,loss_w=w,loss_h=h,loss_obj=c_obj,loss_noobj=c_noobj)
            time.sleep(0.1)
            train_loader.update(1)

            optimizer.zero_grad()
            Total_Loss.backward()
            optimizer.step()
        
        evaluate(train_val_loader,model,writer,epoch,0)# trainset eval
        evaluate(val_loader,model,writer,epoch,1)# valset eval

        torch.save(model,config.WEIGHT_SAVE_PATH)

    evaluate(test_loader,model,writer,config.NUM_EPOCH,2)# testset eval

if __name__ == "__main__":
    main()