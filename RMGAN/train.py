"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/5/19 13:19
"""


import torch
import config
from tqdm import tqdm

import utils
from dataset import MapDataset
from torch.utils.data import DataLoader
from net import Generator,Discriminator
from utils import save_checkpoint,save_some_examples,load_checkpoint

def train_fn(disc,gen,train_loader,opt_disc,opt_gen,L1_LOSS,BCE,epoch):
    loop = tqdm(train_loader,leave=True)
    loop.set_description(desc="training: ")
    all_g_loss = 0
    all_d_loss = 0
    for idx,data in enumerate(loop):
        inputs,target = data
        inputs , target= inputs.to(config.DEVICE),target.to(config.DEVICE)

        #train dsicriminator
        # with torch.cuda.amp.autocast(): x-对应的是卫星拍摄的真实图 y-表示对应卫星拍摄的Google map
        y_fake = gen(inputs)
        D_real = disc(target)
        D_fake = disc(y_fake.detach())
        D_real_loss = BCE(D_real,torch.ones_like(D_real))
        D_fake_loss = BCE(D_fake,torch.zeros_like(D_fake))
        D_loss = (D_real_loss + D_fake_loss) / 2
        all_d_loss += D_loss.item()

        opt_disc.zero_grad()
        D_loss.backward()
        opt_disc.step()

        #train generator
        # with torch.cuda.amp.autocast():
        D_fake = disc(y_fake.detach())
        G_fake_loss = BCE(D_fake,torch.ones_like(D_fake))
        L1 = L1_LOSS(y_fake,target)*config.L1_LAMBDA
        G_loss = G_fake_loss + L1
        all_g_loss += G_loss

        opt_gen.zero_grad()
        G_loss.backward()
        opt_gen.step()

        loop.set_postfix(epoch = epoch)
        if idx % 10 == 0:
            #设置进度条显示的信息，下面表示在显示过程中同时显示损失值
            loop.set_postfix(
               D_loss = D_loss.item(),
               G_loss = G_loss.item()
            )
    return all_d_loss / len(train_loader),all_g_loss / len(train_loader)

def val_fn(disc,gen,val_loader,L1_LOSS,BCE,epoch):
    loop = tqdm(val_loader,leave=True)
    loop.set_description(desc="valing: ")
    for idx,data in enumerate(loop):
        inputs,target = data
        inputs , target= inputs.to(config.DEVICE),target.to(config.DEVICE)

        y_fake = gen(inputs)
        D_fake = disc(y_fake.detach())
        G_fake_loss = BCE(D_fake,torch.ones_like(D_fake))
        L1 = L1_LOSS(y_fake,target)*config.L1_LAMBDA
        G_loss = G_fake_loss + L1

        if idx % 500 == 0:
            #设置进度条显示的信息，下面表示在显示过程中同时显示损失值
            loop.set_postfix(
               epoch = epoch,
               G_loss = G_loss.item()
            )
            save_some_examples(inputs,y_fake, idx,epoch, folder = "images")



def main():
    disc = Discriminator.Discriminator(in_channels=3).to(config.DEVICE)
    print('load discriminator done ...')
    gen = Generator.Generator(in_channles=3).to(config.DEVICE)
    print('load generator done ...')

    opt_disc = torch.optim.Adam(disc.parameters(),lr = config.LEARNING_RATE,betas=(0.5,0.999))
    opt_gen = torch.optim.Adam(gen.parameters(),lr = config.LEARNING_RATE,betas=(0.5,0.999))

    BCE = torch.nn.BCEWithLogitsLoss()
    L1_LOSS = torch.nn.L1Loss()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN,gen,opt_gen,config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC, gen, opt_disc, config.LEARNING_RATE)

    train_dataset = MapDataset(real_dir=config.REAL_TRAIN_DIR,fake_dir=config.FAKE_TRAIN_DIR)
    val_dataset = MapDataset(real_dir=config.REAL_VAL_DIR,fake_dir=config.FAKE_VAL_DIR)

    train_loader = DataLoader(dataset=train_dataset,batch_size=config.BATCH_SIZE,shuffle=True)
    val_loader = DataLoader(dataset=val_dataset,batch_size=4,shuffle=True)
    print('load dataset done ...')
    print('load dataset done ...')

    # g_scaler = torch.cuda.amp.GradScaler()
    # d_scaler = torch.cuda.amp.GradScaler()

    g_loss = []
    d_loss = []
    for epoch in range(config.NUM_EPOCHS):
        gen.train()
        disc.train()
        epoch_d_loss,epoch_g_loss = train_fn(disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE,epoch)
        d_loss.append(epoch_d_loss)
        g_loss.append(epoch_g_loss)

        gen.eval()
        disc.eval()
        with torch.no_grad():
            if config.SAVE_MODEL and epoch % 5 == 0 and epoch > 0:
                save_checkpoint(gen,opt_gen,config.CHECKPOINT_GEN)
                save_checkpoint(disc,opt_disc,config.CHECKPOINT_DISC)
            #进行验证
            val_fn(disc,gen,val_loader,L1_LOSS,BCE,epoch)
    utils.draw(gen_loss=g_loss,disc_loss=d_loss)

if __name__ == '__main__':
    main()