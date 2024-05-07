import torch
from utils import *
from models import *
import time
from torch import nn
import numpy as np
import os
from tqdm import tqdm

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

        
def get_gen_loss(crit_fake_pred):
    gen_loss = -1. * torch.mean(crit_fake_pred)
    return gen_loss


def get_crit_loss(crit_fake_pred, crit_real_pred):

    crit_loss =  (-1* torch.mean(crit_fake_pred)) - (-1* torch.mean(crit_real_pred))
    return crit_loss


def initializeClinicalGAN(input_dim, output_dim, hid_dim,pf_dim,gen_layers,gen_heads,dis_heads,dis_layers,dropout,lr,n_epochs,alpha,clip,loader,data,config,path,gen_clip ,device):
    
    # get the data

    inp_max_len,out_max_len,pad_idx = data['maxInp'],data['maxOut'] , data['codeMap']['types']['PAD']
    enc = Encoder(input_dim,hid_dim,gen_layers,gen_heads,pf_dim,dropout,inp_max_len).to(device)
    dec = Decoder(output_dim,hid_dim,gen_layers,gen_heads,pf_dim,dropout,out_max_len).to(device)
    
    gen = Generator(enc, dec, pad_idx, pad_idx).to(device)
    disc = Discriminator(input_dim, hid_dim, dis_layers, dis_heads, pf_dim, dropout,pad_idx,inp_max_len+out_max_len).to(device)

    gen_opt = torch.optim.Adam(gen.parameters(), lr = lr)
    disc_opt = torch.optim.SGD(disc.parameters(), lr = lr)
    
    lr_schedulerG = NoamLR(gen_opt, warmup_steps=config['warmup_steps'],factor= config['factor'],model_size=config['hid_dim'])
    lr_schedulerD = NoamLR(disc_opt, warmup_steps=config['warmup_steps'],factor= config['factor'],model_size=config['hid_dim'])

    gen.apply(initialize_weights)
    disc.apply(initialize_weights)
    
    criterion = LabelSmoothingCrossEntropy(epsilon = config['eps'], ignore_index = pad_idx)

    
    modelHypermaters = {}

    
    modelHypermaters['trainloader']=loader['trainLoader']
    modelHypermaters['testloader']=loader['testLoader']
    modelHypermaters['valloader']=loader['valLoader']
    
    modelHypermaters['encObject']=enc
    modelHypermaters['decObject']=dec
    modelHypermaters['optimizer_D']=disc_opt
    modelHypermaters['optimizer_G']=gen_opt
    modelHypermaters['genObject']=gen
    modelHypermaters['discObject']=disc
    # to access the mask function inside the generator class
    modelHypermaters['criterion']=criterion
    
    modelHypermaters['device']=device
    modelHypermaters['config']=config
    
    modelHypermaters['n_epochs']=n_epochs
    modelHypermaters['alpha']=alpha
    modelHypermaters['clip']=clip
    
    modelHypermaters['reverseOutTypes'] = data['codeMap']['reverseOutTypes'] 
    modelHypermaters['types'] = data['codeMap']['types'] 
    
    modelHypermaters['path'] = path
    modelHypermaters['gen_clip'] =gen_clip
    modelHypermaters['lr_schedulerG']=lr_schedulerG
    modelHypermaters['lr_schedulerD']=lr_schedulerD
    return modelHypermaters
    
    
def trainCGAN(modelHypermaters):
    
    
    trainLoader = modelHypermaters['trainloader']
    testLoader= modelHypermaters['testloader']
    valLoader= modelHypermaters['valloader']
    
    enc= modelHypermaters['encObject']
    dec= modelHypermaters['decObject']
    disc_opt =modelHypermaters['optimizer_D']
    gen_opt =modelHypermaters['optimizer_G']
    gen= modelHypermaters['genObject']
    disc= modelHypermaters['discObject']
    
    criterion= modelHypermaters['criterion']
    
    device = modelHypermaters['device']
    config= modelHypermaters['config']
     
    n_epochs = modelHypermaters['n_epochs']
    alpha = modelHypermaters['alpha']
    clip = modelHypermaters['clip']
    reverseOutTypes = modelHypermaters['reverseOutTypes']
    gen_clip= modelHypermaters['gen_clip']
    lr_schedulerG = modelHypermaters['lr_schedulerG']
    lr_schedulerD = modelHypermaters['lr_schedulerD']

    
    
    crit_repeats =5
    vLoss = []
    tLoss = []
    for epoch in range(n_epochs):
        totalGen = 0
        totalDis = 0
        gen.train()
        disc.train()
        lr_schedulerG.step()
        lr_schedulerD.step()

        for pair in tqdm(trainLoader,  desc='train'):
            src,trg = src.to(device),trg.to(device)
            ## Update discriminator ##
            DisLoss =0
            for _ in range(crit_repeats):
                disc_opt.zero_grad()
                output, _ = gen(src, trg[:,:-1])
                _,predValues = torch.max(output,2)
                real = joinrealData(convertOutput(pair,reverseOutTypes))
                fake = joinfakeData(pair,convertGenOutput(predValues.tolist(),reverseOutTypes))
                #print(f"real : {real.shape} \n fake : {fake.shape}  \n predValues:{predValues}")
                fake_mask =  gen.make_src_mask(fake)
                real_mask = gen.make_src_mask(real)
                real, fake, fake_mask, real_mask = real.to(device), fake.to(device) , fake_mask.to(device), real_mask.to(device)

                crit_fake_pred = disc(fake,fake_mask)
                crit_real_pred = disc(real, real_mask)
                disc_loss = get_crit_loss(crit_fake_pred, crit_real_pred)
                DisLoss += disc_loss.item()/crit_repeats
                disc_loss.backward(retain_graph=True)
                disc_opt.step()

                for parameters in disc.parameters():
                    parameters.data.clamp_(-clip, clip)
            totalDis += DisLoss
            ## Update generator ##
            gen_opt.zero_grad()
            output, _ = gen(src, trg[:,:-1])
            _,predValues = torch.max(output,2)
            fake = joinfakeData(pair,convertGenOutput(predValues.tolist(),reverseOutTypes))
            fake_mask =  gen.make_src_mask(fake)
            fake, fake_mask =fake.to(device) , fake_mask.to(device)
            #print(f"gen training fake :{predValues}")
            disc_fake_pred = disc(fake,fake_mask)
            gen_loss1 = get_gen_loss(disc_fake_pred)

            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trgs = trg[:,1:].contiguous().view(-1)

            gen_loss2 = criterion(output,trgs)
            gen_loss = (alpha * gen_loss1)  +  gen_loss2
            totalGen += gen_loss.item()
            gen_loss.backward()
            torch.nn.utils.clip_grad_norm_(gen.parameters(), gen_clip)
            gen_opt.step()
            #epoch_loss = gen_loss.item() + disc_loss.item()
        tLoss.append(totalGen/len(trainLoader))
        ## validating

        valid_loss = evaluate(gen, valLoader, criterion,device)
        vLoss.append(valid_loss)
        

        print(f'current learning rate : {lr_schedulerG.get_last_lr()}')
        #print(f'current learning rate Discriminator : {lr_schedulerD.get_last_lr()}')
        print(f'Epoch: {epoch+1:02}')
        print(f" Train loss {totalGen/len(trainLoader)} , validation loss :{valid_loss}")
    
            
    print("Finished Training")
    
def evalt(model,model1,Loader,device,types,test =False):
    model.eval()
    pred_trgs = []
    trgs = []
    inps = []
    with torch.no_grad():
        for i, pair in enumerate(Loader):
            batch_size = len(pair)
            src, trg = padding(pair)
            src,trg = src.to(device),trg.to(device)
            src_mask = model1.make_src_mask(src)
            enc_src = model.encoder(src, src_mask)
            
            pred_trg = [types['SOH']]
            for i in range(100):
                trg_tensor = torch.LongTensor(pred_trg).unsqueeze(0).to(device)
                trg_mask = model1.make_trg_mask(trg_tensor)
                output, attention,_ = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
                #pred_token = output.argmax(2)[:,-1].item()
                output = output.squeeze(0)
                _,pred_token = torch.max(output,1)
                #print(pred_token)
                pred_token = pred_token[-1] # 
                
                pred_trg.append(pred_token.item())
                #print(pred_trg)
                if pred_token == types['EOH']:
                    break
           # pred_trg_words.append([reverseTypes[code] for code in pred_trg])
            #print(pred_trg_words,trg)
            #trg_words.append([reverseTypes[code] for code in trg])
            inp = [code for code in src]
            trg = [code for code in trg]
            ped_trrgs.append(pred_trg)
            trgs.append(trg)
            inps.append(inp)
            #print(f'inps:{inps} , inp len {len(inps)}')        
    return pred_trgs,trgs, inps


def evaluate(model, Loader, criterion,device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, pair in enumerate(Loader):
            batch_size = len(pair)
            src, trg= padding(pair)
            src,trg = src.to(device),trg.to(device)
            #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]
            output, _ = model(src, trg[:,:-1])
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:,1:].contiguous().view(-1)

            #trg = [(trg len - 1) * batch size]
            #output = [(trg len - 1) * batch size, output dim]
            loss = criterion(output, trg)

            epoch_loss += loss.item()
        
    return epoch_loss / len(Loader)


def convList(values):
    acts =[]
    for codes in values:
        for code in codes:
            #print(code)
            #code = code.item()
            code =code.tolist()
            acts.append(code)
    return acts

def recallTop(y_true, y_pred, rank=[20,40,60,200]):
    recall = list()
    for i in range(len(y_pred)):
        thisOne = list()
        codes = y_true[i]
        tops = y_pred[i]
        for rk in rank:
            thisOne.append(len(set(codes).intersection(set(tops[:rk])))*1.0/len(set(codes)))
        recall.append( thisOne )
    return (np.array(recall)).mean(axis=0).tolist()


def joinrealData(pair):
    data = []
    #print(f"pair : {pair} ")
       
    for pair in pair:
        data.append(pair[0][:-1]+pair[1][1:])
            
    data.sort(key=lambda x: len(x), reverse=True)
    return inputVar(data).permute(1,0)

def joinfakeData(pair,output):
    data = []
    #print(f"pair : {pair , len(pair)} output = {output, type(output) , len(output) } ")

    for i in range(len(pair)):
        #print(f"iteration : {i} \n X : {pair[i][0][:-1]} \n Yhat: {output[i]}")
        data.append(pair[i][0][:-1] + output[i])

            
    data.sort(key=lambda x: len(x), reverse=True)
    return inputVar(data).permute(1,0)


def convertOutput(pair,reverseOutTypes):
    newPair = []
    for pair in pair:
        newOutput = []
        for code in pair[1]:
            newOutput.append(reverseOutTypes[code])
        newPair.append((pair[0],newOutput))
    return newPair
        
def convertGenOutput(output,reverseOutTypes):
    newOutputs = []
    for codes in output:
        newOutput = []
        for code in codes:
            #print(f" code :{code} output: {output}")
            newOutput.append(reverseOutTypes[code])
        newOutputs.append(newOutput)
        
    return newOutputs   