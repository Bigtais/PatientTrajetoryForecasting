import torch
from utils import *
from models import *
import time
from torch import nn
import numpy as np
import os
from tqdm import tqdm
from typing import Dict

def create_source_mask(src, source_pad_id = 0, DEVICE='cuda:0'):
    """
    Create a mask for the source sequence.

    Args:
        src (torch.Tensor): The source sequence tensor.
        source_pad_id (int, optional): The padding value for the source sequence. Defaults to 0.
        DEVICE (str, optional): The device to be used for computation. Defaults to 'cuda:0'.

    Returns:
        torch.Tensor: The source mask tensor, and the source padding mask.
    """

    src_seq_len = src.shape[1]
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE)
    source_padding_mask = (src == source_pad_id)
    return src_mask, source_padding_mask

def generate_square_subsequent_mask(tgt_seq_len, DEVICE='cuda:0'):
    """
    Generates a square subsequent mask for self-attention mechanism.

    Args:
        sz (int): The size of the mask.
        DEVICE (str, optional): The device to be used for computation. Defaults to 'cuda:0'.

    Returns:
        torch.Tensor: The square subsequent mask.

    """
    mask = (torch.triu(torch.ones((tgt_seq_len, tgt_seq_len), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_target_mask(tgt, target_pad_id = 0, DEVICE='cuda:0'):
    """
    Create a mask for the target sequence.

    Args:
        tgt (torch.Tensor): The target sequence tensor.
        target_pad_id (int, optional): The padding value for the target sequence. Defaults to 0.
        DEVICE (str, optional): The device to be used for computation. Defaults to 'cuda:0'.

    Returns:
        torch.Tensor: The target mask tensor.
    """

    tgt_seq_len = tgt.shape[1]
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, DEVICE)
    tgt_padding_mask = (tgt == target_pad_id)
    return tgt_mask, tgt_padding_mask

def create_mask(src, tgt, source_pad_id = 0, target_pad_id = 0, DEVICE='cuda:0'):
    """
    Create masks for the source and target sequences.

    Args:
        src (torch.Tensor): The source sequence tensor.
        tgt (torch.Tensor): The target sequence tensor.
        source_pad_id (int, optional): The padding value for the source sequence. Defaults to 0.
        target_pad_id (int, optional): The padding value for the target sequence. Defaults to 0.
        DEVICE (str, optional): The device to be used for computation. Defaults to 'cuda:0'.

    Returns:
        torch.Tensor: The source mask tensor.
        torch.Tensor: The target mask tensor.
        torch.Tensor: The source padding mask tensor.
        torch.Tensor: The target padding mask tensor.
    """

    src_mask, source_padding_mask = create_source_mask(src, source_pad_id, DEVICE)
    tgt_mask, target_padding_mask = create_target_mask(tgt, target_pad_id, DEVICE)

    return src_mask, tgt_mask, source_padding_mask, target_padding_mask

def get_sequences(generator , dataloader : torch.utils.data.dataloader.DataLoader,  source_pad_id : int = 0, tgt_tokens_to_ids : Dict[str, int] =  None, max_len : int = 150,  DEVICE : str ='cuda:0'):
    """
    return relevant forcasted and sequences made by the generator on the dataset.

    Args:
        generator (torch.nn.Module): The generator to be evaluated.
        val_dataloader (torch.utils.data.DataLoader): The validation dataloader.
        source_pad_id (int, optional): The padding token ID for the source input. Defaults to 0.
        DEVICE (str, optional): The device to run the evaluation on. Defaults to 'cuda:0'.
        tgt_tokens_to_ids (dict, optional): A dictionary mapping target tokens to their IDs. Defaults to None.
        max_len (int, optional): The maximum length of the generated target sequence. Defaults to 100.
    Returns:
        List[List[int]], List[List[int]]: The list of relevant and forecasted sequences.
    """

    generator.eval()
    pred_trgs = []
    targets = []

    with torch.inference_mode():
        for source_input_ids, target_input_ids in tqdm(dataloader, desc='scoring'):
            source_input_ids, target_input_ids = source_input_ids.to(DEVICE),target_input_ids.to(DEVICE)
            src_mask, source_padding_mask = create_source_mask(source_input_ids, source_pad_id, DEVICE) 
            memory = generator.batch_encode(source_input_ids, src_mask, source_padding_mask)
            pred_trg = torch.tensor(tgt_tokens_to_ids['BOS'], device= DEVICE).repeat(source_input_ids.size(0)).unsqueeze(1)
            # generate target sequence one token at a time at batch level
            for i in range(max_len):
                trg_mask = generate_square_subsequent_mask(i+1, DEVICE)
                output = generator.decode(pred_trg, memory, trg_mask)
                probs = generator.out(output[:, -1])
                pred_tokens = torch.argmax(probs, dim=1)
                eov_mask = pred_tokens == tgt_tokens_to_ids['EOV']
                if eov_mask.any():
                    # extend with sequences that have reached EOV
                    pred_trgs.extend(torch.cat((pred_trg[eov_mask],torch.tensor(tgt_tokens_to_ids['EOV'], device = DEVICE).unsqueeze(0).repeat(eov_mask.sum(), 1)),dim = -1).cpu().tolist())
                    targets.extend(target_input_ids[eov_mask].cpu().tolist())
                    # store corresponding target sequences
                    target_input_ids = target_input_ids[~eov_mask]
                    # break if all have reached EOV
                    if eov_mask.all():
                        break  
                    pred_trg = torch.cat((pred_trg[~eov_mask], pred_tokens[~eov_mask].unsqueeze(1)), dim=1)
                    memory = memory[~eov_mask]
                else:
                    pred_trg = torch.cat((pred_trg, pred_tokens.unsqueeze(1)), dim=1)
                
    return pred_trgs, targets

        
def get_gen_loss(crit_fake_pred):
    gen_loss = -1. * torch.mean(crit_fake_pred)
    return gen_loss


def get_crit_loss(crit_fake_pred, crit_real_pred):

    crit_loss =  (-1* torch.mean(crit_fake_pred)) - (-1* torch.mean(crit_real_pred))
    return crit_loss


def initializeClinicalGAN(input_dim, output_dim, hid_dim,pf_dim,gen_layers,gen_heads,dis_heads,dis_layers,dropout,lr,n_epochs,alpha,clip,loader,data,config ,gen_clip ,device):

    gen_opt = torch.optim.Adam(generator.parameters(), lr = lr)
    disc_opt = torch.optim.SGD(disc.parameters(), lr = lr)
    
    lr_schedulerG = NoamLR(gen_opt, warmup_steps=config['warmup_steps'],factor= config['factor'],model_size=config['hid_dim'])
    lr_schedulerD = NoamLR(disc_opt, warmup_steps=config['warmup_steps'],factor= config['factor'],model_size=config['hid_dim'])

    generator.apply(initialize_weights)
    disc.apply(initialize_weights)
    
    criterion = LabelSmoothingCrossEntropy(epsilon = config['eps'], ignore_index = pad_idx)

    
    modelHypermaters = {}

    
    modelHypermaters['trainloader'] = loader['trainLoader']
    modelHypermaters['testloader'] = loader['testLoader']
    modelHypermaters['valloader'] = loader['valLoader']
    
    modelHypermaters['optimizer_D'] = disc_opt
    modelHypermaters['optimizer_G'] = gen_opt
    modelHypermaters['genObject'] = gen
    modelHypermaters['discObject'] = disc
    # to access the mask function inside the generator class
    modelHypermaters['criterion'] = criterion
    
    modelHypermaters['n_epochs'] = n_epochs
    modelHypermaters['alpha'] = alpha
    modelHypermaters['clip'] = clip
    
    modelHypermaters['reverseOutTypes'] = data['codeMap']['reverseOutTypes'] 
    modelHypermaters['types'] = data['codeMap']['types'] 
    
    modelHypermaters['gen_clip'] = gen_clip
    modelHypermaters['lr_schedulerG'] = lr_schedulerG
    modelHypermaters['lr_schedulerD'] = lr_schedulerD
    return modelHypermaters
    
    
def trainCGAN(disc_opt, gen_opt, lr_schedulerG,  lr_schedulerD, generator, discriminator, criterion, valLoader, n_epochs, trainLoader, alpha, disc_clip, gen_clip):
         
    alpha = modelHypermaters['alpha']
    clip = modelHypermaters['clip']
    gen_clip= modelHypermaters['gen_clip']
    
    crit_repeats =5
    vLoss = []
    tLoss = []
    for epoch in range(n_epochs):
        totalGen = 0
        totalDis = 0
        generator.train()
        discriminator.train()
        lr_schedulerG.step()
        lr_schedulerD.step()

        for pair in tqdm(trainLoader,  desc='train'):
            src,trg = src.to(device),trg.to(device)
            ## Update discriminator ##
            DisLoss =0
            for _ in range(crit_repeats):
                disc_opt.zero_grad()
                # get sequences from generator
                real, fake, fake_mask, real_mask = real.to(device), fake.to(device) , fake_mask.to(device), real_mask.to(device)

                crit_fake_pred = discriminator(fake,fake_mask)
                crit_real_pred = discriminator(real, real_mask)
                
                disc_loss = get_crit_loss(crit_fake_pred, crit_real_pred)
                DisLoss += disc_loss.item()/crit_repeats
                disc_loss.backward(retain_graph=True)
                disc_opt.step()

                torch.nn.utils.clip_grad_value_(discriminator.parameters(), disc_clip)

            totalDis += DisLoss
            ## Update generator ##
            gen_opt.zero_grad()
            output, _ = generator(src, trg[:,:-1])
            _,predValues = torch.max(output,2)
            fake = joinfakeData(pair,convertGenOutput(predValues.tolist(),reverseOutTypes))
            fake_mask =  generator.make_src_mask(fake)
            fake, fake_mask =fake.to(device) , fake_mask.to(device)
            #print(f"generator training fake :{predValues}")
            disc_fake_pred = discriminator(fake,fake_mask)
            gen_loss1 = get_gen_loss(disc_fake_pred)

            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trgs = trg[:,1:].contiguous().view(-1)

            gen_loss2 = criterion(output,trgs)
            gen_loss = (alpha * gen_loss1)  +  gen_loss2
            totalGen += gen_loss.item()
            gen_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), gen_clip)
            gen_opt.step()
            #epoch_loss = gen_loss.item() + disc_loss.item()
        tLoss.append(totalGen/len(trainLoader))
        ## validating

        valid_loss = evaluate(generator, valLoader, criterion,device)
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