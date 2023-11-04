import torch
from torch import nn, optim
import argparse
import os
import utils
from clip import clip
from clip import model as c_model
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from PIL import Image
# from dataset import CLIP_COCO_dataset
BATCH_SIZE = 64
EPOCH = 20
LR=1e-5
device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
import json
import re
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import numpy as np

import wandb
import time
import datetime
from shuffle_func import Text_Des
from datasets import clip_coco_retrieval_train, clip_coco_retrieval_eval, flickr_dataset
from utils import cosine_lr_schedule
from torch.cuda.amp import GradScaler, autocast

@torch.no_grad()
def evaluation(model, data_loader, device, validate=False):
    model.eval()


    print('Computing features for evaluation...')

    text_feat = data_loader.dataset.text_feat
    image_feat = data_loader.dataset.image_feat
    text_embeds = torch.cat(text_feat,dim=0).to(device)
    image_embeds = torch.stack(image_feat).to(device)
    
    logits_per_image, logits_per_text = model(image_embeds, text_embeds)

    sims_matrix = logits_per_image.cpu()
    return sims_matrix.numpy(), sims_matrix.T.numpy()

    # score_matrix_i2t = torch.full((len(data_loader.dataset.image_feat),len(data_loader.dataset.text_feat)),-100.0, dtype=torch.float16).cpu()
    # for i,sims in enumerate(sims_matrix): 
    #     topk_sim, topk_idx = sims.topk(k=256, dim=0)
    #     score_matrix_i2t[int(i), topk_idx.type(torch.int64)]=topk_sim

    # sims_matrix_t = logits_per_text
    # score_matrix_t2i = torch.full((len(data_loader.dataset.text_feat),len(data_loader.dataset.image_feat)),-100.0, dtype=torch.float16).cpu()
    # for i,sims in enumerate(sims_matrix_t): 
    #     topk_sim, topk_idx = sims.topk(k=256, dim=0)
    #     score_matrix_t2i[int(i), topk_idx.type(torch.int64)]=topk_sim

    # return score_matrix_i2t.numpy(), score_matrix_t2i.numpy()
    

    # if validate:
        # metric_logger = utils.MetricLogger(delimiter="  ")
        # metric_logger.add_meter('eval_loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        # header = 'Evaluation loss'
        # print_freq = 50
        # step=0
        # for i,(images, index) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        #     step+=1
        #     text_ids= data_loader.dataset.img2txt[index]
        #     captions = [data_loader.dataset.text_feat[i] for i in text_ids]
        #     for caption in captions:
        #         images= images.to(device)
        #         texts = caption.to(device)
        #         reshape_caption = texts.squeeze(dim=1) # [batch_size*num_texts, 1, embeddings] -> [batch_size*num_texts, embeddings]
        #         ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
        # # cur_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
        # img2text = data_loader.dataset.img2txt
        # text2img = data_loader.dataset.txt2img
        # img2text_values =list(img2text.values())
        # img2text_gt = torch.tensor(img2text_values, dtype=torch.int64)  # Adjust dtype as needed
        # text2img_values = list(text2img.values())
        # text2img_gt = torch.tensor(text2img_values, dtype=torch.int64)  # Adjust dtype as needed
        # print('img2text',img2text_gt.shape)
        # print('text2img',text2img_gt.shape)

 
@torch.no_grad()
def small_batch_evaluate(model, data_loader, device):
    model.eval()

    print('Computing features for small batch evaluation...')
    img2txt = data_loader.dataset.img2txt
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('eval_loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Evaluation loss'
    print_freq = 50
    step=0
    for i,(images, index) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            step+=1
            text_ids= img2txt[index]
            captions = [data_loader.dataset.text_feat[i] for i in text_ids]

    

@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
    
    #Images->Text 
    ranks = np.zeros(scores_i2t.shape[0])
    for index,score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
  
    #Text->Images 
    ranks = np.zeros(scores_t2i.shape[0])
    
    for index,score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)        

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result =  {'txt_r1': tr1,
                    'txt_r5': tr5,
                    'txt_r10': tr10,
                    'txt_r_mean': tr_mean,
                    'img_r1': ir1,
                    'img_r5': ir5,
                    'img_r10': ir10,
                    'img_r_mean': ir_mean,
                    'r_mean': r_mean}
    return eval_result

# https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model,eval): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        if not eval:
            p.grad.data = p.grad.data.float() 

def train(train_dataloader,val_dataloader, model, optimizer, loss_func, device, epoch, dataset_len, 
          scheduler=None, scaler = None, shuffled=False):
    model.train()
    loss_img = loss_func
    loss_txt = loss_func
    best = 0
    print("start training")
    start_time = time.time()    
    for epoch in range(EPOCH):
        cosine_lr_schedule(optimizer, epoch, EPOCH, LR, 0 )
        # scheduler.step()
        total_loss = 0
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
        metric_logger.add_meter('loss_total', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

        # add your own code to track the training progress.
        header = 'Train Epoch: [{}]'.format(epoch)
        print_freq = 50
        step = 0
        for i,(images, captions) in enumerate(metric_logger.log_every(train_dataloader, print_freq, header)):
        # for batch in train_dataloader :
            optimizer.zero_grad()
            step+=1

            # caption dim=1 is not 1 -> captions contain shuffled texts
            if captions.size(1) != 1:
                # images: torch.Size([128, 3, 224, 224]) captions: torch.Size([128, 4, 77])
                split_size = 1
                # split in tuple of 4, each size is captions: torch.Size([128, 1, 77])
                split_tensors = torch.split(captions, split_size, dim=1)

                positvie_caption = split_tensors[0].squeeze(dim=1)
                negative_captions = [split_tensors[idx].squeeze(dim=1) for idx in range(1, 4)]
                images = images.to(device)
                positive_texts = positvie_caption.to(device)
                with autocast():

                    # text loss
                    logits_per_image, positive_logits_per_text = model(images, positive_texts)
                    positive_ground_truth = torch.eye(images.size(0),device=device)
                    # positive_ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
                    positive_txt_loss =loss_txt(positive_logits_per_text,positive_ground_truth)
                    negative_loss = 0
                    for negative_caption in negative_captions:
                        negative_texts = negative_caption.to(device)
                        logits_per_image, negative_logits_per_text = model(images, negative_texts)
                        negative_ground_truth = torch.zeros_like(logits_per_image,device=device)
                        negative_loss += loss_txt(negative_logits_per_text,negative_ground_truth)

                    txt_loss = (positive_txt_loss + negative_loss)/4

                    # image loss
                    img_loss = loss_img(logits_per_image,positive_ground_truth)
                    cur_loss = (txt_loss + img_loss)/2
                    total_loss += cur_loss.item()

            else:
                # no shuffled texts
                # images: torch.Size([128, 3, 224, 224]) captions: torch.Size([128, 1, 77])
                reshape_caption = captions
                # ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
                ground_truth = torch.eye(images.size(0),device=device)
                reshape_caption = reshape_caption.squeeze(dim=1) # [batch_size*num_texts, 1, embeddings] -> [batch_size*num_texts, embeddings]

                images= images.to(device)
                texts = reshape_caption.to(device)
                
                # no shuffle shape=(batch_size, batch_size)
                with autocast():
                    logits_per_image, logits_per_text = model(images, texts)


                    cur_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
                    total_loss += cur_loss.item()

            # cur_loss.backward()
            scaler.scale(cur_loss).backward()

            if device == "cpu":
                # optimizer.step()
                scaler.step(optimizer)
            else : 
                convert_models_to_fp32(model, False)
                # optimizer.step()
                scaler.step(optimizer)
                c_model.convert_weights(model)
            scaler.update()

            metric_logger.update(loss_ita=cur_loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            wandb.log({'loss_ita': cur_loss.item(), 'lr': optimizer.param_groups[0]["lr"]})
        
        epoch_loss = total_loss / step
        metric_logger.update(loss_total=epoch_loss)
        wandb.log({'loss_total': epoch_loss})
        print("Averaged stats:", metric_logger.global_avg())     

        #eval and saved best trained

        val_result=evaluation(model, val_dataloader, device, False)
        score_test_i2t, score_test_t2i=evaluation(model, val_dataloader, device, False)
        val_result = itm_eval(score_test_i2t, score_test_t2i, val_dataloader.dataset.txt2img, val_dataloader.dataset.img2txt) 
        if val_result['r_mean']>best:
            save_obj = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            if shuffled:
                torch.save(save_obj, os.path.join('outputs', 'shuffled_checkpoint_best_r'+str(val_result['r_mean'])+'_lr'+
                                              str(optimizer.param_groups[0]["lr"])+'_epoch'+str(epoch)+'.pth'))  
            else:
                torch.save(save_obj, os.path.join('outputs', 'original_checkpoint_best_r'+str(val_result['r_mean'])+'_lr'+
                                              str(optimizer.param_groups[0]["lr"])+'_epoch'+str(epoch)+'.pth'))  

            best = val_result['r_mean']        

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 

def main(eval=False,pretrained=False,dataset='coco', shuffled=False):
    
    if pretrained:
        # model, preprocess = clip.load("ViT-L/14",device=device,jit=False) #Must set jit=False for training
        model, preprocess = clip.load("ViT-B/32",device=device,jit=False) #Must set jit=False for training
        checkpoint = torch.load("outputs/finetuned_coco_no_shuffle_lr1e-6.pt")

        # Use these 3 lines if you use default model setting(not training setting) of the clip. For example, if you set context_length to 100 since your string is very long during training, then assign 100 to checkpoint['model_state_dict']["context_length"] 
        # checkpoint['model_state_dict']["input_resolution"] = model.input_resolution #default is 224
        # checkpoint['model']["context_length"] = model.context_length # default is 77
        # checkpoint['model']["vocab_size"] = model.vocab_size 

        model.load_state_dict(checkpoint['model'])
    else:
        model, preprocess = clip.load("ViT-B/32",device=device,jit=False) #Must set jit=False for training

    print("load dataset")
    if dataset=='coco':
        train_ann_root = '/ltstorage/home/2pan/dataset/COCO/coco_karpathy_train.json'
        test_ann_root = '/ltstorage/home/2pan/dataset/COCO/coco_karpathy_test.json'
        val_ann_root = '/ltstorage/home/2pan/dataset/COCO/coco_karpathy_val.json'
        image_root = '/ltstorage/home/2pan/dataset/COCO/'
        # create dataloader
        test_dataset = clip_coco_retrieval_eval(image_root, test_ann_root, preprocess)
        test_dataloader = DataLoader(test_dataset,batch_size = BATCH_SIZE, num_workers=4, shuffle=False)
        val_dataset = clip_coco_retrieval_eval(image_root, val_ann_root, preprocess)
        val_dataloader = DataLoader(val_dataset,batch_size = BATCH_SIZE, num_workers=4, shuffle=False)
        if not eval:
            # destroy the text
            if shuffled:
                text_desc = Text_Des()
                perturb_functions = [text_desc.shuffle_nouns_and_verb_adj, text_desc.shuffle_allbut_nouns_verb_adj,
                                    text_desc.shuffle_all_words]
                train_dataset = clip_coco_retrieval_train(image_root, train_ann_root, preprocess, shuffle_func_list=perturb_functions)
                dataset_len = len(train_dataset)
                train_dataloader = DataLoader(train_dataset,batch_size = BATCH_SIZE, num_workers=4, shuffle=True) #Define your own dataloader
            else:
                train_dataset = clip_coco_retrieval_train(image_root, train_ann_root, preprocess, shuffle_func_list=None)
                dataset_len = len(train_dataset)
                train_dataloader = DataLoader(train_dataset,batch_size = BATCH_SIZE, num_workers=4, shuffle=True) #Define your own dataloader


    elif dataset == 'flickr':
        # train_ann_root = '/ltstorage/home/2pan/dataset/COCO/coco_karpathy_train.json'
        test_ann_root = '/ltstorage/home/2pan/dataset/Flickr/test_flickr.csv'
        # val_ann_root = '/ltstorage/home/2pan/dataset/COCO/coco_karpathy_val.json'
        image_root = '/ltstorage/home/2pan/dataset/Flickr/flickr30k-images'
        all_ann_root = '/ltstorage/home/2pan/dataset/Flickr/flickr_annotations_30k.csv'
        # create dataloader
        all_dataset = flickr_dataset(image_root, test_ann_root, preprocess)
        dataset_len = len(all_dataset)
        test_dataloader = DataLoader(all_dataset,batch_size = BATCH_SIZE, num_workers=4, shuffle=True) #Define your own dataloader


    if device == "cpu":
        model.float()
    else :
        c_model.convert_weights(model) # Actually this line is unnecessary since clip by default already on float16

    loss_func = nn.CrossEntropyLoss()
    # loss_func = nn.BCEWithLogitsLoss()
    # optimizer = optim.Adam(model.parameters(), lr=1e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=1e-5) 
    optimizer = optim.Adam(model.parameters(), lr=LR,betas=(0.9,0.98),eps=1e-6,weight_decay=0.001) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
    # optimizer = optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) 

    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) # cosine_lr_schedule?
    scaler = GradScaler()

    if eval:
        # eval_result=evaluation(model, val_dataloader, device, False)
        # eval_result=evaluation(model, test_dataloader, device, False)
        # convert_models_to_fp32(model,eval)
        score_test_i2t, score_test_t2i=evaluation(model, val_dataloader, device, False)
        test_result = itm_eval(score_test_i2t, score_test_t2i, val_dataloader.dataset.txt2img, val_dataloader.dataset.img2txt) 
        print(test_result)
    else:
        wandb.init(
        # set the wandb project where this run will be logged
        project="finetune_clip",
        
        # track hyperparameters and run metadata
        config={
        "epochs": 1,
        })
        train(train_dataloader, val_dataloader, model=model, optimizer=optimizer, loss_func=loss_func, device=device, epoch=0, 
              dataset_len=dataset_len, scaler=scaler, shuffled = shuffled)

        score_test_i2t, score_test_t2i=evaluation(model, test_dataloader, device)
        test_result = itm_eval(score_test_i2t, score_test_t2i, test_dataloader.dataset.txt2img, test_dataloader.dataset.img2txt) 
        print("test",test_result)
        score_val_i2t, score_val_t2i=evaluation(model, val_dataloader, device)
        val_result = itm_eval(score_val_i2t, score_val_t2i, val_dataloader.dataset.txt2img, val_dataloader.dataset.img2txt) 
        print("val",val_result)
        save_obj = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': EPOCH,
        }
        if shuffled:
            torch.save(save_obj, os.path.join('outputs', 'shuffled_checkpoint_best_r'+str(test_result['r_mean'])+'_lr'+
                                            str(optimizer.param_groups[0]["lr"])+'_epoch'+str(EPOCH)+'.pth'))  
        else:
            torch.save(save_obj, os.path.join('outputs', 'original_checkpoint_best_r'+str(test_result['r_mean'])+'_lr'+
                                            str(optimizer.param_groups[0]["lr"])+'_epoch'+str(EPOCH)+'.pth')) 


if __name__ == "__main__":
    main(eval=True, dataset='coco', shuffled=False)
    # model, preprocess = clip.load("ViT-B/32",device="cuda:3",jit=False) #Must set jit=False for training
    # checkpoint = torch.load("outputs/finetuned_coco_no_shuffle.pt")
    # model.load_state_dict(checkpoint['model'])
    # print(model.state_dict().keys())
    # with open("outputs/ori_model_structure.txt",'w') as f:
    #     f.write(str(model.state_dict))
        # for key,value in model.state_dict():
        #     f.write()
    
    