from attr import attrib
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
BATCH_SIZE = 128
EPOCH = 20
LR=1e-6
WARMUP = 3000
WD = 0.001
device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
import json
import re
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import numpy as np

import wandb
from shuffle_func import Text_Des
from datasets import clip_coco_retrieval_train, clip_coco_retrieval_eval, clip_vg_retrieval_eval, flickr_dataset, clip_vg_retrieval_train
from scheduler import cosine_lr
from torch.cuda.amp import GradScaler, autocast
from collections import Counter
import time, datetime
from utils import MetricLogger, cosine_lr_schedule

# https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model,eval): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        if not eval:
            p.grad.data = p.grad.data.float() 

def train(train_dataloader,val_dataloader, test_dataloader, model, optimizer, loss_func, device, total_epoch, init_lr, dataset_len, 
          scheduler=None, scaler = None, shuffled=False, name=""):
    model.train()
    loss_img = loss_func
    loss_txt = loss_func
    best = 0
    print("start training")
    start_time = time.time()    
    # total_step=0
    for epoch in range(total_epoch):
        cosine_lr_schedule(optimizer, epoch, total_epoch, init_lr, 0 )
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
            # total_step +=1
            # scheduler(total_step)

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

                    # logits
                    positive_logits_per_image, positive_logits_per_text = model(images, positive_texts)
                    # positive ground truth
                    positive_ground_truth = torch.eye(images.size(0),device=device)
                    # positive text loss
                    positive_txt_loss =loss_txt(positive_logits_per_text,positive_ground_truth)
                    # positive image loss
                    positive_img_loss = loss_img(positive_logits_per_image,positive_ground_truth)
                    negative_txt_loss = 0
                    negative_img_loss = 0
                    for negative_caption in negative_captions:
                        negative_texts = negative_caption.to(device)
                        negative_logits_per_image, negative_logits_per_text = model(images, negative_texts)
                        negative_ground_truth = torch.zeros_like(negative_logits_per_image,device=device)
                        # negative_txt_loss += loss_txt(negative_logits_per_text,negative_ground_truth)
                        negative_img_loss += loss_img(negative_logits_per_image, negative_ground_truth)

                    txt_loss = (positive_txt_loss + (negative_txt_loss/3)) /2
                    img_loss = (positive_img_loss + (negative_img_loss/3)) /2

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

        # _=evaluation(model, val_dataloader, device, True)
        score_test_i2t, score_test_t2i=evaluation(model, val_dataloader, device, True)
        val_result = itm_eval(score_test_i2t, score_test_t2i, val_dataloader.dataset.txt2img, val_dataloader.dataset.img2txt) 
        if val_result['r_mean']>best:
            save_obj = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            if shuffled:
                filename = '{}shuffled_checkpoint_best_epoch{}.pth'.format(name, epoch)
                torch.save(save_obj, os.path.join('outputs',filename))  
            else:
                filename = '{}original_checkpoint_best_epoch{}.pth'.format(name, epoch)
                torch.save(save_obj, os.path.join('outputs',filename))  

            best = val_result['r_mean']        

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 

@torch.no_grad()
def evaluation(model, data_loader, device, validate=False):
    model.eval()

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

    print('Computing features for evaluation...')

    text_feat = data_loader.dataset.text_feat
    image_feat = data_loader.dataset.image_feat
    text_embeds = torch.cat(text_feat,dim=0).to(device)
    image_embeds = torch.stack(image_feat).to(device)
    
    logits_per_image, logits_per_text = model(image_embeds, text_embeds)

    sims_matrix = logits_per_image.cpu()
    

    if validate:
        print('Computing validation...')
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('eval_loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Evaluation loss'
        print_freq = 50
        step=0
        total_loss = 0
        for i,(images, captions) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            step+=1
            images= images.to(device)
            reshape_caption = captions.squeeze(dim=1) # [batch_size*num_texts, 1, embeddings] -> [batch_size*num_texts, embeddings]
            texts = reshape_caption.to(device)
            ground_truth = torch.eye(images.size(0),device=device)
            
            loss_func = nn.CrossEntropyLoss()
            # no shuffle shape=(batch_size, batch_size)
            with autocast():
                logits_per_image, logits_per_text = model(images, texts)

                cur_loss = (loss_func(logits_per_image,ground_truth) + loss_func(logits_per_text,ground_truth))/2
                total_loss += cur_loss.item()
            metric_logger.update(eval_loss=cur_loss.item())
            wandb.log({"eval_loss":cur_loss.item()})
        wandb.log({"eval_avg_loss":total_loss/step})
        print(metric_logger.global_avg())

    return sims_matrix.numpy(), sims_matrix.T.numpy()

@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt, pos_id=None):
    
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
        if txt2img[index] is None:
            ranks[index] = 100
        else:
            ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / (pos_id)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / (pos_id)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / (pos_id)        

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

def main(eval=False,pretrained="",dataset='coco', shuffled=False, name=""):
    
    model, preprocess = clip.load("ViT-B/32",device=device,jit=False) #Must set jit=False for training
    if pretrained != "":
        # model, preprocess = clip.load("ViT-L/14",device=device,jit=False) #Must set jit=False for training
        print("loading pretrained model")
        checkpoint = torch.load(pretrained,map_location=device)

        # Use these 3 lines if you use default model setting(not training setting) of the clip. For example, if you set context_length to 100 since your string is very long during training, then assign 100 to checkpoint['model_state_dict']["context_length"] 
        # checkpoint['model_state_dict']["input_resolution"] = model.input_resolution #default is 224
        # checkpoint['model']["context_length"] = model.context_length # default is 77
        # checkpoint['model']["vocab_size"] = model.vocab_size 

        model.load_state_dict(checkpoint['model'])
        # model.load_state_dict(checkpoint['state_dict'])

    print("load dataset")
    if dataset=='coco':
        train_ann_root = '/ltstorage/home/2pan/dataset/COCO/coco_karpathy_train.json'
        test_ann_root = '/ltstorage/home/2pan/dataset/COCO/coco_karpathy_test.json'
        val_ann_root = '/ltstorage/home/2pan/dataset/COCO/coco_karpathy_val.json'
        image_root = '/ltstorage/home/2pan/dataset/COCO/'
        # create dataloader
        test_dataset = clip_coco_retrieval_eval(image_root, test_ann_root, preprocess)
        test_dataloader = DataLoader(test_dataset,batch_size = BATCH_SIZE, num_workers=4, shuffle=False)
        val_dataset = clip_coco_retrieval_eval(image_root, val_ann_root, preprocess, validate=True)
        val_dataloader = DataLoader(val_dataset,batch_size = BATCH_SIZE, num_workers=4, shuffle=False)
        if not eval:
            # destroy the text
            if shuffled:
                text_desc = Text_Des()
                perturb_functions = [text_desc.shuffle_nouns_and_verb_adj, text_desc.shuffle_allbut_nouns_verb_adj,
                                    text_desc.shuffle_all_words]
                # ['a pizza sitting on a plate on a table with drinks', 
                # 'a table sitting on a plate on a drinks with pizza', 
                # 'with pizza sitting on a plate a on table a drinks', 
                # 'plate a with on a drinks sitting table on a pizza']
                train_dataset = clip_coco_retrieval_train(image_root, train_ann_root, preprocess, shuffle_func_list=perturb_functions)
                dataset_len = len(train_dataset)
                train_dataloader = DataLoader(train_dataset,batch_size = BATCH_SIZE, num_workers=4, shuffle=True) #Define your own dataloader
            else:
                train_dataset = clip_coco_retrieval_train(image_root, train_ann_root, preprocess, shuffle_func_list=None)
                dataset_len = len(train_dataset)
                train_dataloader = DataLoader(train_dataset,batch_size = BATCH_SIZE, num_workers=4, shuffle=True) #Define your own dataloader

    elif dataset == 'flickr':
        test_ann_root = '/ltstorage/home/2pan/dataset/Flickr/flickr30k_test.json'
        image_root = '/ltstorage/home/2pan/dataset/Flickr/flickr30k-images'
        # all_ann_root = '/ltstorage/home/2pan/dataset/Flickr/flickr_annotations_30k.csv'
        # create dataloader
        all_dataset = clip_coco_retrieval_eval(image_root, test_ann_root, preprocess)
        dataset_len = len(all_dataset)
        test_dataloader = DataLoader(all_dataset,batch_size = BATCH_SIZE, num_workers=4, shuffle=False) #Define your own dataloader
    
    elif dataset == 'vg':
        test_vg_ann_root = '/ltstorage/home/2pan/dataset/VG_Attribution/test_visual_genome_attribution.json'
        test_retrieval_ann_root = '/ltstorage/home/2pan/dataset/VG_Attribution/test_retrieval_VG_attribution.json'
        image_root = '/ltstorage/home/2pan/dataset/VG_Attribution/images'
        # all_ann_root = '/ltstorage/home/2pan/dataset/Flickr/flickr_annotations_30k.csv'
        # create dataloader
        all_dataset = clip_vg_retrieval_eval(image_root, test_retrieval_ann_root,test_vg_ann_root, preprocess,
                                             sep=False, exc=True)
        test_dataloader = DataLoader(all_dataset,batch_size = BATCH_SIZE, num_workers=4, shuffle=False) #Define your own dataloader
        if not eval:
            train_ann_root = '/ltstorage/home/2pan/dataset/VG_Attribution/train_visual_genome_attribution.json'
            # train_ann_root = '/ltstorage/home/2pan/dataset/VG_Attribution/train_retrieval_VG_attribution.json'
            # create dataloader
            train_dataset = clip_vg_retrieval_train(image_root, train_ann_root, preprocess)
            dataset_len = len(train_dataset)
            train_dataloader = DataLoader(train_dataset,batch_size = BATCH_SIZE, num_workers=4, shuffle=True) #Define your own dataloader

            val_retrieval_ann_root = '/ltstorage/home/2pan/dataset/VG_Attribution/val_retrieval_VG_attribution.json'
            val_vg_ann_root = '/ltstorage/home/2pan/dataset/VG_Attribution/val_visual_genome_attribution.json'
            val_dataset = clip_vg_retrieval_eval(image_root, val_retrieval_ann_root,val_vg_ann_root, preprocess)
            val_dataloader = DataLoader(val_dataset,batch_size = BATCH_SIZE, num_workers=4, shuffle=False)

    elif dataset == "neglog":
        test_vg_ann_root = '/ltstorage/home/2pan/dataset/VG_Attribution/test_visual_genome_attribution.json'
        test_retrieval_ann_root = '/ltstorage/home/2pan/dataset/VG_Attribution/test_retrieval_neglog_VG_attribution.json'
        image_root = '/ltstorage/home/2pan/dataset/VG_Attribution/images'
        # all_ann_root = '/ltstorage/home/2pan/dataset/Flickr/flickr_annotations_30k.csv'
        # create dataloader
        all_dataset = clip_vg_retrieval_eval(image_root, test_retrieval_ann_root,test_vg_ann_root, preprocess,
                                             sep=False, exc=False)
        test_dataloader = DataLoader(all_dataset,batch_size = BATCH_SIZE, num_workers=4, shuffle=False) #Define your own dataloader
        if not eval:
            train_ann_root = '/ltstorage/home/2pan/dataset/VG_Attribution/train_visual_genome_attribution.json'
            # train_ann_root = '/ltstorage/home/2pan/dataset/VG_Attribution/train_retrieval_VG_attribution.json'
            # create dataloader
            train_dataset = clip_vg_retrieval_train(image_root, train_ann_root, preprocess)
            dataset_len = len(train_dataset)
            train_dataloader = DataLoader(train_dataset,batch_size = BATCH_SIZE, num_workers=4, shuffle=True) #Define your own dataloader

            val_retrieval_ann_root = '/ltstorage/home/2pan/dataset/VG_Attribution/val_retrieval_VG_attribution.json'
            val_vg_ann_root = '/ltstorage/home/2pan/dataset/VG_Attribution/val_visual_genome_attribution.json'
            val_dataset = clip_vg_retrieval_eval(image_root, val_retrieval_ann_root,val_vg_ann_root, preprocess)
            val_dataloader = DataLoader(val_dataset,batch_size = BATCH_SIZE, num_workers=4, shuffle=False)

    elif dataset == "composition":
        test_vg_ann_root = '/ltstorage/home/2pan/dataset/VG_Attribution/test_visual_genome_relation.json'
        test_retrieval_ann_root = '/ltstorage/home/2pan/dataset/VG_Attribution/test_retrieval_cp_VG_relation.json'
        image_root = '/ltstorage/home/2pan/dataset/VG_Attribution/images'
        # all_ann_root = '/ltstorage/home/2pan/dataset/Flickr/flickr_annotations_30k.csv'
        # create dataloader
        all_dataset = clip_vg_retrieval_eval(image_root, test_retrieval_ann_root,test_vg_ann_root, preprocess,
                                             sep=True, exc=True)
        test_dataloader = DataLoader(all_dataset,batch_size = BATCH_SIZE, num_workers=4, shuffle=False) #Define your own dataloader
        if not eval:
            train_ann_root = '/ltstorage/home/2pan/dataset/VG_Attribution/train_visual_genome_relation.json'
            # train_ann_root = '/ltstorage/home/2pan/dataset/VG_Attribution/train_retrieval_VG_attribution.json'
            # create dataloader
            train_dataset = clip_vg_retrieval_train(image_root, train_ann_root, preprocess)
            dataset_len = len(train_dataset)
            train_dataloader = DataLoader(train_dataset,batch_size = BATCH_SIZE, num_workers=4, shuffle=True) #Define your own dataloader

            val_retrieval_ann_root = '/ltstorage/home/2pan/dataset/VG_Attribution/val_retrieval_cp_VG_relation.json'
            val_vg_ann_root = '/ltstorage/home/2pan/dataset/VG_Attribution/val_visual_genome_relation.json'
            val_dataset = clip_vg_retrieval_eval(image_root, val_retrieval_ann_root,val_vg_ann_root, preprocess)
            val_dataloader = DataLoader(val_dataset,batch_size = BATCH_SIZE, num_workers=4, shuffle=False)

    elif dataset == "spatial":
        test_vg_ann_root = '/ltstorage/home/2pan/dataset/VG_Attribution/test_visual_genome_relation.json'
        test_retrieval_ann_root = '/ltstorage/home/2pan/dataset/VG_Attribution/test_retrieval_sp_VG_relation.json'
        image_root = '/ltstorage/home/2pan/dataset/VG_Attribution/images'
        # all_ann_root = '/ltstorage/home/2pan/dataset/Flickr/flickr_annotations_30k.csv'
        # create dataloader
        all_dataset = clip_vg_retrieval_eval(image_root, test_retrieval_ann_root,test_vg_ann_root, preprocess,
                                             sep=False, exc=True) # only true or false
        test_dataloader = DataLoader(all_dataset,batch_size = BATCH_SIZE, num_workers=4, shuffle=False) #Define your own dataloader
        if not eval:
            train_ann_root = '/ltstorage/home/2pan/dataset/VG_Attribution/train_visual_genome_relation.json'
            # train_ann_root = '/ltstorage/home/2pan/dataset/VG_Attribution/train_retrieval_VG_attribution.json'
            # create dataloader
            train_dataset = clip_vg_retrieval_train(image_root, train_ann_root, preprocess)
            dataset_len = len(train_dataset)
            train_dataloader = DataLoader(train_dataset,batch_size = BATCH_SIZE, num_workers=4, shuffle=True) #Define your own dataloader

            val_retrieval_ann_root = '/ltstorage/home/2pan/dataset/VG_Attribution/val_retrieval_sp_VG_relation.json'
            val_vg_ann_root = '/ltstorage/home/2pan/dataset/VG_Attribution/val_visual_genome_relation.json'
            val_dataset = clip_vg_retrieval_eval(image_root, val_retrieval_ann_root,val_vg_ann_root, preprocess)
            val_dataloader = DataLoader(val_dataset,batch_size = BATCH_SIZE, num_workers=4, shuffle=False)

    if device == "cpu":
        model.float()
    else :
        c_model.convert_weights(model) # Actually this line is unnecessary since clip by default already on float16

    if eval:
        score_test_i2t, score_test_t2i=evaluation(model, test_dataloader, device, False)
        test_result = itm_eval(score_test_i2t, score_test_t2i, test_dataloader.dataset.txt2img, 
                               test_dataloader.dataset.img2txt, test_dataloader.dataset.pos_id) 
        print(test_result)
    else:
        wandb.init(
        # set the wandb project where this run will be logged
        project="train_vg",
        
        # track hyperparameters and run metadata
        config={
        "epochs": 1,
        },
        name = name)

        loss_func = nn.CrossEntropyLoss()
        # loss_func = nn.BCEWithLogitsLoss()
        # optimizer = optim.Adam(model.parameters(), lr=1e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=1e-5) 
        optimizer = optim.Adam(model.parameters(), lr=LR,betas=(0.9,0.98),eps=1e-6,weight_decay=WD) 
        # optimizer = optim.Adam(model.parameters(), lr=LR,betas=(0.9,0.98),eps=1e-6,weight_decay=0.001) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
        # optimizer = optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) 

        scaler = GradScaler()
        scheduler = cosine_lr(optimizer, LR, WARMUP, len(train_dataloader))

        train(train_dataloader, val_dataloader, test_dataloader, model=model, optimizer=optimizer, loss_func=loss_func,
               device=device, total_epoch=EPOCH, init_lr=LR, 
              dataset_len=dataset_len, scaler=scaler, shuffled = shuffled, scheduler=scheduler, name=name)

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
            filename = '{}shuffled_checkpoint_final_r{:.2f}_epoch{}.pth'.format(name,test_result['r_mean'], EPOCH)
            torch.save(save_obj, os.path.join('outputs',filename))  
        else:
            filename = '{}original_checkpoint_final_r{:.2f}_epoch{}_batch{}_lr{}_wd{}.pth'.format(name, test_result['r_mean'], EPOCH, BATCH_SIZE, LR, WD)
            torch.save(save_obj, os.path.join('outputs',filename))  


if __name__ == "__main__":
    name = 'vg_1-n_'
    # retrieval task
    # task = ["","/ltstorage/home/2pan/CLIP/outputs/composition/cp_zs-2pos1neg-ViT-B-32_checkpoint_final_epoch20.pth",
    #          "/ltstorage/home/2pan/CLIP/outputs/composition/cp_zs-2pos-ViT-B-32_checkpoint_final_epoch20.pth",
    #          "/ltstorage/home/2pan/CLIP/outputs/composition/cp_zs-cor_exc-ViT-B-32_checkpoint_final_epoch20.pth",
    #          "/ltstorage/home/2pan/CLIP/outputs/composition/cp_zs-and_exc-ViT-B-32_checkpoint_final_epoch20.pth",
    #          "/ltstorage/home/2pan/CLIP/outputs/composition/cp_zs-cor-ViT-B-32_checkpoint_final_epoch20.pth"]
    # task = ["",
    #         "/ltstorage/home/2pan/CLIP/outputs/spatial/sp_zs-4rel-ViT-B-32_checkpoint_final_epoch20.pth",
    #         "/ltstorage/home/2pan/CLIP/outputs/spatial/sp_zs-left_right-ViT-B-32_checkpoint_final_epoch20.pth"]
    task = ["",
            "/ltstorage/home/2pan/CLIP/outputs/img_retrieval/original_checkpoint_final_r72.35_epoch20.pth",
            "/ltstorage/home/2pan/CLIP/outputs/img_retrieval/img2txt_shuffled_checkpoint_final_r72.35_epoch20.pth",
            "/ltstorage/home/2pan/CLIP/outputs/img_retrieval/2neg_shuffled_checkpoint_final_r72.43_epoch20.pth"]
    for pretrained in task:
        print(pretrained)

        main(eval=True, pretrained=pretrained, dataset='flickr', shuffled=False, name=name)
    # main(eval=True, pretrained="/ltstorage/home/2pan/CLIP/outputs/composition/cp_zs-cor_none-ViT-B-32_checkpoint_final_epoch20.pth", dataset='composition', shuffled=False, name=name)
    # main(eval=True, pretrained="/ltstorage/home/2pan/CLIP/outputs/vg_1-1_original_checkpoint_final_r56.34_epoch20_batch128_lr1e-06_wd0.001.pth", dataset='vg', shuffled=True, name=name)
    # main(eval=True, pretrained="outputs/shuffled_checkpoint_best_epoch5.pth", dataset='coco', shuffled=False)

    # Attribute Ownership task

    
    