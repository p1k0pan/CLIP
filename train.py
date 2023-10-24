import torch
from torch import nn, optim
import argparse
import os
import utils
from clip import clip
from clip import model as c_model
from torch.utils.data import DataLoader, Dataset
from PIL import Image
# from dataset import CLIP_COCO_dataset
BATCH_SIZE = 128
EPOCH = 20
import json
import re
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

import wandb
import time
import datetime
from shuffle_func import Text_Des

def pre_caption(caption,max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~])",       
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
            
    return caption

class clip_coco_retrieval_train(Dataset):
    def __init__(self, image_root, ann_root, preprocess, max_words=30, shuffle_func_list=None):

        self.image_root = image_root
        self.max_words = max_words
        self.preprocess = preprocess
        self.shuffle_func_list = shuffle_func_list
        self.annotation = json.load(open(ann_root,'r'))

        self.img_ids = {}  
        n = 0
        for ann in self.annotation:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1    
        
    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        ann = self.annotation[idx]

        img_path = os.path.join(self.image_root,ann['image'])      
        image = self.preprocess(Image.open(img_path)) # Image from PIL module

        text = pre_caption(ann['caption'], self.max_words)
        
        if self.shuffle_func_list is not None:
            shuffled_texts = []
            for shuffle_func in self.shuffle_func_list:
                shuffled_texts.append(shuffle_func(text))

            all_caption = [text] + shuffled_texts # 4 texts: 1 original text + 3 shuffled texts
            all_caption = clip.tokenize(all_caption)
        else:
            all_caption = clip.tokenize(text)

        return image,all_caption

class clip_coco_retrieval_eval(Dataset):
    
    def __init__(self, image_root, ann_root, preprocess, max_words=30):

        self.image_root = image_root
        self.preprocess = preprocess
        self.annotation = json.load(open(ann_root,'r'))

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption,max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        image_path = os.path.join(self.image_root, self.annotation[index]['image'])        
        image = Image.open(image_path).convert('RGB')    
        image = self.preprocess(image)  

        return image, index

def main():
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
    model, preprocess = clip.load("ViT-L/14",device=device,jit=False) #Must set jit=False for training
    ann_root = '/ltstorage/home/2pan/dataset/COCO/coco_karpathy_train.json'
    image_root = '/ltstorage/home/2pan/dataset/COCO/'

    # destroy the text
    text_desc = Text_Des()
    perturb_functions = [text_desc.shuffle_nouns_and_verb_adj, text_desc.shuffle_allbut_nouns_verb_adj,
                        text_desc.shuffle_all_words]
    
    # create dataloader
    dataset = clip_coco_retrieval_train(image_root, ann_root, preprocess, shuffle_func_list=perturb_functions)
    dataset_len = len(dataset)
    train_dataloader = DataLoader(dataset,batch_size = BATCH_SIZE, num_workers=4, shuffle=True) #Define your own dataloader

    #https://github.com/openai/CLIP/issues/57
    # def convert_models_to_fp32(model): 
    #     for p in model.parameters(): 
    #         p.data = p.data.float() 
    #         p.grad.data = p.grad.data.float() 


    if device == "cpu":
        model.float()
    else :
        c_model.convert_weights(model) # Actually this line is unnecessary since clip by default already on float16

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=1e-5) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
    # optimizer = optim.Adam(model.parameters(), lr=1e-6,betas=(0.9,0.98),eps=1e-6,weight_decay=0.001) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
    # optimizer = optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


    print("start training")
    start_time = time.time()    

    for epoch in range(EPOCH):
        scheduler.step()
        total_loss = 0
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
        metric_logger.add_meter('loss_total', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

        # add your own code to track the training progress.
        header = 'Train Epoch: [{}]'.format(epoch)
        print_freq = 50
        for i,(images, captions) in enumerate(metric_logger.log_every(train_dataloader, print_freq, header)):
        # for batch in train_dataloader :
            optimizer.zero_grad()

            # when they have different length -> captions contain shuffled texts
            # images: torch.Size([128, 3, 224, 224]) captions: torch.Size([128, 4, 77])
            if len(images) != len(captions):
                reshape_caption = captions.view(-1, 1, captions.size(-1)) # [batch_size, num_texts=4, embeddings] -> [batch_size*num_texts, 1, embeddings]
                ground_truth = torch.arange(0, len(reshape_caption), 4,dtype=torch.long,device=device) # every 4th caption is the ground truth
            else:
                # no shuffled texts
                reshape_caption = captions
                ground_truth = torch.arange(len(images),dtype=torch.long,device=device)

            reshape_caption = reshape_caption.squeeze(dim=1) # [batch_size*num_texts, 1, embeddings] -> [batch_size*num_texts, embeddings]
            images= images.to(device)
            texts = reshape_caption.to(device)
            
            logits_per_image, logits_per_text = model(images, texts)


            cur_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
            total_loss += cur_loss
            cur_loss.backward()
            if device == "cpu":
                optimizer.step()
            else : 
                # convert_models_to_fp32(model)
                optimizer.step()
                c_model.convert_weights(model)

            metric_logger.update(loss_ita=cur_loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            wandb.log({'loss_ita': total_loss.item(), 'lr': optimizer.param_groups[0]["lr"]})
        
        epoch_loss = total_loss / dataset_len
        metric_logger.update(loss_total=epoch_loss.item())
        wandb.log({'loss_total': epoch_loss.item()})
        print("Averaged stats:", metric_logger.global_avg())     
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 

    save_obj = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': EPOCH,
                }
    torch.save(save_obj, 'outputs/no_shuffle.pt')  


if __name__ == "__main__":
    # wandb.init(
    # # set the wandb project where this run will be logged
    # project="finetune_clip",
    
    # # track hyperparameters and run metadata
    # config={
    # "epochs": 1,
    # })
    main()
    