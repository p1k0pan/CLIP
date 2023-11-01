from torch.utils.data import DataLoader, Dataset
from PIL import Image
import json
import re
import os
from clip import clip
import pandas as pd
from ast import literal_eval

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
        self.text_feat = []
        self.image_feat = []

        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann['image'])
            img_path = os.path.join(self.image_root,ann['image'])      
            img = self.preprocess(Image.open(img_path).convert('RGB'))
            self.image_feat.append(img)
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                text = pre_caption(caption,max_words)
                self.text.append(text)
                txt = clip.tokenize(text)
                self.text_feat.append(txt)
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1

    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        return self.image_feat[index], index

class flickr_dataset(Dataset):
    def __init__(self, image_root, ann_root, preprocess, max_words=30):

        self.image_root = image_root
        self.preprocess = preprocess
        self.annotation = pd.read_csv(ann_root)
        self.annotation['img_id'] = self.annotation['img_id'].astype(int)

        self.txt2img = {}
        self.img2txt = {}
        self.text_feat = []
        self.image_feat = []

        txt_id = 0
        for index, row in self.annotation.iterrows():
            # image
            captions = literal_eval(row['raw'])
            # sentids = literal_eval(row['sentids'])
            # self.img2txt[row['img_id']]=sentids
            self.img2txt[index] = []
            img_path = os.path.join(self.image_root,row['filename'])      
            img = self.preprocess(Image.open(img_path).convert('RGB'))
            self.image_feat.append(img)
            # caption
            # sent_dict = {key: row['img_id'] for key in sentids}
            # self.txt2img.update(sent_dict)
            for caption in captions:
                text = pre_caption(caption,max_words)
                txt = clip.tokenize(text)
                self.text_feat.append(txt)
                self.img2txt[index].append(txt_id)
                self.txt2img[txt_id] = index
                txt_id += 1

    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        return self.image_feat[index], self.text_feat[index]