# from attr import attrib
import torch
from torch import nn, optim
import argparse
import os
import utils
from clip import clip
from clip import model as c_model
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
# from dataset import CLIP_COCO_dataset
BATCH_SIZE = 128
EPOCH = 20
LR=1e-6
WARMUP = 3000
device = "cuda:2" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
import json
import numpy as np
import pandas as pd

import wandb
from scheduler import cosine_lr
from torch.cuda.amp import GradScaler, autocast
from collections import Counter
from datasets_zoo import snare_datasets
import time, datetime
from utils import MetricLogger, cosine_lr_schedule
import subprocess

# https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model,eval): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        if not eval:
            p.grad.data = p.grad.data.float() 

def train(train_dataloader,val_dataloader, model, optimizer, loss_func, device, total_epoch, init_lr, 
          scheduler=None, scaler = None, name="", val_dataset=None, dataset="composition"):
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
        for i,batch in enumerate(metric_logger.log_every(train_dataloader, print_freq, header)):
        # for batch in train_dataloader :
            optimizer.zero_grad()
            step+=1
            relation = batch['relation'] 
            # images: torch.Size([128, 3, 224, 224]) captions: torch.Size([128, 1, 77])
            images = batch["image_options"]
            images= images.to(device)

            # caption_options: [true_caption, false_caption, split_semantic]
            # batch['caption_options'][0][0]: the happy man and the orange phone
            # batch['caption_options'][1][0]: the orange man and the happy phone
            # batch['caption_options'][2][0]: the man and the phone are happy and orange respectively
            caption_options = batch["caption_options"] #[true_caption, false_caption, split_semantic]
            cur_loss = 0
            positive_loss = 0
            negative_loss = 0
            for idx in range(len(caption_options)):
                caption = caption_options[idx]
                caption_tokenized = torch.cat([clip.tokenize(c) for c in caption]).to(device)

                with autocast():
                    logits_per_image, logits_per_text = model(images, caption_tokenized)
                    assert logits_per_image.size(0) == images.size(0), "size not compatible"
                    if dataset == 'composition':
                        # if idx !=1 :
                        #     # if idx ==0: # only use true caption as positive on training
                        #     #     ground_truth = torch.eye(logits_per_image.size(0),device=device)
                        #     #     cur_loss += (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
                        #     # if idx == 2: # only use split_semantic as positive on training
                        #     #     ground_truth = torch.eye(logits_per_image.size(0),device=device)
                        #     #     cur_loss += (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
                        #     ground_truth = torch.eye(logits_per_image.size(0),device=device)
                        #     cur_loss += (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
                        # else:
                        #     # continue # don't use negative sample on training
                        #     negative_ground_truth = torch.zeros_like(logits_per_image,device=device)
                        #     cur_loss += (loss_img(logits_per_image,negative_ground_truth) + loss_txt(logits_per_text,negative_ground_truth))/2
                        if idx ==0 :
                            ground_truth = torch.eye(logits_per_image.size(0),device=device)
                            positive_loss += (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
                        else:
                            # if idx == 1:
                            #     negative_ground_truth = torch.zeros_like(logits_per_image,device=device)
                            #     cur_loss += (loss_img(logits_per_image,negative_ground_truth) + loss_txt(logits_per_text,negative_ground_truth))/2
                            # continue # don't use negative sample on training
                            negative_ground_truth = torch.zeros_like(logits_per_image,device=device)
                            negative_loss += (loss_img(logits_per_image,negative_ground_truth) + loss_txt(logits_per_text,negative_ground_truth))/2
                        cur_loss = positive_loss + negative_loss 
                    elif dataset == "spatial":
                        # if idx == 0 or idx == 1: # only left or right
                        ground_truth = torch.zeros_like(logits_per_image,device=device)
                        for rel in range(len(relation)):
                            if relation[rel] == idx:
                                ground_truth[rel,rel] = 1

                        # ground_truth = torch.eye(logits_per_image.size(0),device=device) # only true caption as positive on training
                        cur_loss += (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2

                   
                    total_loss += cur_loss

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

            metric_logger.update(loss_ita=cur_loss)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            wandb.log({'loss_ita': cur_loss, 'lr': optimizer.param_groups[0]["lr"]})
        
        epoch_loss = total_loss / step
        metric_logger.update(loss_total=epoch_loss)
        wandb.log({'loss_total': epoch_loss})
        print("Averaged stats:", metric_logger.global_avg())     

        #eval and saved best trained

        # _=evaluation(model, val_dataloader, device, True)
        convert_models_to_fp32(model, True)
        all_scores=evaluation(model, val_dataloader, device, True, dataset)
        c_model.convert_weights(model)
        if val_dataset is not None:
            val_result = val_dataset.evaluate_scores(all_scores)
            df = pd.DataFrame(val_result)
            if dataset == 'composition':
                correct_df = df[df['Attributes'] == 'correct']
                separate_df = df[df['Attributes'] == 'and']
                negative_df = df[df['Attributes'] == 'exchange']
                top = correct_df['correct_top-1'].values + separate_df['and_top-1'].values - negative_df['exchange_top-1'].values
            elif dataset == 'spatial':
                correct_df = df[df['Attributes'] == 'classification']
                top = correct_df['top-1'].values
            wandb.log({'top1-acc': top})
            if top>best:
                save_obj = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                }
                filename = '{}_checkpoint_best_epoch{}.pth'.format(name, epoch)
                torch.save(save_obj, os.path.join('outputs',filename))  

                best = top

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 

@torch.no_grad()
def evaluation(model, data_loader, device, validate=False, dataset="composition"):
    model.eval()

    print('Computing features for evaluation...')

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('eval_loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Evaluation loss'
    print_freq = 50
    step=0
    total_loss = 0
    scores = []
    for i,batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
    # for batch in train_dataloader :
        step+=1
        
        image_options = []
        image_embeddings = model.encode_image(batch["image_options"].to(device)).cpu()  # B x D
        image_embeddings = image_embeddings / image_embeddings.norm(dim=1, keepdim = True)
        image_options.append(np.expand_dims(image_embeddings.numpy(), axis=1))

        caption_options = []
        relation = batch['relation']
        # print(batch['caption_options'])
        """
        [('a road with a red dirt on a small moped on a man helmet',), 
        ('a man a red helmet a a small moped on on dirt road with',), 
        ('man a a small on helmet a red with moped on dirt road a',)]

        "A man with a red helmet on a small moped on a dirt road. ", 
        
        """
        for idx in range(len(batch["caption_options"])):
            caption_tokenized = torch.cat([clip.tokenize(c) for c in batch["caption_options"][idx]])
            caption_embeddings = model.encode_text(caption_tokenized.to(device)).cpu()  # B x D

            # caption_embeddings = caption_embeddings - self.blank

            # caption_embeddings = caption_embeddings / np.linalg.norm(caption_embeddings, axis=1,
            #                                                             keepdims=True)  # B x D
            caption_embeddings = caption_embeddings / caption_embeddings.norm(dim=1, keepdim = True)
            caption_options.append(np.expand_dims(caption_embeddings.numpy(), axis=1))
            
            # validate need calculate loss
            if validate:
                logits_per_image = image_embeddings @ caption_embeddings.t()
                logits_per_text = logits_per_image.T
                loss_func = nn.CrossEntropyLoss()
                cur_loss = 0
                positive_loss = 0
                negative_loss = 0
                with autocast():
                    if dataset == "composition":
                        # if idx != 1:
                        #     # if idx == 0:
                        #     #     ground_truth = torch.eye(logits_per_image.size(0),device="cpu")
                        #     #     cur_loss += (loss_func(logits_per_image,ground_truth) + loss_func(logits_per_text,ground_truth))/2
                        #     # if idx == 2: # only use split_semantic as positive on training
                        #     #     ground_truth = torch.eye(logits_per_image.size(0),device="cpu")
                        #     #     cur_loss += (loss_func(logits_per_image,ground_truth) + loss_func(logits_per_text,ground_truth))/2
                        #     ground_truth = torch.eye(logits_per_image.size(0),device="cpu")
                        #     cur_loss += (loss_func(logits_per_image,ground_truth) + loss_func(logits_per_text,ground_truth))/2
                        # else:
                        #     negative_ground_truth = torch.zeros_like(logits_per_image,device="cpu")
                        #     cur_loss += (loss_func(logits_per_image,negative_ground_truth) + loss_func(logits_per_text,negative_ground_truth))/2
                        #     # continue
                        if idx == 0:
                            ground_truth = torch.eye(logits_per_image.size(0),device="cpu")
                            positive_loss += (loss_func(logits_per_image,ground_truth) + loss_func(logits_per_text,ground_truth))/2
                        else:
                            # if idx == 1:
                            #     negative_ground_truth = torch.zeros_like(logits_per_image,device="cpu")
                            #     cur_loss += (loss_func(logits_per_image,negative_ground_truth) + loss_func(logits_per_text,negative_ground_truth))/2
                            negative_ground_truth = torch.zeros_like(logits_per_image,device="cpu")
                            negative_loss += (loss_func(logits_per_image,negative_ground_truth) + loss_func(logits_per_text,negative_ground_truth))/2
                        cur_loss = positive_loss + negative_loss 
                            # continue
                    elif dataset == "spatial":
                        # if idx == 0 or idx == 1:
                        ground_truth = torch.zeros_like(logits_per_image,device="cpu")
                        for rel in range(len(relation)):
                            if relation[rel] == idx:
                                ground_truth[rel,rel] = 1
                        # ground_truth = torch.eye(logits_per_image.size(0),device="cpu")
                        cur_loss += (loss_func(logits_per_image,ground_truth) + loss_func(logits_per_text,ground_truth))/2


                    total_loss += cur_loss
                wandb.log({"eval_loss":cur_loss})

        # print(len(caption_options))
        image_options = np.concatenate(image_options, axis=1)  # B x K x D
        # print("image_options", image_options.shape)
        caption_options = np.concatenate(caption_options, axis=1)  # B x L x D
        # print("caption_options", caption_options.shape)
        batch_scores = np.einsum("nkd,nld->nkl", image_options, caption_options)  # B x K x L

      
        # # 例子矩阵
        # nkd = image_options
        # nld = caption_options
        #
        # # 广播矩阵形状
        # nkd_expanded = np.expand_dims(nkd, axis=2)  # shape: (3, 4, 1, 2)
        # nld_expanded = np.expand_dims(nld, axis=1)  # shape: (3, 1, 5, 2)
        #
        # # 计算差值
        # diff = nkd_expanded - nld_expanded  # shape: (3, 4, 5, 2)
        #
        # # 计算平方差并求和
        # squared_diff = diff ** 2  # shape: (3, 4, 5, 2)
        # sum_squared_diff = np.sum(squared_diff, axis=-1)  # shape: (3, 4, 5)
        #
        # # 计算欧式距离
        # batch_scores = np.sqrt(sum_squared_diff)  # shape: (3, 4, 5)
        #
        # batch_scores = batch_scores_cos / batch_scores

        scores.append(batch_scores)
    if validate:
        wandb.log({"eval_avg_loss":total_loss/step})
    print(metric_logger.global_avg())
    all_scores = np.concatenate(scores, axis=0)  # N x K x L
    
    return all_scores


def main(eval=False,pretrained="",dataset='composition', name=""):
    
    model, preprocess = clip.load("ViT-B/32",device=device,jit=False) #Must set jit=False for training
    print(device)
    if pretrained != "":
        # model, preprocess = clip.load("ViT-L/14",device=device,jit=False) #Must set jit=False for training
        print("loading pretrained model")
        checkpoint = torch.load(pretrained)
        # checkpoint = torch.load(pretrained,map_location=device)

        # Use these 3 lines if you use default model setting(not training setting) of the clip. For example, if you set context_length to 100 since your string is very long during training, then assign 100 to checkpoint['model_state_dict']["context_length"] 
        # checkpoint['model_state_dict']["input_resolution"] = model.input_resolution #default is 224
        # checkpoint['model']["context_length"] = model.context_length # default is 77
        # checkpoint['model']["vocab_size"] = model.vocab_size 

        model.load_state_dict(checkpoint['model'])

    print("load dataset")
    # root_dir="/ltstorage/home/2pan/dataset/VG_Attribution"
    # annotation_file = os.path.join(root_dir, "visual_genome_relation.json")
    # train_file = os.path.join(root_dir, "train_visual_genome_relation.json")
    # test_file = os.path.join(root_dir, "test_visual_genome_relation.json")
    # val_file = os.path.join(root_dir, "val_visual_genome_relation.json")
    # image_dir = os.path.join(root_dir, "images")
    # if not os.path.exists(image_dir):
    #     print("Image Directory for VG_Attribution could not be found!")
    #     os.makedirs(root_dir, exist_ok=True)
    #     image_zip_file = os.path.join(root_dir, "vgr_vga_images.zip")
    #     subprocess.call(["gdown", "--no-cookies", "1qaPlrwhGNMrR3a11iopZUT_GPP_LrgP9", "--output", image_zip_file])
    #     subprocess.call(["unzip", "vgr_vga_images.zip"], cwd=root_dir)

    # if not os.path.exists(annotation_file):
    #     subprocess.call(["gdown", "--id", "1kX2iCHEv0CADL8dSO1nMdW-V0NqIAiP3", "--output", annotation_file])

    root_dir="/ltstorage/home/2pan/dataset/gvqa"
    train_file = os.path.join(root_dir, "gvqa/seed0/spatial_relation_train.json")
    test_file = os.path.join(root_dir, "gvqa/seed0/spatial_relation_test.json")
    val_file = os.path.join(root_dir, "gvqa/seed0/spatial_relation_val.json")
    image_dir = os.path.join(root_dir, "images")

    # create dataloader
    if eval:
        with open(test_file, "r") as f:
            test_dataset = json.load(f)

        for item in test_dataset:
            # item["image_path"] = os.path.join(image_dir, item["image_path"])
            item["image_path"] = os.path.join(image_dir, item["image_id"])

        if dataset == "composition":
            test_dataset = snare_datasets.VG_Relation(preprocess, subordination_relation=True, dataset=test_dataset)
        elif dataset == "spatial":
            test_dataset = snare_datasets.VG_Relation(preprocess, multi_spatial_relation=True, dataset=test_dataset)
        test_dataloader = DataLoader(test_dataset,batch_size = BATCH_SIZE, num_workers=4, shuffle=False)
    else:
        with open(train_file, "r") as f:
            train_dataset = json.load(f)
            for item in train_dataset:
                # item["image_path"] = os.path.join(image_dir, item["image_path"])
                item["image_path"] = os.path.join(image_dir, item["image_id"])
        with open(test_file, "r") as f:
            test_dataset = json.load(f)
            for item in test_dataset:
                # item["image_path"] = os.path.join(image_dir, item["image_path"])
                item["image_path"] = os.path.join(image_dir, item["image_id"])
        with open(val_file, "r") as f:
            val_dataset = json.load(f)
            for item in val_dataset:
                # item["image_path"] = os.path.join(image_dir, item["image_path"])
                item["image_path"] = os.path.join(image_dir, item["image_id"])

        if dataset == 'composition':
            test_dataset = snare_datasets.VG_Relation(preprocess, subordination_relation=True, dataset=test_dataset)
            train_dataset = snare_datasets.VG_Relation(preprocess, subordination_relation=True, dataset=train_dataset)
            val_dataset = snare_datasets.VG_Relation(preprocess, subordination_relation=True, dataset=val_dataset)
        elif dataset == 'spatial':
            root_dir="/ltstorage/home/2pan/dataset/gvqa"
            train_file = os.path.join(root_dir, "gvqa/seed0/spatial_relation_train.json")
            test_file = os.path.join(root_dir, "gvqa/seed0/spatial_relation_test.json")
            val_file = os.path.join(root_dir, "gvqa/seed0/spatial_relation_val.json")
            image_dir = os.path.join(root_dir, "images")
            test_dataset = snare_datasets.VG_Relation(preprocess, multi_spatial_relation=True, dataset=test_dataset)
            train_dataset = snare_datasets.VG_Relation(preprocess, multi_spatial_relation=True, dataset=train_dataset)
            val_dataset = snare_datasets.VG_Relation(preprocess, multi_spatial_relation=True, dataset=val_dataset)

        train_dataloader = DataLoader(train_dataset,batch_size = BATCH_SIZE, num_workers=4, shuffle=True)
        # train_dataloader = DataLoader(train_dataset,batch_size = BATCH_SIZE, num_workers=4, shuffle=True)
        val_dataloader = DataLoader(val_dataset,batch_size = BATCH_SIZE, num_workers=4, shuffle=False)
        test_dataloader = DataLoader(test_dataset,batch_size = BATCH_SIZE, num_workers=4, shuffle=False)

    # for batch in test_dataloader:
    #     for idx in range(len(batch['caption_options'])):
    #         print(idx)
    #         print(batch['caption_options'][idx])
    #     return

    if device == "cpu":
        model.float()
    else :
        c_model.convert_weights(model) # Actually this line is unnecessary since clip by default already on float16

    if eval:
        print("evaluating dataset", root_dir)
        # print("below",test_dataset.all_relations.count('below')) # 20
        # print("left",test_dataset.all_relations.count('to the left of')) # 695
        # print("right",test_dataset.all_relations.count('to the right of')) # 695
        # print("on",test_dataset.all_relations.count('on')) #158
        convert_models_to_fp32(model, True)
        # print(test_dataset.all_attributes)
        all_scores=evaluation(model, test_dataloader, device, False, dataset)
        if test_dataset is not None:
            test_result = test_dataset.evaluate_scores(all_scores)
            for record in test_result:
                record.update({"Model": "clip", "Dataset": dataset, "name": name})
            print(test_result)
            output_file = os.path.join("outputs/", f"eval_{name}.csv")
            df = pd.DataFrame(test_result)

            # print(f"Saving results to {output_file}")
            # if os.path.exists(output_file):
            #     all_df = pd.read_csv(output_file, index_col=0)
            #     all_df = pd.concat([all_df, df])
            #     all_df.to_csv(output_file)
            # else:
            #     df.to_csv(output_file)
    else:
        wandb.init(
        # set the wandb project where this run will be logged
        project="vg_relation",
        name= name,
        
        # track hyperparameters and run metadata
        config={
        "epochs": 1,
        })

        loss_func = nn.CrossEntropyLoss()
        # loss_func = nn.BCEWithLogitsLoss()
        # optimizer = optim.Adam(model.parameters(), lr=1e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=1e-5) 
        optimizer = optim.Adam(model.parameters(), lr=LR,betas=(0.9,0.98),eps=1e-6,weight_decay=0.001) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
        # optimizer = optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) 

        scaler = GradScaler()
        scheduler = cosine_lr(optimizer, LR, WARMUP, len(train_dataloader))

        train(train_dataloader, val_dataloader, model=model, optimizer=optimizer, loss_func=loss_func, device=device, total_epoch=EPOCH, 
            init_lr = LR, scaler=scaler, scheduler=scheduler, name=name, val_dataset=val_dataset, dataset = dataset)

        convert_models_to_fp32(model, True)
        all_scores=evaluation(model, test_dataloader, device, False, dataset)
        if test_dataset is not None:
            test_result = test_dataset.evaluate_scores(all_scores)
            print(test_result)
            for record in test_result:
                record.update({"Model": "clip", "Dataset": dataset, "name": name})
            output_file = os.path.join("outputs/", f"{dataset}_clip_{name}_epoch-{EPOCH}_lr-{LR}_batch-{BATCH_SIZE}.csv")

            df = pd.DataFrame(test_result)
            # correct_df = df[df['Attributes'] == 'correct']
            # separate_df = df[df['Attributes'] == 'separate']
            save_obj = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': EPOCH,
            }
            filename = '{}_checkpoint_final_epoch{}.pth'.format(name, EPOCH)
            torch.save(save_obj, os.path.join('outputs',filename))  
        
            print(f"Saving results to {output_file}")
            if os.path.exists(output_file):
                all_df = pd.read_csv(output_file, index_col=0)
                all_df = pd.concat([all_df, df])
                all_df.to_csv(output_file)
            else:
                df.to_csv(output_file)


if __name__ == "__main__":
    name = 'sp_zs-4rel-ViT-B-32'
    # retrieval task
    # main(eval=True, pretrained="", dataset='composition', name=name)
    main(eval=False, pretrained="", dataset='spatial', name=name)

    # with open("/ltstorage/home/2pan/dataset/VG_Attribution/visual_genome_relation.json") as f:
    #     data = json.load(f)
    # all_relations=[]
    # for item in data:
    #     all_relations.append(item["relation_name"])

    # print("below",all_relations.count('below')) #209
    # print("left",all_relations.count('to the left of')) # 7741
    # print("right",all_relations.count('to the right of')) # 7741
    # print("on",all_relations.count('on')) # 1684
    
    