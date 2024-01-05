import os
import random
import argparse
import yaml
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models as torchvision_models

from datasets import build_dataset
from datasets.utils import build_data_loader
import clip
from utils import *
import dino.utils as utils
import itertools
import json
import numpy as np

def compute_prototype(dalle_train_loader, dino_model):
    with torch.no_grad():
        train_features = []
        cache_img_feas = []
        lbs_lst = []
        for i, (imgs, lbs, gt_lbs, impath) in enumerate(dalle_train_loader):
            imgs = imgs.cuda()
            img_feas = dino_model(imgs)
            train_features.append(img_feas)
            lbs = lbs.cuda()
            lbs_lst.append(lbs)
        cache_img_feas.append(torch.cat(train_features, dim=0).unsqueeze(0))
            
        cache_img_feas = torch.cat(cache_img_feas, dim=0).mean(dim=0)
        cache_img_feas /= cache_img_feas.norm(dim=-1, keepdim=True)
        cache_img_feas = cache_img_feas.permute(1, 0)
        lbs_lst = torch.cat(lbs_lst, dim=0)

    grouped_features = {}
    for i in range(lbs_lst.size(0)):
        label = lbs_lst[i].item()
        if label not in grouped_features:
            grouped_features[label] = []
        grouped_features[label].append(cache_img_feas[:, i])

    prototypes_dict = {}
    for label in grouped_features:
        features_list = grouped_features[label]
        prototypes_dict[label] = torch.stack(features_list, dim=1).mean(dim=1)   
    prototypes = torch.stack(list(prototypes_dict.values()), dim=1)  #  dim * class_num

    return prototypes

def get_dino_feas(train_loader, dino_model):
    with torch.no_grad():
        train_features = []
        cache_img_feas = []
        lbs_lst = []
        for i, (imgs, lbs, gt_lbs, impath) in enumerate(train_loader):
            imgs = imgs.cuda()
            img_feas = dino_model(imgs)
            train_features.append(img_feas)
            lbs = lbs.cuda()
            lbs_lst.append(lbs)
        
        cache_img_feas.append(torch.cat(train_features, dim=0).unsqueeze(0))          
        cache_img_feas = torch.cat(cache_img_feas, dim=0).mean(dim=0)
        cache_img_feas /= cache_img_feas.norm(dim=-1, keepdim=True)
        lbs_lst = torch.cat(lbs_lst, dim=0)

    return cache_img_feas

def get_clip_feas(train_loader, clip_model):
    with torch.no_grad():
        train_features = []
        cache_img_feas = []
        lbs_lst = []
        gt_lbs_lst = []
        impath_list = []
        for i, (imgs, lbs, gt_lbs, impath) in enumerate(train_loader):
            imgs, lbs = imgs.cuda(), lbs.cuda()
            img_feas = clip_model.encode_image(imgs)
            img_feas /= img_feas.norm(dim=-1, keepdim=True)
            train_features.append(img_feas)
            lbs = lbs.cuda()
            lbs_lst.append(lbs)
            gt_lbs_lst.append(gt_lbs)
            impath_list.append(impath)
        
        cache_img_feas.append(torch.cat(train_features, dim=0).unsqueeze(0))
        cache_img_feas = torch.cat(cache_img_feas, dim=0).mean(dim=0)
        lbs_lst = torch.cat(lbs_lst, dim=0)
        gt_lbs_lst = torch.cat(gt_lbs_lst, dim=0)

        combined_list = []
        for ii in range(len(impath_list)):
            combined_list.extend(impath_list[ii])


    return cache_img_feas, lbs_lst, gt_lbs_lst, combined_list

def clean_model(cfg, classname_clip_text_feas, clip_model, dino_model, dataset, dalle_dataset, train_tranform, re_clean=True):
    
    print("\nCleaning training data.")

    train_loader = build_data_loader(cfg, data_source=dataset.train_x, noise_data=dataset._noise_data, noise_class=dataset._nciai, img_type='original', batch_size=256, tfm=train_tranform, is_train=True, shuffle=False)        

    dalle_train_loader = build_data_loader(cfg, data_source=dalle_dataset.train_x, noise_data=dataset._noise_data, noise_class=dataset._nciai, img_type='dalle', batch_size=256, tfm=train_tranform, is_train=True, shuffle=False)
    
    if re_clean == True:
    
        # dalle_dino branch
        prototypes = compute_prototype(dalle_train_loader, dino_model) # compute dalle_dino based protopyes;  prototypes: dim * class_num
        dino_img_feas = get_dino_feas(train_loader, dino_model) # dino_img_feas: img_num * dim
        branch1_similarity = dino_img_feas @ prototypes # branch1_similarity: img_num * class_num

        # clip branch
        clip_img_feas, lbs, gt_lbs, impaths = get_clip_feas(train_loader, clip_model) 
        clip_logits = 100. * clip_img_feas @ classname_clip_text_feas
        branch2_similarity = F.softmax(clip_logits, dim=1)

        final_similarity = cfg['Alpha'] * branch1_similarity + (1-cfg['Alpha']) * branch2_similarity
        prob, pred_lb = torch.max(final_similarity, dim=1)

        lbs = lbs.cpu().numpy()
        prob = prob.cpu().numpy()
        pred_lb = pred_lb.cpu().numpy()
       
        # nlb = cfg['open_world']['nlb']
        # if nlb <= 0.3:
        #     threshold = 0.8
        # elif nlb > 0.3 and nlb <= 0.6:
        #     threshold = 0.7
        # elif nlb > 0.6 and nlb <= 0.8:
        #     threshold = 0.8 # 0.6       
        # elif nlb > 0.8 and nlb <= 1.0:
        #     threshold = 0.5          

        threshold = cfg['Beta']
        mask1 = prob >= threshold # If the model prediction value is greater than or equal to the threshold, believe the model prediction result.
        lbs[mask1] = pred_lb[mask1]

        generated_corrected_data(lbs, gt_lbs, impaths, cfg)

    dataset = build_dataset(cfg['dataset'], cfg, clean_data=True)
    return dataset

def generated_corrected_data(lbs, gt_lbs, impaths, cfg):

    file_path = os.path.join(cfg['root_path'], cfg['dataset'], 'split_cleaned_' + cfg['dataset'] + '.json')

    split_cleaned_dataset = {} 

    for ii in range(len(lbs)):
        
        if cfg['dataset'] in ['caltech-101', 'ucf101', 'dtd', 'eurosat', 'food-101', 'stanford_cars']:
            new_path = os.path.join(impaths[ii].split('/')[-2:][0], impaths[ii].split('/')[-2:][1])
        elif cfg['dataset'] in ['SUN397']:
            if len(impaths[ii].split('/')) > 10:
                new_path = os.path.join(impaths[ii].split('/')[-4:][0], impaths[ii].split('/')[-4:][1], impaths[ii].split('/')[-4:][2], impaths[ii].split('/')[-4:][3])  
            else:
                new_path = os.path.join(impaths[ii].split('/')[-3:][0], impaths[ii].split('/')[-3:][1], impaths[ii].split('/')[-3:][2])        
        else:
            new_path = impaths[ii].split('/')[-1]
        new_data = [
                new_path, 
                int(gt_lbs[ii]), # ground truth label
                '', 
                int(lbs[ii]), # corrected label
                '', 
                0 
            ]   
        
        if "train" in split_cleaned_dataset:
            split_cleaned_dataset["train"].append(new_data)
        else:
            split_cleaned_dataset["train"] = [new_data]

    json_str = json.dumps(split_cleaned_dataset)
    

    if cfg['dataset'] in ['fgvc']:
        file_path = os.path.join(cfg['root_path'], 'fgvc_aircraft', 'data', 'split_cleaned_' + cfg['dataset'] + '.json')

    with open(file_path, "w") as f:
        f.write(json_str) 
