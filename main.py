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
from clean_dataset import clean_model
from models.adapters import TeFu_Adapter

# 'imagenet', 'pets', 'caltech101', 'dtd', 'food101', 'sun', 'cars', 'ucf', 'eurosat', 'fgvc', 'oxford_flower' 
main_path = './configs/imagenet/config.yaml'

def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='settings in yaml format', type=str, default=main_path)
    args = parser.parse_args()

    return args
    
def run_TeFu_Adapter(cfg,
                     net,
                     clip_model,
                     dino_model,
                     classname_clip_text_feas,
                     train_loader,
                     dalle_train_loader,
                     clip_img_feas,
                     dino_img_feas,
                     test_lbs
                     ):

    optimizer = torch.optim.AdamW(
    itertools.chain(net.parameters()),
    lr=cfg['lr'], 
    eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader))
    best_acc, best_epoch = 0.0, 0
    Lambda, Omega = cfg['Lambda'], cfg['Omega']
    for train_idx in range(cfg['train_epoch']):
        net.train()
        correct_samples, all_samples = 0, 0
        loss_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, cfg['train_epoch']))

        # origin img
        for i, (imgs, lbs, gt_lbs, impath) in enumerate(train_loader):
            imgs, lbs = imgs.cuda(), lbs.cuda()
            with torch.no_grad():
                clip_img_features = clip_model.encode_image(imgs)
                clip_img_features /= clip_img_features.norm(dim=-1, keepdim=True)
                dino_img_features = dino_model(imgs)
                dino_img_features /= dino_img_features.norm(dim=-1, keepdim=True)

            clip_logits = 100. * clip_img_features @ classname_clip_text_feas
            TeFu_logits = net(clip_img_features,dino_img_features)  
            TeFu_logits = text_guide(clip_logits, TeFu_logits)
            TeFu_logits = ((-1) * (Omega - Omega * TeFu_logits)).exp()
            logits = clip_logits +  Lambda * TeFu_logits
            loss = F.cross_entropy(logits, lbs)

            acc = cls_acc(logits, lbs)
            correct_samples += acc / 100 * len(logits)
            all_samples += len(logits)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        # dalle img
        for i, (imgs, lbs, gt_lbs, impath) in enumerate(dalle_train_loader):
            imgs, lbs = imgs.cuda(), lbs.cuda()
            with torch.no_grad():
                clip_img_features = clip_model.encode_image(imgs)
                clip_img_features /= clip_img_features.norm(dim=-1, keepdim=True)
                dino_img_features = dino_model(imgs)
                dino_img_features /= dino_img_features.norm(dim=-1, keepdim=True)

            clip_logits = 100. * clip_img_features @ classname_clip_text_feas
            TeFu_logits = net(clip_img_features,dino_img_features)  
            TeFu_logits = text_guide(clip_logits, TeFu_logits)
            TeFu_logits = ((-1) * (Omega - Omega * TeFu_logits)).exp()
            logits = clip_logits + Lambda * TeFu_logits
            loss = F.cross_entropy(logits, lbs)

            acc = cls_acc(logits, lbs)
            correct_samples += acc / 100 * len(logits)
            all_samples += len(logits)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
   
        current_lr = scheduler.get_last_lr()[0]
        print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))

        # test
        correct_samples, all_samples = 0, 0
        net.eval()    
        with torch.no_grad():
            TeFu_logits = net(clip_img_feas,dino_img_feas) 

        clip_logits = 100. * clip_img_feas @ classname_clip_text_feas
        TeFu_logits = text_guide(clip_logits, TeFu_logits)
        TeFu_logits = ((-1) * (Omega - Omega * TeFu_logits)).exp()
        logits = clip_logits + Lambda * TeFu_logits
        acc = cls_acc(logits, test_lbs)

        print('test_Acc: {:.4f}'.format(acc))

        if acc > best_acc:
            best_acc = acc
            best_epoch = train_idx

            save_dict = {
                        'model': net.state_dict()
                        # 'optimizer': optimizer.state_dict()
                        }
            torch.save(save_dict,
                       os.path.join(cfg['cache_dir'] + "/best_TeFu_adapter" + str(cfg['shots']) + 'shots_' + str(cfg['open_world']['nlb']) + 'noise_lb.pt')
            )
    print(f"**** After fine-tuning, best test accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")

    print("\n-------- Searching hyperparameters. --------")
    # Search Hyperparameters
    save_path = os.path.join(cfg['cache_dir'] + "/best_TeFu_adapter" + str(cfg['shots']) + 'shots_' + str(cfg['open_world']['nlb']) + 'noise_lb.pt')
    state_dict = torch.load(save_path)
    state_dict_model = state_dict['model']
    net.load_state_dict(state_dict_model, strict=True)  
    acc = search_hp(cfg, net, clip_img_feas, dino_img_feas, test_lbs, classname_clip_text_feas)
    if acc > best_acc:
        best_acc = acc
    
    print(f"**** After searching hyperparameters, best test accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")

def main():

    # Load config file
    args = get_arguments()
    assert (os.path.exists(args.config))
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    cache_dir = os.path.join('./caches', cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir

    print("\nRunning configs.")
    print(cfg, "\n")

    # CLIP
    clip_model, preprocess = clip.load(cfg['clip_backbone'])
    clip_model.eval()

    # DINO
    dino_model = torchvision_models.__dict__[cfg['dino_backbone']](num_classes=0)
    dino_model.fc = nn.Identity()
    dino_model.cuda()
    utils.load_pretrained_weights(dino_model, "dino/dino_resnet50_pretrain.pth", "teacher", "vit_small", 16)
    dino_model.eval()

    # Prepare dataset
    random.seed(1)
    torch.manual_seed(1)
    
    print("Preparing dataset.")
    dataset = build_dataset(cfg['dataset'], cfg, clean_data=False) 
    test_loader = build_data_loader(cfg, data_source=dataset.test, noise_data=None, noise_class=None, img_type='original', batch_size=64, is_train=False, tfm=preprocess, shuffle=False)
    
    train_tranform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])
    dalle_dataset = build_dataset(cfg['dalle_dataset'], cfg)
    dalle_train_loader = build_data_loader(cfg, data_source=dalle_dataset.train_x, noise_data=dataset._noise_data, noise_class=dataset._nciai, img_type='dalle', batch_size=256, tfm=train_tranform, is_train=True, shuffle=True)

    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    with open(cfg['gpt3_prompt_file']) as f:
        gpt3_prompt = json.load(f)
    classname_clip_text_feas = gpt_clip_classifier(dataset.classnames, gpt3_prompt, clip_model, dataset.template)

    if cfg['open_world']['is_open_world'] == True and cfg['open_world']['clean_dataset'] == True:
        cleaned_dataset = clean_model(cfg, classname_clip_text_feas, clip_model, dino_model, dataset, dalle_dataset, train_tranform, re_clean=True)
        train_loader = build_data_loader(cfg, data_source=cleaned_dataset.train_x, noise_data=dataset._noise_data, noise_class=dataset._nciai, img_type='cleaned', batch_size=256, tfm=train_tranform, is_train=True, shuffle=True)
    else:
        cleaned_dataset = None
        train_loader = build_data_loader(cfg, data_source=dataset.train_x, noise_data=dataset._noise_data, noise_class=dataset._nciai, img_type='original', batch_size=256, tfm=train_tranform, is_train=True, shuffle=True)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    print("\nLoading CLIP feature.")
    test_clip_img_feas, test_lbs = pre_CLIP_load_features(cfg, "test", clip_model, test_loader)
    print("\nLoading DINO feature.")
    test_dino_img_feas, test_lbs = pre_DINO_load_features(cfg, "test", dino_model, test_loader)
    
    # # ------------------------------------------ Text-guided Fusion Adapter ------------------------------------------

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    net = TeFu_Adapter(cfg).cuda()
    net = nn.DataParallel(net, device_ids=[0]) 

    run_TeFu_Adapter(cfg,
                    net,
                    clip_model,
                    dino_model,
                    classname_clip_text_feas,
                    train_loader,
                    dalle_train_loader,
                    test_clip_img_feas,
                    test_dino_img_feas,
                    test_lbs
                    )

                         
if __name__ == '__main__':
    main()
