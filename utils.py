from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import clip

def search_hp(cfg, 
            net,
            clip_img_feas,
            dino_img_feas,
            test_lbs,
            classname_clip_text_feas
            ):
    """refer to 'Prompt, Generate, then Cache: Cascade of Foundation Models makes Strong Few-shot Learners, CVPR 2023.'"""
    
    best_acc = 0 
    best_Lambda = 0
    if cfg['search_hp'] == True:
    
        Omega_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
        Lambda_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]
        # Lambda_list = [0.05,0.1,0.15,0.,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.5,0.9,0.95,1.0,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4,1.45,1.5]
        with torch.no_grad():
            TeFu_logits = net(clip_img_feas, dino_img_feas) 
        clip_logits = 100. * clip_img_feas @ classname_clip_text_feas
        TeFu_logits = text_guide(clip_logits, TeFu_logits)
        
        for Omega in Omega_list:
            for Lambda in Lambda_list:
                TeFu_logits = ((-1) * (Omega - Omega * TeFu_logits)).exp()
                logits = clip_logits + Lambda * TeFu_logits  
                acc = cls_acc(logits, test_lbs)      
                if acc > best_acc:
                    print("New best setting, Lambda: {:.2f}; accuracy: {:.2f}".format(Lambda, acc))
                    best_acc = acc
                    best_Lambda = Lambda  
                    best_Omega = Omega   

        print("\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))

    return best_acc


def cls_acc(output, lbs, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(lbs.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / lbs.shape[0]
    return acc


def pre_CLIP_load_features(cfg, split, clip_model, loader):

    if cfg['load_pre_feat'] == False:
        features, labels = [], []

        with torch.no_grad():
            for i, (imgs, lbs, gt_lbs, impath) in enumerate(tqdm(loader)):
                lbs = torch.where(lbs < 0, torch.tensor(int(cfg['num_classes'])-1), lbs)
                imgs, lbs = imgs.cuda(), lbs.cuda()
                image_features = clip_model.encode_image(imgs)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                features.append(image_features)
                labels.append(lbs)

        features, labels = torch.cat(features), torch.cat(labels)

        torch.save(features, cfg['cache_dir'] + "/" + split + "_" + "clip_fea.pt")
        torch.save(labels, cfg['cache_dir'] + "/" + split + "_" + "clip_lb.pt")
 
    else:
        features = torch.load(cfg['cache_dir'] + "/" + split + "_" + "clip_fea.pt")
        labels = torch.load(cfg['cache_dir'] + "/" + split + "_" + "clip_lb.pt")
    
    return features, labels

def pre_DINO_load_features(cfg, split, dino_model, loader):
    
    if cfg['load_pre_feat'] == False:
        features, labels = [], []

        with torch.no_grad():
            for i, (imgs, lbs, gt_lbs, impath) in enumerate(tqdm(loader)):
                lbs = torch.where(lbs < 0, torch.tensor(int(cfg['num_classes'])-1), lbs)
                imgs, lbs = imgs.cuda(), lbs.cuda()
                image_features = dino_model(imgs)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                features.append(image_features)
                labels.append(lbs)

        features, labels = torch.cat(features), torch.cat(labels)

        torch.save(features, cfg['cache_dir'] + "/" + split + "_" + "dino_fea.pt")
        torch.save(labels, cfg['cache_dir'] + "/" + split + "_" + "dino_lb.pt")
   
    else:
        features = torch.load(cfg['cache_dir'] + "/" + split + "_" + "dino_fea.pt")
        labels = torch.load(cfg['cache_dir'] + "/" + split + "_" + "dino_lb.pt")
    
    return features, labels


def gpt_clip_classifier(classnames, gpt_prompts, clip_model, template):
    with torch.no_grad():
        classname_clip_text_feas = []
        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = []
            for t in gpt_prompts[classname]:
                texts.append(t)
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            classname_clip_text_feas.append(class_embedding)

        classname_clip_text_feas = torch.stack(classname_clip_text_feas, dim=1).cuda()
    return classname_clip_text_feas

def get_features(clip_model, dino_model, imgs):
    
    clip_image_features = clip_model.encode_image(imgs)
    clip_image_features /= clip_image_features.norm(dim=-1, keepdim=True)
    
    dino_image_features = dino_model(imgs)
    dino_image_features /= dino_image_features.norm(dim=-1, keepdim=True)
    
    return clip_image_features, dino_image_features

def text_guide(zero_logtis, logit, normalize='mean'):
    # refer to "Prompt, Generate, then Cache: Cascade of Foundation Models makes Strong Few-shot Learners, CVPR, 2023"
    logit = F.log_softmax(logit,dim=1)
    softmax_fun = nn.Softmax(dim=1)
    if normalize == 'mean':
        logits_std = torch.std(zero_logtis, dim=1, keepdim=True)
        logits_mean = torch.mean(zero_logtis, dim=1, keepdim=True)
        zero_logtis = (zero_logtis - logits_mean) / logits_std   

        logits_std = torch.std(logit, dim=1, keepdim=True)
        logits_mean = torch.mean(logit, dim=1, keepdim=True)
        current_normalize_logits = (logit - logits_mean) / logits_std 
    else:
        raise("error normalize!")
    similarity_matrix = []
    normalize_logits = []    
    current_similarity = current_normalize_logits * zero_logtis
    current_similarity = torch.sum(current_similarity, dim=1, keepdim=True)
    
    similarity_matrix.append(current_similarity)
    normalize_logits.append(current_normalize_logits)
    similarity_matrix = torch.stack(similarity_matrix, dim=-2)
    similarity_matrix = softmax_fun(similarity_matrix)
    normalize_logits = torch.stack(normalize_logits, dim=-2)
    
    result_logits = torch.sum(normalize_logits * similarity_matrix, dim=1)    
    
    return result_logits

