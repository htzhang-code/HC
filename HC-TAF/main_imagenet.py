import os
import random
import argparse
import yaml
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np

from topology import PersistentHomologyCalculation

from datasets.imagenet import ImageNet
import clip
from utils import *

import sys

def get_arguments():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='settings of Tip-Adapter in yaml format')
    args = parser.parse_args()

    return args

def run_tip_adapter(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights):
    
    # Zero-shot CLIP
    clip_logits = 100. * test_features @ clip_weights
    acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(acc))

    # Tip-Adapter
    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    
    affinity = test_features @ cache_keys
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    
    tip_logits = clip_logits + cache_logits * alpha
    acc = cls_acc(tip_logits, test_labels)
    print("**** Tip-Adapter's test accuracy: {:.2f}. ****\n".format(acc))

    # Search Hyperparameters
    _ = search_hp(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights)

def compute_distance_matrix(x, p=2):
    x_flat = x.view(x.size(0), -1)
    distances = torch.norm(x_flat[:, None] - x_flat, dim=2, p=p)
    return distances

def get_pairings(distances):
    signature_calculator = PersistentHomologyCalculation()
    pairs_0, pairs_1 = signature_calculator(distances.detach().cpu().numpy())
    return pairs_0
    
def clean_pairs(pairs, label):
    pairs_list = pairs.tolist()
    pairs_cache = []
    for i, pair_i in enumerate(pairs_list):
        birth = pair_i[0]
        death = pair_i[1]
        if label[birth] == label[death]:
            pass
        else:
            pairs_cache.append(pair_i)
    return np.array(pairs_cache)
    
def drop_pairs(pairs, label):
    pairs_list = pairs.tolist()
    pairs_cache = []
    for i, pair_i in enumerate(pairs_list):
        birth = pair_i[0]
        if birth in range(0, label.size(0)):
            pass
        else:
            pairs_cache.append(pair_i)
    return np.array(pairs_cache)

def get_survive_from_pairs(pairs_0):
    # Split 0th order and 1st order features (edges and cycles)
    pairs_birth = torch.from_numpy(pairs_0[:, 0]).to('cuda:0')
    pairs_death = torch.from_numpy(pairs_0[:, 1]).to('cuda:0')

    return pairs_birth, pairs_death
    
def run_tip_adapter_F(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights, clip_model, train_loader_F):
    
    # Enable the cached keys to be learnable
    adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(clip_model.dtype).cuda()
    adapter.weight = nn.Parameter(cache_keys.t())
        
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=cfg['lr'], eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader_F))
    
    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    best_acc, best_epoch = 0.0, 0

    for train_idx in range(cfg['train_epoch']):
        # Train
        adapter.train()
        correct_samples, all_samples = 0, 0
        loss_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, cfg['train_epoch']))

        for i, (images, target) in enumerate(train_loader_F):
            images, target = images.cuda(), target.cuda()
            with torch.no_grad():
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            affinity = adapter(image_features)
            cache_affinity = ((-1) * (beta - beta * affinity)).exp()
            cache_logits = cache_affinity @ cache_values
            clip_logits = 100. * image_features @ clip_weights
            tip_logits = clip_logits + cache_logits * alpha

            xentropy_loss = F.cross_entropy(tip_logits, target)
            
            batch_idx = i + 1
            
            ###
            if batch_idx == 1:
                image_features_topo = image_features
                cache_affinity_topo = cache_affinity
                target_topo = target
                
            elif batch_idx > 1:
                image_features_last = torch.load("./last/image_features_"+str(batch_idx-1)+".pth")
                cache_affinity_last = torch.load("./last/cache_affinity_"+str(batch_idx-1)+".pth")
                target_last = torch.load("./last/target_"+str(batch_idx-1)+".pth")
                
                image_features_topo = torch.cat((image_features,image_features_last), dim=0)
                cache_affinity_topo = torch.cat((cache_affinity,cache_affinity_last), dim=0)
                target_topo = torch.cat((target,target_last), dim=0)
            
            target_list = target.cpu().data.numpy().tolist()
            target_topo_list  = target_topo.cpu().data.numpy().tolist()
                
            topo_idx_list = []
            for topo_idx, target_topo_i in enumerate(target_topo_list):
                if target_topo_i in target_list:
                    topo_idx_list.append(topo_idx)
            topo_idx_tensor = torch.LongTensor(topo_idx_list).cuda()
            
            image_features_topo = image_features_topo.index_select(0, topo_idx_tensor)
            cache_affinity_topo = cache_affinity_topo.index_select(0, topo_idx_tensor)
            target_topo = target_topo.index_select(0, topo_idx_tensor)
            
            # size: (1000, 16000)
            cache_label = cache_values.t()
            # size: (256, 16000)
            label_features = cache_label.index_select(0, target_topo)
            
            ##
            image_distances = compute_distance_matrix(image_features_topo)
            image_pairs = get_pairings(image_distances)
            image_pairs_cleaned = clean_pairs(image_pairs, target_topo)
            
            pairs_birth_cleaned, pairs_death_cleaned = get_survive_from_pairs(image_pairs_cleaned)
            
            image_death = cache_affinity_topo.index_select(0, pairs_death_cleaned)
            image_birth = cache_affinity_topo.index_select(0, pairs_birth_cleaned)
            
            label_death = label_features.index_select(0, pairs_death_cleaned)
            label_birth = label_features.index_select(0, pairs_birth_cleaned)
            
            image_survive = image_death - image_birth
            text_survive = label_death - label_birth
            image_survive_normalized = image_survive / (image_survive.norm(dim=-1, keepdim=True) + 1e-8)
            text_survive_normalized = text_survive / (text_survive.norm(dim=-1, keepdim=True) + 1e-8)
            
            survive_cosine = torch.sum(image_survive_normalized * text_survive_normalized, dim=-1)
            survive_cosine_similarity = 1. - survive_cosine
            survive_cosine_loss = torch.mean(survive_cosine_similarity)
            
            torch.save(image_features, "./last/image_features_"+str(batch_idx)+".pth")
            torch.save(cache_affinity, "./last/cache_affinity_"+str(batch_idx)+".pth")
            torch.save(target, "./last/target_"+str(batch_idx)+".pth")
            
            ###
            target_list = target.cpu().data.numpy().tolist()
            target_topo_list = target_topo.cpu().data.numpy().tolist()
            target_topo_dup_set = list({x for x in target_topo_list if target_topo_list.count(x) > 1})
            
            moment_list = pairs_birth_cleaned.cpu().data.numpy().tolist()
            moment_list.extend(pairs_death_cleaned.cpu().data.numpy().tolist())
            
            target_dup_list = []
            index_dup_list = []
            for idx, target_i in enumerate(target_topo_list):
                if target_i in target_topo_dup_set and idx in moment_list:
                    target_dup_list.append(target_i)
                    index_dup_list.append(idx)
            
            index_dup_tensor = torch.LongTensor(index_dup_list).to('cuda:0')
            cache_affinity_dup = cache_affinity_topo.index_select(0, index_dup_tensor)
            target_dup_tensor = torch.LongTensor(target_dup_list).to('cuda:0')
            target_dup = cache_label.index_select(0, target_dup_tensor)
            
            cache_affinity_dup_norm = cache_affinity_dup.norm(dim=-1, keepdim=True)
            target_dup_norm = target_dup.norm(dim=-1, keepdim=True)
            target_affinity_norm = target_dup_norm / cache_affinity_dup_norm
            cache_affinity_dup = target_affinity_norm * cache_affinity_dup
            
            aff_target_dup = cache_affinity_dup - target_dup
            
            aff_target_dup_normalized = aff_target_dup / (aff_target_dup.norm(dim=-1, keepdim=True) + 1e-8)
            
            aff_tgt_topo_collect = []
            for dup_index, target_dup_i in enumerate(target_dup_list):
                index_dup_i = index_dup_list[dup_index]
                index_in_topo = []
                for topo_idx, target_topo_i in enumerate(target_topo_list):
                    if target_dup_i == target_topo_i and topo_idx != index_dup_i:
                        index_in_topo.append(topo_idx)
                index_in_topo_tensor = torch.LongTensor(index_in_topo).to('cuda:0')
                aff_topo_i = cache_affinity_topo.index_select(0, index_in_topo_tensor)
                
                dup_index_tensor = torch.LongTensor([dup_index]).to('cuda:0')
                target_dup_i = target_dup.index_select(0, dup_index_tensor)
                
                aff_topo_i_norm = aff_topo_i.norm(dim=-1, keepdim=True)
                target_dup_i_norm = target_dup_i.norm(dim=-1, keepdim=True)
                target_aff_norm = target_dup_i_norm / aff_topo_i_norm
                aff_topo_i = target_aff_norm * aff_topo_i
                
                aff_tgt_topo_i = aff_topo_i - target_dup_i
                aff_tgt_topo_normalized_i = aff_tgt_topo_i / (aff_tgt_topo_i.norm(dim=-1, keepdim=True) + 1e-8)
                aff_tgt_topo_collect.append(aff_tgt_topo_normalized_i)
            
            aff_tgt_sim = []
            for dup_index, target_dup_i in enumerate(target_dup_list):
                dup_index_tensor = torch.LongTensor([dup_index]).to('cuda:0')
                aff_tgt_topo_normalized_i = aff_tgt_topo_collect[dup_index]
                num_topo_i = aff_tgt_topo_normalized_i.size(0)
                aff_target_dup_normalized_i = aff_target_dup_normalized.index_select(0, dup_index_tensor)
                aff_tgt_cosine = torch.sum(aff_tgt_topo_normalized_i*aff_target_dup_normalized_i, dim=-1)
                aff_tgt_cosine_similarity = 1. - aff_tgt_cosine
                aff_tgt_sim_mean = torch.mean(aff_tgt_cosine_similarity, dim=-1, keepdim=True)
                aff_tgt_sim.append(aff_tgt_sim_mean)
            aff_tgt_similarity = torch.cat(aff_tgt_sim, dim=0)
            aff_tgt_cosine_loss = torch.mean(aff_tgt_similarity)
            
            loss = xentropy_loss + survive_cosine_loss + aff_tgt_cosine_loss
            topo_loss = survive_cosine_loss + 100. * aff_tgt_cosine_loss
                       
            ###
            acc = cls_acc(tip_logits, target)
            correct_samples += acc / 100 * len(tip_logits)
            all_samples += len(tip_logits)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            topo_loss.backward(retain_graph=True)
            b_grads = []
            for p in adapter.parameters():
                b_grads.append(p.grad.clone())
            
            optimizer.zero_grad()
            xentropy_loss.backward()
            
            for p, b_grad in zip(adapter.parameters(), b_grads):
                a_grad = p.grad.clone()
                a_grad_norm = a_grad.norm(dim=-1, keepdim=True)
                b_grad_norm = b_grad.norm(dim=-1, keepdim=True)
                ratio_ab = a_grad_norm / (b_grad_norm + 1e-8)
                
                p.grad = a_grad + 0.4*ratio_ab*b_grad
                
            optimizer.step()
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))

        # Eval
        adapter.eval()

        affinity = adapter(test_features)
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        clip_logits = 100. * test_features @ clip_weights
        tip_logits = clip_logits + cache_logits * alpha
        acc = cls_acc(tip_logits, test_labels)

        print("**** Tip-Adapter-F's test accuracy: {:.2f}. ****\n".format(acc))
        if acc > best_acc:
            best_acc = acc
            best_epoch = train_idx
            torch.save(adapter.weight, cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")
    
    adapter.weight = torch.load(cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")
    print(f"**** After fine-tuning, Tip-Adapter-F's best test accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")

    # Search Hyperparameters
    _ = search_hp(cfg, affinity, cache_values, test_features, test_labels, clip_weights, adapter=adapter)


def main():

    log_print = open('performance.log', 'w')
    sys.stdout = log_print
    sys.stderr = log_print
    
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
    clip_model, preprocess = clip.load(cfg['backbone'])
    clip_model.eval()

    # ImageNet dataset
    random.seed(1)
    torch.manual_seed(1)
    
    print("Preparing ImageNet dataset.")
    imagenet = ImageNet(cfg['root_path'], cfg['shots'], preprocess)

    test_loader = torch.utils.data.DataLoader(imagenet.test, batch_size=64, num_workers=8, shuffle=False)

    train_loader_cache = torch.utils.data.DataLoader(imagenet.train, batch_size=256, num_workers=8, shuffle=False)
    train_loader_F = torch.utils.data.DataLoader(imagenet.train, batch_size=256, num_workers=8, shuffle=True)

    # Textual features
    print("Getting textual features as CLIP's classifier.")
    clip_weights = clip_classifier(imagenet.classnames, imagenet.template, clip_model)

    # Construct the cache model by few-shot training set
    print("\nConstructing cache model by few-shot visual features and labels.")
    cache_keys, cache_values, cache_labels = build_cache_model(cfg, clip_model, train_loader_cache)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader)

    # ------------------------------------------ Tip-Adapter ------------------------------------------
    run_tip_adapter(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights)
    
    # ------------------------------------------ Tip-Adapter-F ------------------------------------------
    run_tip_adapter_F(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights, clip_model, train_loader_F)
           

if __name__ == '__main__':
    main()