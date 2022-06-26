from __future__ import print_function
from operator import index
### Packages
import os
import yaml
import random
import easydict
import argparse
import torch
import wandb
import logging
import numpy as np
import torch.nn.functional as F

from matplotlib import cm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

## User defined functions
from dataloader import get_loader
from network import *
from lib import *
from eval1 import *

wandb.init(project="source_trail2", entity="ajaykumar1247")

## Training settings
parser = argparse.ArgumentParser(description='Pytorch Universal Black-Box Domain Adaptation',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--config', type=str, default='./configs/officehome-train-config.yaml',help='/path/to/config/file')
parser.add_argument('--exp_name', type=str, default='officeHome/UDA/Clas2S-NRC-3IM-target-soft', help='experiment name')
parser.add_argument('--source_dir', type=str, default='save/officeHome/UDA/CE-T0.05/Art', metavar='B',help='path to source list')
parser.add_argument('--source_num', type=int, default=15, help="num of source classes")
parser.add_argument('--source_path', type=str, default='../data/officeHome/source_Art.txt', metavar='B',help='path to source list')

parser.add_argument('--target_path', nargs='+', default=['../data/officeHome/target_Product.txt'],help='list of target paths')
parser.add_argument("--gpu_devices", type=int, nargs='+', default=[0], help="")
parser.add_argument('--seed', type=int, default=2022, help="random seed")
parser.add_argument('--net', type=str, default='resnet50', help="resnet18, resnet50, and etc")
parser.add_argument('--bottleneck', type=str, default="bn", choices=["ori", "bn"])
parser.add_argument('--bottleneck_dim', type=int, default=256)
parser.add_argument('--classifier', type=str, default="wn", choices=["linear", "wn"])

args = parser.parse_args()
config_file = args.config
conf = yaml.safe_load((open(config_file)))
conf = easydict.EasyDict(conf)
gpu_devices = ','.join([str(gpu_id) for gpu_id in args.gpu_devices])
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices

## Seed - for fixed randomness
seed = args.seed
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)


if torch.cuda.is_available():
    device = torch.device("cuda")

batch_size = conf.data.dataloader.batch_size

source_path = args.source_path
print1 = "\n-----Source model training with "+source_path.split("/")[-1][7:-4]
print1 += '-----------loding source data -----------\n'
print(print1)
source_train_loader,source_val_loader,source_test_loader = get_loader(source_path, source_path, source_path, batch_size=batch_size, return_id=True, balanced=conf.data.dataloader.class_balance)


## Pre-trained source model - black box API
S_netF = ResBase(args.net).cuda()
    # classifier = feat_classifier_simpl(class_num=source_num_class, feat_dim=feat_extractor.in_features).cuda()
    # bottleneck = feat_bootleneck(type=args.bottleneck, feature_dim=feat_extractor.in_features, bottleneck_dim=args.bottleneck_dim).cuda()
    # classifier = feat_classifier(type=args.classifier, class_num = source_num_class, bottleneck_dim=args.bottleneck_dim).cuda()
S_netC = ResClassifier_MME(num_classes=args.source_num, input_size=S_netF.get_feature_dim(), temp=0.05).cuda()


modelpath = args.source_dir + '/source_F.pt'   
S_netF.load_state_dict(torch.load(modelpath))
modelpath = args.source_dir + '/source_C.pt'   
S_netC.load_state_dict(torch.load(modelpath))

source_model = nn.Sequential(S_netF, S_netC).cuda()
source_model.eval()


logger = logging.getLogger(__name__)

wandb.config.update({"learning_rate": conf.train.lr ,"epochs": conf.train.min_epoch,"batch_size": batch_size})



for target_path in args.target_path:
    print2 = "\n-----Adaptation on "+target_path.split("/")[-1][7:-4]
    print2 += '-----------loading target data -----------\n'
    print(print2)
    logger.info(print2)
    _, target_loader,test_loader = get_loader(target_path, target_path,target_path, batch_size=batch_size, return_id=True,balanced=conf.data.dataloader.class_balance)

    ### Out/record files
    taskname = args.source_dir.split("/")[-1]+"2"+target_path.split("/")[-1][7:-4]
    foldername = os.path.join('save','target',args.exp_name,taskname)
    log_file = foldername+'/log'

    fig_folder = os.path.join(foldername,'tsne-figs/')
    if not os.path.exists(fig_folder):
        os.makedirs(fig_folder)

    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))
        print("record in %s " % log_file)

    logging.basicConfig(filename=log_file,format="%(message)s")
    logger.setLevel(logging.INFO)

    



    ### Networks
    feat_extractor = ResBase(args.net).cuda()
    # classifier = feat_classifier_simpl(class_num=source_num_class, feat_dim=feat_extractor.in_features).cuda()
    # bottleneck = feat_bootleneck(type=args.bottleneck, feature_dim=feat_extractor.in_features, bottleneck_dim=args.bottleneck_dim).cuda()
    # classifier = feat_classifier(type=args.classifier, class_num = source_num_class, bottleneck_dim=args.bottleneck_dim).cuda()
    classifier = ResClassifier_MME(num_classes=args.source_num, input_size=feat_extractor.get_feature_dim(), temp=0.05).cuda()

    ## Optimizers
    param_group = []
    learning_rate = conf.train.lr
    for k, v in feat_extractor.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate*0.1}]
    for k, v in classifier.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]   
    optimizer = torch.optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    ### Training 

    acc_init = 0
    max_iters = conf.train.min_epoch * len(target_loader)
    interval_iter = max_iters // 50
    iter_num = 0

    entropy_threshold = np.log(args.source_num) / 2

    feat_extractor.train()
    # bottleneck.train()
    classifier.train()

    iter_target = iter(target_loader)

    best = {'iter':0,'HOS':0.0,'AA':0.0,'Acc_Cls':[]}
    best_netF = feat_extractor.state_dict()
    best_netC = classifier.state_dict()
    # best_netB = bottleneck.state_dict()

    num_sample = target_loader.dataset.__len__()
    print(num_sample,feat_extractor.get_feature_dim())
    feat_bank = torch.randn(num_sample, feat_extractor.get_feature_dim())
    score_bank = torch.randn(num_sample,args.source_num).cuda()
    print(feat_bank.shape)
    feat_extractor.eval()
    classifier.eval()
    with torch.no_grad():
        for data_t in target_loader:
            inputs = data_t[0].cuda()
            indx = data_t[2]
            output = feat_extractor(inputs)  # a^t
            output_norm = F.normalize(output)
            outputs = classifier(output)
            outputs = nn.Softmax(-1)(outputs)
            feat_bank[indx] = output_norm.detach().clone().cpu()
            score_bank[indx] = outputs.detach().clone()  #.cpu()

    print(score_bank.shape)
    K_NN,KK_NN = 2,2 

    criterion = torch.nn.CrossEntropyLoss().cuda()

    while iter_num < max_iters:
        alpha = max(1.0 - (iter_num /max_iters), 0)
        try:
            inputs_target, labels_target,index_t = iter_target.next()
        except:
            iter_target = iter(target_loader)
            inputs_target, labels_target,index_t = iter_target.next()

        if inputs_target.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iters)
        classifier.weight_norm()

        inputs_target, labels_target = inputs_target.cuda(), labels_target.cuda()
        outputs_source = source_model(inputs_target)
        preds_source = nn.Softmax(dim=1)(outputs_source)
        feat_t = feat_extractor(inputs_target)
        outputs_target = classifier(feat_t)
        probs_target = nn.Softmax(dim=1)(outputs_target)
        # classifier_loss = CrossEntropyLabelSmooth(num_classes=source_num_class, epsilon=0.1)(outputs_source, labels_target)            
        # classifier_loss = criterion(outputs_source, outputs_target)
        distill_loss = -torch.sum(preds_source*torch.log(probs_target+1e-5),dim=1).mean()
        loss_distillation = alpha * distill_loss


        ## Self - training loss
        loss_knw_ = torch.tensor(0).float().cuda()
        loss_unk_ = torch.tensor(0).float().cuda()
        loss_ent_k = torch.tensor(0).float().cuda()
        loss_div_k  = torch.tensor(0).float().cuda()
       
        entropy_t = -torch.sum(probs_target * torch.log(probs_target + 1e-5), dim=1)
        
        idx1 = torch.where(entropy_t < entropy_threshold - conf.train.margin)[0]
        if len(idx1) > 0:
            loss_ent_k = entropyLoss(outputs_target[idx1])
            loss_div_k = div_loss(outputs_target[idx1]) #locDiv_loss(out_t[idx1],n_source,0.2) #
            loss_knw_ = loss_ent_k + loss_div_k #locDiv_loss(out_t[idx1],n_source,0.1)
        idx2 = torch.where(entropy_t > entropy_threshold + conf.train.margin)[0]
        if len(idx2) > 0:
            # loss_unk_ +=  -entropy_loss(out_t[idx2])
            loss_unk_ = uniEnt_loss(outputs_target[idx2],args.source_num)

        loss_self_ = (3*loss_knw_) + loss_unk_

        loss_self = (1-alpha) * loss_self_

        ## NRC
        softmax_out = probs_target
        with torch.no_grad():
            output_f_norm = F.normalize(feat_t)
            output_f_ = output_f_norm.cpu().detach().clone()

            feat_bank[index_t] = output_f_.detach().clone().cpu()
            score_bank[index_t] = softmax_out.detach().clone()

            
            distance = output_f_ @ feat_bank.T
            _, idx_near = torch.topk(distance,dim=-1,largest=True,k=K_NN + 1)
            idx_near = idx_near[:, 1:]  #batch x K
            score_near = score_bank[idx_near]  #batch x K x C
            #score_near=score_near.permute(0,2,1)

            fea_near = feat_bank[idx_near]  #batch x K x num_dim
            fea_bank_re = feat_bank.unsqueeze(0).expand(fea_near.shape[0], -1,-1)  # batch x n x dim
            distance_ = torch.bmm(fea_near,fea_bank_re.permute(0, 2,1))  # batch x K x n
            _, idx_near_near = torch.topk(distance_, dim=-1, largest=True,k=KK_NN + 1)  # M near neighbors for each of above K ones
            idx_near_near = idx_near_near[:, :, 1:]  # batch x K x M
            tar_idx_ = index_t.unsqueeze(-1).unsqueeze(-1).detach().clone().cpu()
            match = (idx_near_near == tar_idx_).sum(-1).float()  # batch x K
            weight = torch.where(match > 0., match,torch.ones_like(match).fill_(0.1)) # batch x K

            weight_kk = weight.unsqueeze(-1).expand(-1, -1,KK_NN)  # batch x K x M
            weight_kk = weight_kk.fill_(0.1)

            # removing the self in expanded neighbors, or otherwise you can keep it and not use extra self regularization
            #weight_kk[idx_near_near == tar_idx_]=0

            score_near_kk = score_bank[idx_near_near]  # batch x K x M x C

            #print(weight_kk.shape)
            weight_kk = weight_kk.contiguous().view(weight_kk.shape[0],-1)  # batch x KM
            
            score_near_kk = score_near_kk.contiguous().view(score_near_kk.shape[0], -1,score_near_kk.shape[-1])  # batch x KM x C
   
            score_self = score_bank[tar_idx_]
        
        softmax_out_un = softmax_out.unsqueeze(1).expand(-1, K_NN,-1)  # batch x K x C
        
        ## nn
        loss_N =  torch.mean((F.kl_div(softmax_out_un, score_near, reduction='none').sum(-1) * weight.cuda()).sum(1))  #
        ## nn of nn
        output_re = softmax_out.unsqueeze(1).expand(-1, K_NN * KK_NN,-1)  # batch x C x 1
        const = torch.mean((F.kl_div(output_re, score_near_kk, reduction='none').sum(-1) * weight_kk.cuda()).sum(1)) # kl_div here equals to dot product since we do not use log for score_near_kk
        loss_E = torch.mean(const)

        # # self, if not explicitly removing the self feature in expanded neighbor then no need for this
        loss_S = 0.0 #-torch.mean((softmax_out * score_self).sum(-1))

        msoftmax = softmax_out.mean(dim=0)
        loss_D = torch.sum(msoftmax * torch.log(msoftmax + 1e-8))

        loss_nrc_ = loss_N + loss_E + loss_D
        loss_nrc = 1.0 * loss_nrc_

        all_loss =  loss_distillation + loss_self + loss_nrc

        optimizer.zero_grad()
        all_loss.backward()
        optimizer.step()

        if (iter_num-1) % interval_iter == 0 or iter_num == max_iters:

            wandb_loss = {}
            # wandb_loss['L_distill'] = distill_loss
            wandb_loss["Total"] = float(all_loss)
            wandb_loss["Dis"] = float(distill_loss)
            wandb_loss["a.Dis"] = float(loss_distillation)
            wandb_loss["Self"] = float(loss_self_)
            wandb_loss["(1-a).Self"] = float(loss_self)
            wandb_loss["Knw"] = float(loss_knw_)
            wandb_loss["Unk"] = float(loss_unk_)
            wandb_loss["Kn_Ent"] = float(loss_ent_k)
            wandb_loss["Kn_Div"] = float(loss_div_k)

            wandb_loss["Reg"] = float(loss_nrc_)
            wandb_loss["b.Reg"] = float(loss_nrc)
            wandb_loss['N'] = float(loss_N)
            wandb_loss['E'] = float(loss_E)
            wandb_loss['D'] = float(loss_D)
            wandb_loss['S'] = float(loss_S)
            wandb.log(wandb_loss)

            loss_print = 'Train [{}/{} ({:.2f}%)]\tLosses::  Distill: {:.6f} a. Distill: {:.6f} '.format(iter_num, max_iters,100 * float(iter_num / max_iters),distill_loss.item(),loss_distillation.item())
            loss_print += "Self: {}, (1-a).Self : {},Knw: {} (Ent : {}, Div: {}), unk: {} ".format(loss_self_,loss_self,loss_knw_,loss_ent_k,loss_div_k,loss_unk_)
            loss_print += "Reg: {}, b.Reg: {} ".format(loss_nrc_,loss_nrc)

            print(loss_print)
            logger.info(loss_print)

            feat_extractor.eval()
            classifier.eval()
            # bottleneck.eval()
            hscore,aa,acc_cls,tsne = test_uni(iter_num,test_loader,log_file,feat_extractor,None,classifier,args.source_num,entropy_threshold)
            # test_DANCE(iter_num,test_loader,log_file,10,source_num_class,feat_extractor,classifier,entropy_threshold)
            aa_src = test_cls(iter_num,source_test_loader,log_file,feat_extractor,None,classifier)

            if best['HOS'] < hscore:
                best['HOS'] = hscore
                best['AA'] = aa
                best['Acc_Cls'] = acc_cls
                best['iter'] = iter_num
                best_netF = feat_extractor.state_dict()
                best_netC = classifier.state_dict()
                # best_netB = bottleneck.state_dict()

            feat_extractor.train()
            classifier.train()
            # bottleneck.train()

            if (iter_num-1) % (10*interval_iter) == 0 or iter_num == max_iters:

            # if epoch%20 == 0:
                test_embeddings = tsne[0]
                test_predictions = tsne[1]
                tsne = TSNE(2, verbose=1)
                tsne_proj = tsne.fit_transform(test_embeddings)
                # Plot those points as a scatter plot and label them based on the pred labels
                cmap = cm.get_cmap('tab20')
                fig, ax = plt.subplots(figsize=(10,10))
                for lab in range(args.source_num):
                    indices = test_predictions==lab
                    ax.scatter(tsne_proj[indices,0],tsne_proj[indices,1], c=np.array(cmap(lab)).reshape(1,4), label = lab ,alpha=0.5)
                ax.legend(fontsize='large', markerscale=2)
                fig_file = os.path.join(fig_folder,'tsne_test_{}.png'.format(iter_num))
                fig.savefig(fig_file)

    print("Best results:: {}".format(best))
    logger.info(best)
    torch.save(best_netF, os.path.join(foldername, "source_F_bestH.pt"))
    torch.save(best_netC, os.path.join(foldername, "source_C_bestH.pt"))
    # torch.save(best_netB, os.path.join(foldername, "source_B_bestH.pt"))
    torch.save(feat_extractor.state_dict(), os.path.join(foldername, "source_F.pt"))
    torch.save(classifier.state_dict(), os.path.join(foldername, "source_C.pt"))
    # torch.save(bottleneck.state_dict(), os.path.join(foldername, "source_B.pt"))


    hscore,aa,acc_cls,_ = test_uni(iter_num,test_loader,log_file,feat_extractor,None,classifier,args.source_num,entropy_threshold)
            # test_DANCE(iter_num,test_loader,log_file,10,source_num_class,feat_extractor,classifier,entropy_threshold)
    aa_src = test_cls(iter_num,source_test_loader,log_file,feat_extractor,None,classifier)



    

    

    
