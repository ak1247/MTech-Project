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


## User defined functions
from dataloader import get_loader
from network import *
from lib import *
from eval1 import *

wandb.init(project="source_trail1", entity="ajaykumar1247")

## Training settings
parser = argparse.ArgumentParser(description='Pytorch Universal Black-Box Domain Adaptation',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--config', type=str, default='./configs/officehome-train-config.yaml',help='/path/to/config/file')
parser.add_argument('--exp_name', type=str, default='officeHome/UDA/CE-LS_BotNeck', help='experiment name')
parser.add_argument('--source_path', type=str, default='../data/officeHome/source_Clipart.txt', metavar='B',help='path to source list')
parser.add_argument('--target_path', nargs='+', default=['../data/officeHome/target_Product.txt'],help='list of target paths')
parser.add_argument("--gpu_devices", type=int, nargs='+', default=[3], help="")
parser.add_argument('--seed', type=int, default=2021, help="random seed")
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
## Data paths
source_path = args.source_path
target_path = args.target_path[0]

### Out/record files
taskname = source_path.split("/")[-1][7:-4]
foldername = os.path.join('save',args.exp_name,taskname)
log_file = foldername+'/log'

if not os.path.exists(os.path.dirname(log_file)):
    os.makedirs(os.path.dirname(log_file))
print("record in %s " % log_file)

logger = logging.getLogger(__name__)
logging.basicConfig(filename=log_file,format="%(message)s")
logger.setLevel(logging.INFO)

## Data Loading
print1 = "\n-----Source model training with "+source_path.split("/")[-1][7:-4]
print1 += '-----------loding source data -----------\n'
print(print1)
logger.info(print1)
source_train_loader,source_val_loader,source_test_loader = get_loader(source_path, source_path, source_path, batch_size=batch_size, return_id=True, balanced=conf.data.dataloader.class_balance)

print2 = "\n-----Testing on "+target_path.split("/")[-1][7:-4]
print2 += '-----------loading target data -----------\n'
print(print2)
logger.info(print2)
_, target_loader,test_loader = get_loader(target_path, target_path,target_path, batch_size=batch_size, return_id=True,balanced=conf.data.dataloader.class_balance)


## get numbers of shared and source classes
source_classes = set(source_train_loader.dataset.labels)
source_num_class = len(source_classes)
print('--source_number_classes:', source_num_class)
print("Source classes: {}".format(source_classes))




### Networks
feat_extractor = ResBase(args.net).cuda()
# classifier = feat_classifier_simpl(class_num=source_num_class, feat_dim=feat_extractor.in_features).cuda()
bottleneck = feat_bootleneck(type=args.bottleneck, feature_dim=feat_extractor.in_features, bottleneck_dim=args.bottleneck_dim).cuda()
classifier = feat_classifier(type=args.classifier, class_num = source_num_class, bottleneck_dim=args.bottleneck_dim).cuda()

## Optimizers
param_group = []
learning_rate = conf.train.lr
for k, v in feat_extractor.named_parameters():
    param_group += [{'params': v, 'lr': learning_rate*0.1}]
for k, v in classifier  .named_parameters():
    param_group += [{'params': v, 'lr': learning_rate}]   
optimizer = torch.optim.SGD(param_group)
optimizer = op_copy(optimizer)

param_group1 = []
for k, v in feat_extractor.named_parameters():
    param_group1 += [{'params': v, 'lr': 0.1}]
for k, v in classifier  .named_parameters():
    param_group1 += [{'params': v, 'lr': 1}]   
optimizer1 = torch.optim.SGD(param_group1)


wandb.config.update({"learning_rate": conf.train.lr ,"epochs": conf.train.min_epoch,"batch_size": batch_size})


### Training 

acc_init = 0
max_iters = conf.train.min_epoch * len(source_train_loader)
interval_iter = max_iters // 10
iter_num = 0

entropy_threshold = np.log(source_num_class) / 2

feat_extractor.train()
bottleneck.train()
classifier.train()

iter_source = iter(source_train_loader)

best = {'iter':0,'HOS':0.0,'AA':0.0,'Acc_Cls':[]}
best_netF = feat_extractor.state_dict()
best_netC = classifier.state_dict()
best_netB = bottleneck.state_dict()

while iter_num < max_iters:
    try:
        inputs_source, labels_source,_ = iter_source.next()
    except:
        iter_source = iter(source_train_loader)
        inputs_source, labels_source,_ = iter_source.next()

    if inputs_source.size(0) == 1:
        continue

    iter_num += 1
    lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iters)

    inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
    outputs_source = classifier(bottleneck(feat_extractor(inputs_source)))
    classifier_loss = CrossEntropyLabelSmooth(num_classes=source_num_class, epsilon=0.1)(outputs_source, labels_source)            
    

    optimizer.zero_grad()
    classifier_loss.backward()
    optimizer.step()

    if (iter_num-1) % interval_iter == 0 or iter_num == max_iters:

        wandb_loss = {}
        wandb_loss['loss'] = classifier_loss
        loss_print = 'Train [{}/{} ({:.2f}%)]\tLoss Source: {:.6f} '.format(iter_num, max_iters,100 * float(iter_num / max_iters),classifier_loss.item())
        wandb.log(wandb_loss)

        print(loss_print)
        logger.info(loss_print)

        feat_extractor.eval()
        classifier.eval()
        bottleneck.eval()
        hscore,aa,acc_cls = test_uni(iter_num,test_loader,log_file,feat_extractor,bottleneck,classifier,source_num_class,entropy_threshold)
        # test_DANCE(iter_num,test_loader,log_file,10,source_num_class,feat_extractor,classifier,entropy_threshold)
        aa_src = test_cls(iter_num,source_test_loader,log_file,feat_extractor,bottleneck,classifier)

        if best['HOS'] < hscore:
            best['HOS'] = hscore
            best['AA'] = aa
            best['Acc_Cls'] = acc_cls
            best['iter'] = iter_num
            best_netF = feat_extractor.state_dict()
            best_netC = classifier.state_dict()
            best_netB = bottleneck.state_dict()

        feat_extractor.train()
        classifier.train()
        bottleneck.train()


print("Best results:: {}".format(best))
logger.info(best)
torch.save(best_netF, os.path.join(foldername, "source_F_bestH.pt"))
torch.save(best_netC, os.path.join(foldername, "source_C_bestH.pt"))
torch.save(best_netB, os.path.join(foldername, "source_B_bestH.pt"))
torch.save(feat_extractor.state_dict(), os.path.join(foldername, "source_F.pt"))
torch.save(classifier.state_dict(), os.path.join(foldername, "source_C.pt"))
torch.save(bottleneck.state_dict(), os.path.join(foldername, "source_B.pt"))

    

feat_extractor.eval()
classifier.eval()

for target_path in args.target_path:
    print2 = "\n-----Testing on "+target_path.split("/")[-1][7:-4]
    print2 += '-----------loading target data -----------\n'
    print(print2)
    logger.info(print2)
    _, target_loader, test_loader = get_loader(target_path, target_path,target_path, batch_size=batch_size, return_id=True,balanced=conf.data.dataloader.class_balance)


    hscore,aa,acc_cls = test_uni(iter_num,test_loader,log_file,feat_extractor,bottleneck,classifier,source_num_class,entropy_threshold)
    # test_DANCE(iter_num,test_loader,log_file,10,source_num_class,feat_extractor,classifier,entropy_threshold)

