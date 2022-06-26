import wandb
import torch
import logging
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import recall_score


def test_cls(step, dataset_test, filename, G, B,C1):
    G.eval()
    C1.eval()
    if B is not None:
        B.eval()

    all_pred = []
    all_gt = []
    for batch_idx, data in enumerate(dataset_test):
        with torch.no_grad():
            img_t, label_t, index_t = data[0], data[1], data[2]
            img_t, label_t = img_t.cuda(), label_t.cuda()
            feat = G(img_t)
            if B is None:
                out_t = C1(feat)
            else:
                out_t = C1(B(feat))
            out_t = F.softmax(out_t, dim=1)
            _, pred = out_t.data.max(1)
            pred = pred.cpu().numpy()

            all_gt += list(label_t.data.cpu().numpy())
            all_pred += list(pred)
    # list to numpy

    y_true = np.array(all_gt)
    y_pred = np.array(all_pred)

    recall_avg_auc = recall_score(y_true, y_pred, labels=np.unique(y_true), average=None)
    overall_acc = np.mean(y_true == y_pred)
        
    output = [step, list(recall_avg_auc), 'AA %s' % float(recall_avg_auc.mean()),'OA %s' % float(overall_acc)]

    wandb_acc = {}
    wandb_acc["source_AA"] = float(recall_avg_auc.mean())*100.0
    wandb_acc["source_OA"] = float(overall_acc)*100.0
    wandb.log(wandb_acc)
    

    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=filename, format="%(message)s")
    logger.setLevel(logging.INFO)
    print('\n', output, '\n')
    logger.info(output)
    return float(recall_avg_auc.mean())*100.0


def test_uni(step, dataset_test, filename, G, B,C1,unk_class, threshold):
    G.eval()
    C1.eval()
    if B is not None:
        B.eval()

    all_pred = []
    all_gt = []
    entropy_scores = []
    for batch_idx, data in enumerate(dataset_test):
        with torch.no_grad():
            img_t, label_t, index_t = data[0], data[1], data[2]
            img_t, label_t = img_t.cuda(), label_t.cuda()
            feat = G(img_t)
            if B is None:
                out_t = C1(feat)
            else:
                out_t = C1(B(feat))
            out_t = F.softmax(out_t, dim=1)
            entr = -torch.sum(out_t * torch.log(out_t), 1).data.cpu().numpy()
            _, pred = out_t.data.max(1)
            pred = pred.cpu().numpy()

            all_gt += list(label_t.data.cpu().numpy())
            all_pred += list(pred)
            entropy_scores += list(entr)
    # list to numpy
    all_gt_np = np.array(all_gt)
    all_pred_np = np.array(all_pred)
    entropy_scores_np = np.array(entropy_scores)

    y_true = np.array(all_gt)
    y_pred = np.array(all_pred)

    pred_unk = np.where(entropy_scores_np > threshold)
    y_pred[pred_unk[0]] = unk_class

    recall_avg_auc = recall_score(y_true, y_pred, labels=np.unique(y_true), average=None)
    overall_acc = np.mean(y_true == y_pred)

    unk_idx = np.where(y_true == unk_class)  
    if len(unk_idx[0]) != 0:
        correct_unk = np.where(y_pred[unk_idx[0]] == unk_class)
        acc_unk = len(correct_unk[0]) / len(unk_idx[0])
        # v1 share in overall
        shared_idx = np.where(y_true != unk_class)
        shared_gt = y_true[shared_idx[0]]
        pre_shared = y_pred[shared_idx[0]]
        acc_shared1 = np.mean(shared_gt == pre_shared)
        # v2 share in average
        acc_shared = recall_avg_auc[:-1].mean()

        h_score = 2 * acc_unk * acc_shared / (acc_unk + acc_shared)
        h_score1 = 2 * acc_unk * acc_shared1 / (acc_unk + acc_shared1)

        output = [step, list(recall_avg_auc),'AA %s' % float(recall_avg_auc.mean()),'H-score %s' % float(h_score),'H-score(Inst) %s' % float(h_score1),'Acc Shared %s' % float(acc_shared),'Acc Unk %s' % float(acc_unk),'Acc Shared(Inst) %s' % float(acc_shared1),]

        wandb_acc = {}
        accs_cw = list(recall_avg_auc)
        for i in range(len(accs_cw)):
            wandb_acc["%s" %(i)] = accs_cw[i]*100.0
        wandb_acc["AA"] = float(recall_avg_auc.mean())*100.0
        wandb_acc["H-score"] = float(h_score)*100.0
        wandb_acc["H-score(Inst)"] = float(h_score1)*100.0
        wandb_acc['Acc shared'] = float(acc_shared)*100.0
        wandb_acc['Acc shared(Inst)'] = float(acc_shared1)*100.0
        wandb_acc['Acc unk'] = float(acc_unk)*100.0
        wandb.log(wandb_acc)

    else:
        output = [step, list(recall_avg_auc), 'AA %s' % float(recall_avg_auc.mean()),
                  'OA %s' % float(overall_acc)]

        wandb_acc = {}
        accs_cw = list(recall_avg_auc)
        for i in range(len(accs_cw)):
            wandb_acc["%s" %(i)] = accs_cw[i]*100.0
        wandb_acc["AA"] = float(recall_avg_auc.mean())*100.0
        wandb_acc["OA"] = float(overall_acc)*100.0
        wandb.log(wandb_acc)


    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=filename, format="%(message)s")
    logger.setLevel(logging.INFO)
    print('\n', output, '\n')
    logger.info(output)
    
    return h_score,float(recall_avg_auc.mean())*100.0,list(recall_avg_auc)


def test_DANCE(step, dataset_test, filename, n_share, unk_class, G, C1, threshold):
    G.eval()
    C1.eval()
    correct = 0
    correct_close = 0
    size = 0
    class_list = [i for i in range(n_share)]
    class_list.append(unk_class)
    per_class_num = np.zeros((n_share + 1))
    per_class_correct = np.zeros((n_share + 1)).astype(np.float32)
    per_class_correct_cls = np.zeros((n_share + 1)).astype(np.float32)
    all_pred = []
    all_gt = []
    for batch_idx, data in enumerate(dataset_test):
        with torch.no_grad():
            img_t, label_t, path_t = data[0], data[1], data[2]
            img_t, label_t = img_t.cuda(), label_t.cuda()
            feat = G(img_t)
            out_t = C1(feat)
            out_t = F.softmax(out_t)
            entr = -torch.sum(out_t * torch.log(out_t), 1).data.cpu().numpy()
            pred = out_t.data.max(1)[1]
            k = label_t.data.size()[0]
            pred_cls = pred.cpu().numpy()
            pred = pred.cpu().numpy()

            pred_unk = np.where(entr > threshold)
            pred[pred_unk[0]] = unk_class
            all_gt += list(label_t.data.cpu().numpy())
            all_pred += list(pred)
            for i, t in enumerate(class_list):
                t_ind = np.where(label_t.data.cpu().numpy() == t)
                correct_ind = np.where(pred[t_ind[0]] == t)
                correct_ind_close = np.where(pred_cls[t_ind[0]] == i)
                per_class_correct[i] += float(len(correct_ind[0]))
                per_class_correct_cls[i] += float(len(correct_ind_close[0]))
                per_class_num[i] += float(len(t_ind[0]))
                correct += float(len(correct_ind[0]))
                correct_close += float(len(correct_ind_close[0]))
            size += k
    per_class_acc = per_class_correct / per_class_num
    close_p = float(per_class_correct_cls.sum() / per_class_num.sum())
    print('\nTest set including unknown classes:  Accuracy: {}/{} ({:.0f}%)  '
        '({:.4f}%)\n'.format(
            correct, size,
            100. * correct / size, float(per_class_acc.mean())))
    output = [step, list(per_class_acc), 'per class mean acc %s'%float(per_class_acc.mean()),
              float(correct / size), 'closed acc %s'%float(close_p)]
    
    print(per_class_acc)
    output = [step, list(recall_avg_auc),'AA %s' % float(recall_avg_auc.mean()),'H-score %s' % float(h_score),'H-score(Inst) %s' % float(h_score1),'Acc Shared %s' % float(acc_shared),'Acc Unk %s' % float(acc_unk),'Acc Shared(Inst) %s' % float(acc_shared1),]

    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=filename, format="%(message)s")
    logger.setLevel(logging.INFO)
    print(output)
    logger.info(output)