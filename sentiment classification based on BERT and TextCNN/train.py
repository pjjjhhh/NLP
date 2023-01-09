from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score
from tensorboardX import SummaryWriter
from dataset import TCDataset
from model import Bert, FC, TextCNN
import numpy as np
from collections import OrderedDict
from future.utils import iteritems


def decrease_learning_rate(optimizer, factor):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * factor


def convert_weights(state_dict):
    tmp_weights = OrderedDict()
    for name, params in iteritems(state_dict):
        tmp_weights[name.replace('module.', '')] = params
    return tmp_weights


def train(args, bert, model, device, train_loader, epoch, writer, model_optimizer, bert_optimizer=None):
    if not args.not_fine_tune:
        bert.train()
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch[0].to(device)
        masks = batch[2].to(device)
        target = batch[3].to(device)

        if args.bert_fc:
            _, bert_out = bert(input_ids, masks)
        elif args.bert_cnn:
            bert_out, _ = bert(input_ids, masks)
        output = model(bert_out)
        loss = F.binary_cross_entropy(output, target)

        if not args.not_fine_tune:
            bert_optimizer.zero_grad()
        model_optimizer.zero_grad()
        loss.backward()
        if not args.not_fine_tune:
            bert_optimizer.step()
        model_optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(input_ids), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            writer.add_scalar('loss', loss.item(), epoch * len(train_loader) + batch_idx)
            # writer.add_scalar('bert_lr', bert_optimizer.param_groups[0]['lr'],
            #                   epoch * len(train_loader) + batch_idx)
            # writer.add_scalar('model_lr', model_optimizer.param_groups[0]['lr'],
            #                   epoch * len(train_loader) + batch_idx)


def val(args, bert, model, device, val_loader, epoch):
    bert.eval()
    model.eval()
    outputs = None
    labels = None
    scores = None
    with torch.no_grad():
        for input_ids, masks, _, target in tqdm(val_loader, total=len(val_loader)):
            input_ids, masks, target = input_ids.to(device), masks.to(device), target.to(device)
            if args.bert_fc:
                _, bert_out = bert(input_ids, masks)
            elif args.bert_cnn:
                bert_out, _ = bert(input_ids, masks)
            output = model(bert_out)
            output_arr = output.data.cpu().numpy()
            # output_arr[:, 0][output_arr[:, 0] >= output_arr[:, 1]] = 1
            # output_arr[:, 1][output_arr[:, 0] >= output_arr[:, 1]] = 0
            # output_arr[:, 1][output_arr[:, 0] < output_arr[:, 1]] = 1
            # output_arr[:, 0][output_arr[:, 0] < output_arr[:, 1]] = 0
            target_arr = target.data.cpu().numpy()
            score_arr = output_arr[:, 1].copy()  # 取列1，深拷贝
            output_arr = np.argmax(output_arr, axis=1)
            target_arr = np.argmax(target_arr, axis=1)

            if labels is None:
                labels = target_arr
                outputs = output_arr
                scores = score_arr
            else:
                labels = np.concatenate((labels, target_arr), axis=0)
                outputs = np.concatenate((outputs, output_arr), axis=0)
                scores = np.concatenate((scores, score_arr), axis=0)

    acc = accuracy_score(labels, outputs)
    prec, recall, f1, _ = precision_recall_fscore_support(labels, outputs, average='binary')
    auc = roc_auc_score(labels, scores)
    print('\nacc:{:.2f}%, prec: {:.2f}, recall: {:.2f}, f1: {:.2f}, AUC:{:.4f}\n'
          .format(100 * acc, 100 * prec, 100 * recall, 100 * f1, auc))
    writer.add_scalar('accuracy', acc * 100, epoch)
    writer.add_scalar('f1', f1 * 100, epoch)
    writer.add_scalar('auc', auc * 100, epoch)

    weight_dict = eval(args.eval_weight)  # 去引号
    total_score = weight_dict['acc'] * 100 * acc + weight_dict['f1'] * 100 * f1 + weight_dict['auc'] * 100 * auc
    return total_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-epochs', type=int, default=25)
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('-data_path', default='dzdp_data')
    parser.add_argument('-save_path', default='save/test')
    parser.add_argument('-num_workers', type=int, default=2)
    parser.add_argument('-log_interval', type=int, default=15)
    parser.add_argument('-eval_weight', type=str, default="{'acc':0.3, 'f1':0.4, 'auc':0.3}")
    parser.add_argument('-FT_coefficient', type=float, default=0.01)
    parser.add_argument('-coefficient', type=float, default=0.1)
    parser.add_argument('-epochs_to_decrease_lr', type=int, default=2)
    parser.add_argument('-not_fine_tune', action='store_true', default=False)
    parser.add_argument('-bert_cnn', action='store_true', default=False)
    parser.add_argument('-bert_fc', action='store_true', default=False)
    parser.add_argument('-checkpoint', default=None)
    args = parser.parse_args()

    bert_cnn, bert_fc = args.bert_cnn, args.bert_fc
    if bert_cnn == bert_fc:
        raise Exception('should be bert_cnn or bert_fc')

    # device = "cuda"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    save_path = args.save_path
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    log_path = os.path.join(save_path, 'logs')
    if not os.path.isdir(log_path):
        os.mkdir(log_path)
    writer = SummaryWriter(log_dir=log_path)

    not_fine_tune = args.not_fine_tune

    bert = Bert().to(device)
    if bert_fc:
        model = FC().to(device)
    elif bert_cnn:
        model = TextCNN(device=device).to(device)

    if not not_fine_tune:
        bert_optimizer = torch.optim.Adam(bert.parameters(),
                                          lr=args.lr * args.FT_coefficient)
    model_optimizer = torch.optim.Adam(model.parameters(),
                                       lr=args.lr)

    train_dataset = TCDataset(data_path=args.data_path,
                              is_train=True)
    val_dataset = TCDataset(data_path=args.data_path,
                            is_train=False)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  pin_memory=True,
                                  shuffle=True,
                                  drop_last=False)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                pin_memory=True,
                                shuffle=False,
                                drop_last=False)

    cntn = 0
    highest_score = 0
    epochs_without_imp = 0
    checkpoint = args.checkpoint
    if checkpoint:
        weights_dict = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        cntn = weights_dict['cntn']
        highest_score = weights_dict['highest_score']
        epochs_without_imp = weights_dict['epochs_without_imp']
        model.load_state_dict(convert_weights(weights_dict['model_state_dict']))
        model_optimizer.load_state_dict(convert_weights(weights_dict['model_optimizer_state_dict']))
        if not not_fine_tune:
            bert.load_state_dict(convert_weights(weights_dict['bert_state_dict']))
            bert_optimizer.load_state_dict(convert_weights(weights_dict['bert_optimizer_state_dict']))
    for epoch in range(1 + cntn, 1 + args.epochs):
        # train
        if not not_fine_tune:
            train(args, bert, model, device, train_dataloader, epoch, writer, model_optimizer, bert_optimizer)
        else:
            train(args, bert, model, device, train_dataloader, epoch, writer, model_optimizer)
        # val
        score = val(args, bert, model, device, val_dataloader, epoch)
        save_dict = {'model_state_dict': model.state_dict(),
                     'cntn': epoch, 'highest_score': highest_score, 'epochs_without_imp': epochs_without_imp,
                     'model_optimizer_state_dict': model_optimizer.state_dict()}
        if not not_fine_tune:
            save_dict['bert_state_dict'] = bert.state_dict()
            save_dict['bert_optimizer_state_dict'] = bert_optimizer.state_dict()
        torch.save(save_dict, args.save_path + "/checkpoint.pth.tar")
        if score > highest_score:
            torch.save(save_dict, args.save_path + "/best_checkpoint.pth.tar")
            highest_score = score
            epochs_without_imp = 0
        else:
            epochs_without_imp += 1
            if epochs_without_imp == args.epochs_to_decrease_lr:
                decrease_learning_rate(model_optimizer, args.coefficient)
                if not not_fine_tune:
                    decrease_learning_rate(bert_optimizer, args.coefficient)
                epochs_without_imp = 0
