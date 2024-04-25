import parse
from parse import args
from dataloader import dataset
from model import Model
import torch


def print_test_result():
    global best_epoch, test_pre, test_recall, test_ndcg
    print(f'Test Result(at {best_epoch:d} epoch):')
    for i, k in enumerate(parse.topks):
        print(f'ndcg@{k:d} = {test_ndcg[i]:f}, recall@{k:d} = {test_recall[i]:f}, pre@{k:d} = {test_pre[i]:f}.')


def train():
    train_loss = model.train_func()
    if epoch % args.show_loss_interval == 0:
        print(f'epoch {epoch:d}, train_loss = {train_loss:f}')


def valid(epoch):
    global best_valid_ndcg, best_epoch, test_pre, test_recall, test_ndcg
    valid_pre, valid_recall, valid_ndcg = model.valid_func()
    for i, k in enumerate(parse.topks):
        print(f'[{epoch:d}/{args.epochs:d}] Valid Result: ndcg@{k:d} = {valid_ndcg[i]:f}, recall@{k:d} = {valid_recall[i]:f}, pre@{k:d} = {valid_pre[i]:f}.')
    if valid_ndcg[-1] > best_valid_ndcg:
        best_valid_ndcg, best_epoch = valid_ndcg[-1], epoch
        test_pre, test_recall, test_ndcg = model.test_func()
        print_test_result()
        return True
    return False


model = Model(dataset).to(parse.device)

best_valid_ndcg, best_epoch = 0., 0
test_pre, test_recall, test_ndcg = torch.zeros(len(args.topks)), torch.zeros(len(args.topks)), torch.zeros(len(args.topks))
valid(epoch=0)
for epoch in range(1, args.epochs+1):
    train()
    if epoch % args.valid_interval == 0:
        if not valid(epoch) and epoch-best_epoch >= args.stopping_step*args.valid_interval:
            break
print('---------------------------')
print_test_result()
