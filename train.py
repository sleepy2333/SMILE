import torch
import datetime
import torch.nn as nn
from model import MULTModel
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import mmDataset, MMDataset, AlignedMoseiDataset, UnAlignedMoseiDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import get_metrics
from spikingjelly.activation_based import functional


def initialize(dataset: str, aligned: bool, feature: str, config):
    if feature == 'glove':
        DS = MMDataset
        dataset_path = 'dataset path'
    elif feature == 'Bert':
        DS = mmDataset
        dataset_path = 'dataset path'
    else:
        raise ValueError('No this feature in the setting.')

    print("Start loading the data....")

    train_data = AlignedMoseiDataset(dataset_path, 'train')
    valid_data = AlignedMoseiDataset(dataset_path, 'valid')
    test_data = AlignedMoseiDataset(dataset_path, 'test')

    print('Successfully loaded.')

    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle=True)

    model = MULTModel()  # Perceiver()

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=config.when, factor=0.1,
                                              verbose=True) if config.scheduler else None
    # loss_fn = nn.L1Loss()
    loss_fn = nn.BCELoss()
    # loss_fn = nn.MultiLabelSoftMarginLoss()

    return train_loader, valid_loader, test_loader, model, optimizer, scheduler, loss_fn


def train(model, optimizer, scheduler, train_data, valid_data, loss_fn, max_epochs, early_stop, PATH, device='cuda'):
    best_valid = 1e8
    best_epoch = 0
    model.train()

    for epoch in range(max_epochs):
        model = model.to(device)
        for l, _, a, _, v, _, targets in train_data:
            ##########
            l, a, v, targets = l.to(device=device, dtype=torch.float), a.to(device=device, dtype=torch.float), v.to(
                device=device, dtype=torch.float), targets.to(device=device, dtype=torch.float)

            # label smoothing
            targets = torch.abs(targets - 0.2).detach()
            outputs = model(l, a, v)
            loss = loss_fn(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.8)  # 0.8
            optimizer.step()
            ####
            functional.reset_net(model)
        ####

        train_loss = loss.item()  # , _, _ = evaluate(model, train_data, loss_fn, device)
        cur_valid, _, _ = evaluate(model, valid_data, loss_fn, device)

        # Decay learning rate by validation loss
        if not scheduler is None:
            scheduler.step(cur_valid)

        if cur_valid <= (best_valid - 1e-6):
            best_valid, best_epoch = cur_valid, epoch
            torch.save(model.cpu(), PATH)

        # if epoch % self.save_every == 0:
        #    torch.save(model.state_dict(), PATH)

        if epoch % 1 == 0:
            print(
                '{}, Epoch: {:>4}, Training Loss: {}, Valid Loss: {}'.format(datetime.datetime.now(), epoch, train_loss,
                                                                             cur_valid))

        # early stop
        # if epoch - best_epoch >= early_stop:
        #    break

    return torch.load(PATH)


def evaluate(model, dataloader, loss_fn, device='cuda'):
    model.eval()
    preds = []
    trues = []
    total_size = 0
    total_loss = 0.0

    with torch.no_grad():
        #############
        for l, _, a, _, v, _, targets in dataloader:
            l, a, v, targets = l.to(device=device, dtype=torch.float), a.to(device=device, dtype=torch.float), v.to(
                device=device, dtype=torch.float), targets.to(device=device, dtype=torch.float)
            outputs = model(l, a, v)
            # targets = targets.unsqueeze(1)
            preds.append(outputs)
            trues.append(targets)
            batch_size = targets.size(0)
            total_size += batch_size
            total_loss += loss_fn(outputs, targets).item() * batch_size
            ####
            functional.reset_net(model)
        ####
        preds = torch.cat(preds, dim=0).cpu().detach()
        trues = torch.cat(trues, dim=0).cpu().detach()

    avg_loss = total_loss / total_size
    model.train()

    return avg_loss, preds, trues


def train_and_eval(config):
    device = 'cuda'
    print(config)
    train_loader, valid_loader, test_loader, model, optimizer, scheduler, loss_fn = initialize(config.dataset,
                                                                                               config.aligned,
                                                                                               config.feature, config)
    model = train(model, optimizer, scheduler, train_loader, valid_loader, loss_fn, config.max_epochs,
                  config.early_stop, config.model_path, device)
    _, preds, trues = evaluate(model.to(device=device), test_loader, loss_fn, device)

    ############################################################################################################
    preds = getBinaryTensor(preds).to(dtype=torch.int)
    test_micro_f1, test_micro_precision, test_micro_recall, test_acc, test_macro_f1, test_hl = get_metrics(preds, trues)
    print('test_A: {}'.format(test_acc))
    print('test_hl: {}'.format(test_hl))
    print('test_miF: {}'.format(test_micro_f1))
    print('test_maF: {}'.format(test_macro_f1))
    print('test_P: {}'.format(test_micro_precision))
    print('test_R: {}'.format(test_micro_recall))
    #############################################################################################################

def getBinaryTensor(imgTensor, boundary=0.35):  # 0.35
    one = torch.ones_like(imgTensor)
    zero = torch.zeros_like(imgTensor)
    return torch.where(imgTensor > boundary, one, zero)
