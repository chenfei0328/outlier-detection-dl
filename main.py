import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import clip_grad_norm_

from tqdm import tqdm
from datetime import datetime
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, roc_curve, precision_recall_curve
import numpy as np
import matplotlib.pyplot as plt

from dl_models import AE, VAE
from load_data import load_np

np_path = 'train102.npz'
WINDOW_SIZE = 128
EPOCH = 20
BATCH_SIZE = 256
LEARNING_RATE = 1e-3


def gen_batch_loader(windows, label):

    sep1 = int(0.6 * len(windows))
    sep2 = int(0.8 * len(windows))

    windows, label = shuffle(windows, label)

    windows = torch.from_numpy(windows).float()
    label = torch.from_numpy(label).float()

    train_set = TensorDataset(windows[:sep1], label[:sep1])
    train_data = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    # print(len(train_data))

    valid_set = TensorDataset(windows[sep1:sep2], label[sep1:sep2])
    valid_data = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    # print(len(valid_data))

    test_set = TensorDataset(windows[sep2:], label[sep2:])
    test_data = DataLoader(test_set, batch_size=1)

    # test set 不需要打乱，搞成batch？
    # test_set = TensorDataset(windows[sep2:], label[sep2:])
    # test_data = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    # print(len(test_data))

    # for i, batch in enumerate(train_data):
    #     print(i)
    #     # batch[0]为数据,batch[1]为标签
    #     print(batch[0])
    #     print(batch[1])

    return train_data, valid_data, test_data

def train(train_data, valid_data):

    net = VAE()
    # print(net)
    if torch.cuda.is_available():
        net = net.cuda()

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)  # 添加正则项，替代dropout


    # optimizer.param_groups[0]
    # 是其中一个参数组，包括['lr']和['weight_decay']
    # 可以直接赋值修改参数
    # 防止有多个参数组，故使用循环
    # def set_learning_rate(optimizer, lr):
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = lr

    # 每10轮衰减为原来的75%
    def set_learning_rate(optimizer):
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.75


    train_losses = []
    valid_losses = []

    prev_time = datetime.now()

    for e in tqdm(range(EPOCH)):
        if e != 0 and e % 10 == 0:
            set_learning_rate(optimizer)

        train_loss = 0.0
        for i, batch in enumerate(train_data):
            window = batch[0].view(BATCH_SIZE, -1)
            # recon_window = net(window)
            # loss = net.loss_function(recon_window, window)
            recon_window, mu, sigma = net(window)
            loss = net.loss_function(recon_window, window, mu, sigma)

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(net.parameters(), max_norm=10.)
            optimizer.step()
            # use float() to reduce menmery 'autograd
            train_loss += float(loss)

        train_loss /= len(train_data)
        train_losses.append(train_loss)

        valid_loss = 0

        net.eval()

        for i, batch in enumerate(valid_data):
            window = batch[0].view(BATCH_SIZE, -1)
            # recon_window = net(window)
            # loss = net.loss_function(recon_window, window)
            recon_window, mu, sigma = net(window)
            loss = net.loss_function(recon_window, window, mu, sigma)
            valid_loss = float(loss)

            break

        valid_losses.append(valid_loss)

        if (e + 1) % 5 == 0:
            cur_time = datetime.now()
            h, remainder = divmod((cur_time - prev_time).seconds, 3600)
            m, s = divmod(remainder, 60)
            time_str = "Time %02d:%02d:%02d" % (h, m, s)

            prev_time = cur_time
            # print('Epoch: {}, Train Loss: {:.4f} '.format(e + 1, train_loss) + time_str)
            print('Epoch: {}, Train Loss: {:.4f}, Valid Loss: {:.4f} '.format(e + 1, train_loss, valid_loss) + time_str)
        net.train()

    final_loss = train_losses[-1]
    return net, train_losses, valid_losses

def predict(net, test_data):
    predictions, labels = np.array([]), np.array([])
    for i, batch in enumerate(test_data):
        window = batch[0].view(1, -1)
        label = batch[1].view(1)
        # recon_window = net(window)
        # loss = net.loss_function(recon_window, window)
        recon_window, mu, sigma = net(window)
        loss = net.loss_function(recon_window, window, mu, sigma)
        predictions = np.concatenate((predictions, np.array([loss])))
        labels = np.concatenate((labels, label))
        #print(label)


    predictions = np.array([1 if x >= 3.3 else 0 for x in predictions])
    predictions = predictions.reshape(1, -1)
    labels = labels.reshape(1, -1)
    print(predictions.shape)
    print(labels.shape)
    print(np.sum(predictions))
    # print(predictions*labels)
    print(classification_report(labels, predictions))
    # 140
    # print(np.sum(labels))
    # fpr, tpr, th = roc_curve(labels, predictions)
    # plt.subplot(2, 1, 1)
    # plt.plot(fpr, tpr)
    # plt.title('ROC curve')
    #
    # pre, recall, th = precision_recall_curve(labels, predictions)
    # print(pre)
    # print(recall)
    #
    # plt.subplot(2, 1, 2)
    # plt.plot(recall, pre)
    # plt.ylim([0.0, 1.0])
    # plt.title('Precision recall curve')
    #
    # plt.show()



def plot_losses(train_losses, valid_losses):
    plt.figure(figsize=(20, 10))
    plt.plot(train_losses, label='train')
    plt.plot(valid_losses, label='valid')
    plt.xlabel('epoch')
    plt.legend(loc='best')
    plt.show()

def save_model(net, path):
    torch.save(net, path)

def restore_net(path):
    new_net = torch.load(path)
    return new_net

if __name__ == '__main__':
    windows, label = load_np(np_path)
    train_data, valid_data, test_data = gen_batch_loader(windows, label)
    # # for i, batch in enumerate(test_data):
    # #     # print(i)
    # #     # batch[0]为数据,batch[1]为标签
    # #     #print(batch[0])
    # #     print(batch[1])
    #
    # net, train_losses, valid_losses = train(train_data, valid_data)
    # plot_losses(train_losses, valid_losses)
    #
    path = 'vae.pkl'
    # save_model(net, path)

    new_net = restore_net(path)
    new_net.eval()
    predict(new_net, test_data)
