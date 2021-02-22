import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from util.utils import get_index_from_one_hot_label

from torch.utils.data import DataLoader

import abc

LOSS_ACC_BATCH_SIZE = 100  # When computing loss and accuracy, use blocks of LOSS_ACC_BATCH_SIZE



# pytorch实现tf.nn.lrn
class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=False):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if self.ACROSS_CHANNELS:
            self.average = nn.AvgPool3d(kernel_size=(local_size, 1, 1),  # 0.2.0_4会报错，需要在最新的分支上AvgPool3d才有padding参数
                                        stride=1,
                                        padding=(int((local_size - 1.0) / 2), 0, 0))
        else:
            self.average = nn.AvgPool2d(kernel_size=local_size,
                                        stride=1,
                                        padding=int((local_size - 1.0) / 2))
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)  # 这里的1.0即为bias
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x


class ModelCNNMnist(nn.Module):
    def __init__(self):
        super(ModelCNNMnist, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # LRN(4, alpha=0.001/0.9, beta=0.75,ACROSS_CHANNELS=True)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            # LRN(4, alpha=0.001/0.9, beta=0.75,ACROSS_CHANNELS=True),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(7 * 7 * 32, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.layer2(self.layer1(x))
        x = x.view(-1, 7 * 7 * 32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def get_weight_dimension(self):
        Total_params = 0
        Trainable_params = 0
        NonTrainable_params = 0
        for param in self.parameters():
            mulValue = np.prod(param.size())  # 使用numpy prod接口计算参数数组所有元素之积
            Total_params += mulValue  # 总参数量
            if param.requires_grad:
                Trainable_params += mulValue  # 可训练参数量
            else:
                NonTrainable_params += mulValue  # 非可训练参数量
        print(f'Total params: {Total_params}')
        print(f'Trainable params: {Trainable_params}')
        print(f'Non-trainable params: {NonTrainable_params}')
        return Total_params

    def loss(self, train_image, train_label, w, sample_indices=None):

        model = ModelCNNMnist()
        model.load_state_dict(w)
        model.eval()

        if sample_indices is None:
            sample_indices = range(0, len(train_label))

        train_loss = []
        criterion = torch.nn.CrossEntropyLoss().cuda()
        # train_loader = DataLoader(train_data, batch_size=128, shuffle=False)

        for i in sample_indices:
            images = train_image[i].reshape(1,1,28,28)
            labels = get_index_from_one_hot_label(train_label[i])
            images, labels = torch.Tensor(images), torch.Tensor(labels).type(torch.long)
            outputs = model(images)

            batch_loss = criterion(outputs, labels)
            train_loss.append(batch_loss.item())

        return np.mean(train_loss)

    def loss_from_prev_gradient_computation():
        pass

    def accuracy(self, test_image, test_label, w, sample_indices=None):
        
        model = ModelCNNMnist()
        model.load_state_dict(w)
        model.eval()

        if sample_indices is None:
            sample_indices = range(0, len(test_label))

        test_acc, correct, total = 0.0, 0.0, 0.0
        # criterion = torch.nn.CrossEntropyLoss().cuda()
        # test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

        for i in sample_indices:
            images = train_image[i].reshape(1,1,28,28)
            labels = get_index_from_one_hot_label(train_label[i])
            images, labels = torch.Tensor(images), torch.Tensor(labels).type(torch.long)

            outputs = model(images)

            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)

            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        test_acc = correct / total
        return test_acc

    def gradient(self, train_image, train_label, w, sample_indices=None):
        
        model = ModelCNNMnist()
        model.load_state_dict(w)
        # model.eval()
        if sample_indices is None:
            sample_indices = range(0, len(train_label))

        criterion = torch.nn.CrossEntropyLoss().cuda()

        for i in sample_indices:
            images = train_image[i].reshape(1,1,28,28)
            labels = get_index_from_one_hot_label(train_label[i])
            images, labels = torch.Tensor(images), torch.Tensor(labels).type(torch.long)
            outputs = model(images)
            batch_loss = criterion(outputs, labels)
            batch_loss.backward()

        gradient_val_list = []
        for param in model.parameters():
            gradient_val_list.append(param.grad.numpy().reshape(-1, 1).flatten())
        gradient_flatten_list = [temp for l in gradient_val_list for temp in l]
        return np.array(gradient_flatten_list)


    def update_w(self, train_image, train_label, w, step_size, sample_indices=None):

        model = ModelCNNMnist()
        model.load_state_dict(w)

        if sample_indices is None:
            sample_indices = range(0, len(train_label))
        
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=step_size)

        for i in sample_indices:
            images = train_image[i].reshape(1,1,28,28)
            labels = get_index_from_one_hot_label(train_label[i])
            images, labels = torch.Tensor(images), torch.Tensor(labels).type(torch.long)
            outputs = model(images)

            optimizer.zero_grad()
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

        return model.state_dict()




