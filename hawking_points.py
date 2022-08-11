from resnetv2 import PreActResNet34
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from resnetv2 import *
import h5py
import numpy as np
from matplotlib import pyplot as plt

ANGLE_R = 2.0 * np.pi / 180.0

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, file_path):
        super().__init__()
        h5_file = h5py.File(file_path, 'r')
        self.patches = h5_file['patches']
        self.coords = h5_file['coords'] if 'coords' in h5_file else None
    
    def __getitem__(self, index):
        if self.coords is not None:
            return self.patches[index], self.coords[index]
        return self.patches[index]
    
    def __len__(self):
        return self.patches.shape[0]

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(16, 16, 5)
        self.conv3 = torch.nn.Conv2d(16, 16, 5)
        self.fc1 = nn.Linear(16 * 12 * 12, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# model = Net().cuda()
model = PreActResNet18(in_planes=1, num_classes=1).cuda()
model.load_state_dict(torch.load('checkpoint/500.ckpt'))
from torchsummary import summary
summary(model, (1, 128, 128))
# model.load_state_dict(torch.load('400mK.ckpt'))
dataloader = torch.utils.data.DataLoader(MyDataset('dataset_plank.hdf5'), batch_size=128, shuffle=True, drop_last=True)
# dataloader = torch.utils.data.DataLoader(MyDataset('smica_lower_train.hdf5'), batch_size=128, shuffle=True, drop_last=True)
real_dataloader = torch.utils.data.DataLoader(MyDataset('smica_upper_test.hdf5'), batch_size=128)
opt = optim.Adam(model.parameters(), lr=3e-5)

def gen_hp(batch_size):
    grid_x, grid_y = torch.meshgrid(torch.linspace(-ANGLE_R, ANGLE_R, 128), torch.linspace(-ANGLE_R, ANGLE_R, 128))
    sigma = (0.5 + torch.rand((batch_size,))) * np.pi / 180.0
    amplitude = 100.0 * (1 + torch.rand((batch_size,)) * 0.0)
    x0 = (torch.rand((batch_size,)) - 0.5) * ANGLE_R
    y0 = (torch.rand((batch_size,)) - 0.5) * ANGLE_R
    hp = torch.exp(-((grid_x - x0[:, None, None])**2 + (grid_y - y0[:, None, None])**2) / (sigma[:, None, None]**2)) * amplitude[:, None, None]
    return hp.cuda()

# plt.imshow(gen_hp(1)[0].cpu())
# plt.show()

iter = 0
for epoch in range(100):
    eval_iters = 0
    pos_logits = []
    neg_logits = []
    model.eval()
    # for batch in dataloader:
    #     eval_iters += 1
    #     with torch.no_grad():
    #         batch = batch.cuda()
    #         batch_pos = batch[:, None] + gen_hp(batch.shape[0])[:, None]
    #         batch_neg = batch[:, None]
    #         batch_pos -= batch_pos.mean((1, 2, 3), keepdim=True)
    #         batch_neg -= batch_neg.mean((1, 2, 3), keepdim=True)
    #         out = model(torch.cat([batch_neg, batch_pos], axis=0))
    #         neg_out, pos_out = out.split(batch.shape[0], dim=0)
    #         # print(batch.std())
    #     pos_logits.append(pos_out.reshape(-1).cpu().numpy())
    #     neg_logits.append(neg_out.reshape(-1).cpu().numpy())
    #     if eval_iters == 1000:
    #         break

    for batch, coords in dataloader:
        # print(batch.shape)
        # plt.hist(batch.reshape(-1).numpy(), bins=100, alpha=0.5, label='real')
        # break
        eval_iters += 1
        with torch.no_grad():
            batch = batch.cuda()
            batch = batch[:, None]
            batch -= batch.mean((1, 2, 3), keepdim=True)
            out = model(batch)
        neg_logits.append(out.reshape(-1).cpu().numpy())
        if eval_iters == 200:
            break
    for batch, coords in real_dataloader:
        # print(batch.shape)
        # plt.hist(batch.reshape(-1).numpy() * 1e6, bins=100, alpha=0.5, label='sim')
        # break
        with torch.no_grad():
            batch = batch.cuda() * 1e6
            batch = batch[:, None]
            batch -= batch.mean((1, 2, 3), keepdim=True)
            out = model(batch)
        for i in range(len(batch)):
            if out[i, 0] > 5:
                print(coords[i])
                plt.imshow(batch[i, 0].cpu())
                plt.show()
        pos_logits.append(out.reshape(-1).cpu().numpy())
    plt.show()
    pos_logits = np.concatenate(pos_logits)
    neg_logits = np.concatenate(neg_logits)
    plt.hist(pos_logits, bins=200, alpha=0.5, label='Real data', density=True)
    plt.hist(neg_logits, bins=200, alpha=0.5, label='Simulations', density=True)
    plt.legend()
    plt.xlabel('Classification score')
    plt.ylabel('Probability density')
    plt.show()
    bins = np.linspace(-5, 10, 200)
    n_pos, _, _ = plt.hist(pos_logits, bins=bins, alpha=0.5, label='Positive samples', density=True, color='orchid')
    n_neg, _, _ = plt.hist(neg_logits, bins=bins, alpha=0.5, label='Negative samples', density=True, color='cadetblue')
    plt.legend()
    plt.xlabel('Classification score')
    plt.ylabel('Probability density')
    plt.show()
    xs = (bins[:-1] + bins[1:]) / 2
    plt.plot(xs, n_pos / (n_pos + n_neg), label='Empirical', color='r')
    plt.plot(xs, 1.0 / (1 + np.exp(-xs)), label='Theoretical', color='g')
    plt.legend()
    plt.xlabel('Classification score')
    plt.ylabel('Fraction of positive samples')
    plt.show()
    model.train()
    for batch, _ in dataloader:
        opt.zero_grad()

        batch = batch.cuda() * 1e6
        batch_pos = batch[:, None] + gen_hp(batch.shape[0])[:, None]
        batch_neg = batch[:, None]
        batch_pos -= batch_pos.mean((1, 2, 3), keepdim=True)
        batch_neg -= batch_neg.mean((1, 2, 3), keepdim=True)
        # print(model(batch.cuda()[:, None]).shape)
        out = model(torch.cat([batch_neg, batch_pos], axis=0))
        neg_out, pos_out = out.split(batch.shape[0], dim=0)
        neg_loss = -F.logsigmoid(-neg_out).mean()
        pos_loss = -F.logsigmoid(pos_out).mean()
        (pos_loss + neg_loss).backward()
        opt.step()

        if iter % 100 == 0:
            torch.save(model.state_dict(), f'checkpoint/{iter}.ckpt')

        print(float(neg_loss.item()) + float(pos_loss.item()), float((neg_out > 0.0).sum().item()) / batch.shape[0], float((pos_out < 0.0).sum().item()) / batch.shape[0])
        iter += 1

        if iter == 1:
            plt.subplot(121)
            plt.imshow(batch_pos[0, 0].cpu())
            plt.subplot(122)
            plt.imshow(batch_neg[0, 0].cpu())
            plt.show()
