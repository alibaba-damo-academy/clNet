# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import argparse
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import contextlib
import torch.cuda.amp
import time
from clnet.sparse_conv.SparseConvWraper import *



@contextlib.contextmanager
def identity_ctx():
    yield


class NetDense(nn.Module):
    def __init__(self):
        super(NetDense, self).__init__()
        net1 = nn.Sequential(
            nn.ConvTranspose2d(1, 32, kernel_size=2, stride=2),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.MaxPool2d(2, 2),
            nn.MaxPool2d(2, 2)
        )
        net2 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            # nn.MaxPool2d(2, 2),
        )
        net3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU()
        )
        self.net = nn.ModuleDict({'net1': net1, 'net2': net2, 'net3': net3})
        self.bottle = nn.Conv2d(128, 10, kernel_size=8, padding=0)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1_1 = nn.Linear(4 * 4 * 64, 128)
        self.fc1_2 = nn.Linear(128, 10)
        self.fc2_1 = nn.Linear(4 * 4 * 64, 64)
        self.fc2_2 = nn.Linear(64, 10)
        self.fc = nn.Linear(20, 10)
        self.dropout1 = nn.Dropout1d(0.25)
        self.dropout2 = nn.Dropout1d(0.5)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        x1 = self.net["net1"](x1)
        x2 = self.net["net2"](x2)

        x_c = torch.concat((x1, x2), 1)
        num_acc = []
        for i in range(5):
            x = self.net["net3"](x_c)
        # print("Cat", isinstance(x, spconv.SparseConvTensor))
            x = self.bottle(x)
            # print("bottle", isinstance(x, spconv.SparseConvTensor))
            x = torch.flatten(x, 1)
            num_acc.append(x)
        # x = self.dropout1(x)
        # x1 = self.fc1_1(x)
        # x1 = F.relu(x1)
        # x1 = self.dropout2(x1)
        # x1 = self.fc1_2(x1)
        # #
        # x2 = self.fc2_1(x)
        # x2 = F.relu(x2)
        # x2 = self.dropout2(x2)
        # x2 = self.fc2_2(x2)
        #
        # x = torch.concat((x1, x2), 1)
        # x = self.fc(x)
        output = 0
        for x in num_acc:
            output += F.log_softmax(x, dim=1)
        # output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    scaler = torch.cuda.amp.grad_scaler.GradScaler()
    amp_ctx = contextlib.nullcontext()
    if args.fp16:
        amp_ctx = torch.cuda.amp.autocast()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data[:, :, 0:8, 0:8]
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        with amp_ctx:
            output = model(data, data)
            loss = F.nll_loss(output, target)
            scale = 1.0
            if args.fp16:
                assert loss.dtype is torch.float32
                scaler.scale(loss).backward()
                # scaler.step() first unscales the gradients of the optimizer's assigned params.
                # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
                # otherwise, optimizer.step() is skipped.
                # scaler.unscale_(optim)

                # Since the gradients of optimizer's assigned params are now unscaled, clips as usual.
                # You may use the same value for max_norm here as you would without gradient scaling.
                # torch.nn.utils.clip_grad_norm_(models[0].net.parameters(), max_norm=0.1)

                scaler.step(optimizer)
                # Updates the scale for next iteration.
                scaler.update()
                scale = scaler.get_scale()
            else:
                loss.backward()
                optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    amp_ctx = contextlib.nullcontext()
    if args.fp16:
        amp_ctx = torch.cuda.amp.autocast()

    with torch.no_grad():
        for data, target in test_loader:
            data = data[:, :, 0:8, 0:8]
            data, target = data.to(device), target.to(device)
            with amp_ctx:
                output = model(data, data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size',
                        type=int,
                        default=64,
                        metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size',
                        type=int,
                        default=1000,
                        metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs',
                        type=int,
                        default=5,
                        metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr',
                        type=float,
                        default=1.0,
                        metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma',
                        type=float,
                        default=0.7,
                        metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda',
                        action='store_true',
                        default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        metavar='N',
        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model',
                        action='store_true',
                        default=False,
                        help='For Saving the current Model')
    parser.add_argument('--fp16',
                        action='store_true',
                        default=False,
                        help='For mixed precision training')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            '../data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                # here we remove norm to get sparse tensor with lots of zeros
                # transforms.Normalize((0.1307,), (0.3081,))
            ])),
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            '../data',
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                # here we remove norm to get sparse tensor with lots of zeros
                # transforms.Normalize((0.1307,), (0.3081,))
            ])),
        batch_size=args.test_batch_size,
        shuffle=True,
        **kwargs)

    # model_dense = NetDense().to("cpu")
    threshold = 1e-8
    model_dense = NetDense().to(device)
    optimizer = optim.Adadelta(model_dense.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    # time_s = time.time()
    # for epoch in range(1, args.epochs + 1):
    #     train(args, model_dense, device, train_loader, optimizer, epoch)
    #     test(args, model_dense, device, test_loader)
    #     scheduler.step()
    #     log_memory_usage("Dense")
    # time_e = time.time()
    # print("Dense Time: ", time_e - time_s)

    for name, module in model_dense.named_modules():
        if isinstance(module, nn.Conv2d):
            module.weight.data[torch.abs(module.weight.data) < threshold] = 0
    test(args, model_dense, device, test_loader)

    model_dense = model_dense.to("cpu")
    # overall_sparsity, layer_sparsity = calculate_sparsity(model_dense, threshold)
    # print(f"Overall Model Sparsity: {overall_sparsity:.2f}%")
    # print("Layer-wise Sparsity:")
    # for name, info in layer_sparsity.items():
    #     print(f"{name}: {info['sparsity_percent']:.2f} %
    #     ({info['near_zero_params']}/{info['total_params']} near-zero params)")

    model_sparse = ModelSparse(model_dense, sparsity_threshold=threshold, subm_conv=False)
    # model_sparse = torch.compile(model_sparse, backend="inductor", mode="default")
    model_sparse.to(device)
    model_sparse.train()
    optimizer = optim.Adadelta(model_sparse.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    del model_dense

    test(args, model_sparse, device, test_loader)
    time_s = time.time()
    for epoch in range(1, args.epochs + 1):
        train(args, model_sparse, device, train_loader, optimizer, epoch)
        test(args, model_sparse, device, test_loader)
        scheduler.step()
        log_memory_usage("Sparse")
    time_e = time.time()
    print("Sparse Time: ", time_e - time_s)


if __name__ == '__main__':
    main()
