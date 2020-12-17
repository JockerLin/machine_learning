import torch
import time, os
from torch import nn  # 常用网络
from torch import optim  # 优化工具包
import torchvision  # 视觉数据集
from matplotlib import pyplot as plt
import glob, cv2
import numpy as np
from torch.autograd import Variable

## 加载数据
batch_size = 512
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data',train=True,download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((0.1307,), (0.3081,))  # 做一个标准化
                                   ])),
    batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data/', train=False,download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                   ])),
    batch_size=batch_size,shuffle=True)
x, y = next(iter(train_loader))
print(x.shape,y.shape,x.min(),x.max())

# 可视化 show
# fig = plt.figure()
# for i in range(6):
#   plt.subplot(2,3,i+1)
#   plt.tight_layout()
#   plt.imshow(x[i][0], cmap='gray', interpolation='none')
#   plt.title("Ground Truth: {}".format(y[i]))
#   plt.xticks([])
#   plt.yticks([])
# plt.show()

relu = nn.ReLU()  # 如果使用torch.sigmoid作为激活函数的话正确率只有60%
# 创建网络

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()

        # xw+b  这里的256,64使我们人根据自己的感觉指定的
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # 因为找不到relu函数，就换成了激活函数
        # x:[b,1,28,28]
        # h1 = relu(xw1+b1)
        x = relu(self.fc1(x))
        # h2 = relu(h1w2+b2)
        x = relu(self.fc2(x))
        # h3 = h2*w3+b3
        x = self.fc3(x)

        return x

# 因为找不到自带的one_hot函数，就手写了一个
def one_hot(label, depth=10):
    out = torch.zeros(label.size(0), depth)
    idx = torch.LongTensor(label).view(-1, 1)
    out.scatter_(dim=1, index=idx, value=1)
    return out

## 训练模型
net = Net()
# 返回[w1,b1,w2,b2,w3,b3]  对象，lr是学习过程
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

train_loss = []
mes_loss = nn.MSELoss()

def valTest():
    ## 准确度测试
    total_correct = 0
    for x, y in test_loader:
        x = x.view(x.size(0), 28*28)
        out = net(x)
        # out : [b,10]  =>  pred: [b]
        pred = out.argmax(dim = 1)
        correct = pred.eq(y).sum().float().item()  # .float之后还是tensor类型，要拿到数据需要使用item()
        total_correct += correct
    total_num = len(test_loader.dataset)
    acc = total_correct/total_num
    print('准确率acc:', acc)


def training():
    for epoch in range(10):
        for batch_idx, (x, y) in enumerate(train_loader):
            # x:[b,1,28,28],y:[512]
            # [b,1,28,28]  =>  [b,784]
            x = x.view(x.size(0), 28 * 28)
            # =>[b,10]
            out = net(x)
            # [b,10]
            y_onehot = one_hot(y)
            # loss = mse(out,y_onehot)
            loss = mes_loss(out, y_onehot)

            # 清零梯度
            optimizer.zero_grad()
            # 计算梯度
            loss.backward()
            # w' = w -lr*grad
            # 更新梯度，得到新的[w1,b1,w2,b2,w3,b3]
            optimizer.step()

            train_loss.append(loss.item())
            if batch_idx % 10 == 0:
                print("epoch:{}, batch:{}, loss:{}".format(epoch, batch_idx, loss.item()))
                valTest()

    cur_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    file_name = "./checkpoint/{}-ckpt.pth".format(cur_time)
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(net.state_dict(), file_name)


def FolderImages():
    checkpoint = torch.load('./checkpoint/2020-12-17-11-38-48-ckpt.pth')  # 加载模型
    net.load_state_dict(checkpoint)

    for jpgfile in glob.glob(r'./testnum/*.png'):
        print(jpgfile)  # 打印图片名称，以与结果进行对照
        img = cv2.imread(jpgfile, cv2.IMREAD_GRAYSCALE)  # 读取要预测的图片，读入的格式为BGR
        image = cv2.resize(img, (28, 28))
        height, width = image.shape
        dst = np.zeros((height, width), np.uint8)
        for i in range(height):
            for j in range(width):
                dst[i, j] = 255 - image[i, j]

        image = dst
        # cv读入数据是32x32x3
        # cv2.imshow("f", image)
        # cv2.waitKey(0)
        image = np.expand_dims(image, 0).astype(np.float32)
        image = np.expand_dims(image, 0).astype(np.float32)

        tensor_image = torch.from_numpy(image)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tensor_image = tensor_image.to(device)

        # torch.transpose(tensor_image, 2, 3)

        tensor_image = tensor_image.view(tensor_image.size(0), 28 * 28)
        # torch.set_default_tensor_type(torch.DoubleTensor)
        # 模型要求的输入tensor size是(1,3,32,32)

        output = net(Variable(tensor_image))
        _, predicted = output.max(1)
        print("预测的数字为:", predicted[0])

# plot_curve(train_loss)
# 到现在得到了[w1,b1,w2,b2,w3,b3]

def plot_image(img, label, name):
    fig = plt.figure()
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.tight_layout()
        plt.imshow(img[i][0] * 0.3081 + 0.1307, cmap='gray', interpolation='none')
        plt.title("{}: {}".format(name, label[i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()

def visionUse():
    checkpoint = torch.load('./checkpoint/2020-12-17-11-38-48-ckpt.pth')  # 加载模型
    net.load_state_dict(checkpoint)

    image_num = 8

    # dataiter = iter(test_loader)
    # images, labels = dataiter.next()  # 返回4张图片及标签
    # print(' '.join('%11s' % labels[j] for j in range(image_num)))

    # x = images[:image_num]
    # x = x.view(x.size(0), 28 * 28)
    # outputs = net(x)
    # _, predicted = outputs.max(1)
    # print('预测结果: ', ' '.join('%5s' % predicted[j] for j in range(image_num)))

    x, y = next(iter(test_loader))
    out = net(x.view(x.size(0), 28 * 28))
    pred = out.argmax(dim=1)
    plot_image(x, pred, 'test')

    # input_dt = test_loader[0][:6]
    # out = net(input_dt)
    # pred = out.argmax(dim=1)

    # for x, y in test_loader[:6]:
    #     x = x.view(x.size(0), 28*28)
    #     out = net(x)
    #     # out : [b,10]  =>  pred: [b]
    #     pred = out.argmax(dim = 1)
    #     correct = pred.eq(y).sum().float().item()  # .float之后还是tensor类型，要拿到数据需要使用item()
    #     # total_correct += correct

    # fig = plt.figure()
    # for i in range(6):
    #   plt.subplot(2,3,i+1)
    #   plt.tight_layout()
    #   plt.imshow(x[i][0], cmap='gray', interpolation='none')
    #   plt.title("Ground Truth: {}".format(y[i]))
    #   plt.xticks([])
    #   plt.yticks([])
    # plt.show()

# training()
# FolderImages()
# checkpoint = torch.load('./checkpoint/2020-12-17-11-38-48-ckpt.pth')  # 加载模型
# net.load_state_dict(checkpoint)
# valTest()
visionUse()

