from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
from Network import UNet2D
import time
import numpy as np
from Dataset import Getcasedata , ImgDataset
from evaluation import evaluate
from torchvision import transforms
from visualize import class_to_color,overlay,hu_to_grayscale  #可视化结果
from imageio import imwrite

gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
img_transforms = transforms.Compose([   # 对于输入图形，既需要进行归一化，又需要改它的均值啥的
        transforms.ToTensor(),         # 这个顺便做了一个归一化
        transforms.Normalize((0.5,), (0.5,))    # 这个是在改均值和方差，从而把范围变到了-1到1上
    ])

target_transforms = transforms.Compose([
    #transforms.ToTensor(),
])


def train_model(train_loader):
    model = UNet2D(1, 3).to(gpu)
    # 记住之后必须在每一个步骤向GPU发送输入和目标
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    num_epoch = 50    # 迭代次数

    print("start training!")
    for epoch in range(num_epoch):
        epoch_start_time = time.time()  # 获取当前时间
        train_loss = 0.0

        model.train()
        for x, y in train_loader:
            inputs = x.type(torch.FloatTensor).to(gpu)
            label = y.type(torch.LongTensor).to(gpu)
            optimizer.zero_grad()  # 清空之前的梯度

            outputs = model(inputs)
            outputs = torch.sigmoid(outputs)   #网络中没有将输出归到0-1，这里用一下，再使用交叉熵
            batch_loss = loss(outputs, label)

            current_batchsize = outputs.size()[0]

            batch_loss.backward()  # 获取梯度,反向传播
            optimizer.step()  # 将参数更新值施加到net的parmeters上

            # train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            train_loss += batch_loss.item()*current_batchsize  # 张量中只有一个值就可以使用item()方法读取

        print('[%03d/%03d] %2.2f sec(s)  Loss: %3.6f' % \
              (epoch + 1, num_epoch, time.time() - epoch_start_time, train_loss / train_set.__len__()))
    return model


def predict(caseid, model):
    test_vol, train_seg = Getcasedata(caseid)
    test_set = ImgDataset(test_vol, transform=img_transforms)
    test_load = DataLoader(test_set,batch_size=1)
    model.eval()

    result = np.zeros((test_vol.shape[0],test_vol.shape[1],test_vol.shape[2],3))  #不限制类型就要套2层括号，有的后面还要指定一个参数类型
    with torch.no_grad():
        for i, data in enumerate(test_load):
            test_pred = model(data.type(torch.FloatTensor).to(gpu))
            if(i%100 == 0):
                print(i)
            result[i] = np.reshape(test_pred[0].cpu(),(512,512,3))  # result.shape:slice *width*height*3

    predictions = np.argmax(result, axis=-1)
    predictions = predictions.astype(np.int32)

    result_picture = class_to_color(predictions, [255, 0, 0], [0, 255, 0])
    vol_ims = hu_to_grayscale(test_vol, -512, 512)
    viz_ims = overlay(vol_ims, result_picture, predictions, 0.3)
    for i in range(predictions.shape[0]):
        fpath = ("predictions/{:05d}.png".format(i))
        imwrite(str(fpath),viz_ims[i])

    return evaluate(caseid,result)

if __name__ == '__main__':
    train_vol,train_seg = Getcasedata("case_00000")
    train_set = ImgDataset(train_vol,train_seg,img_transforms,target_transforms)
    train_load = DataLoader(train_set, batch_size = 4)

    model = train_model(train_load)
    print("保存模型！")
    torch.save(model.state_dict(), "model.t7") #保存模型

    '''
    print("载入模型")

    model = UNet2D(1, 3).to(gpu)
    model.load_state_dict(torch.load("model.t7"))
    caseid= "case_00001"

    tk_dice, tu_dice = predict(caseid,model)
    print(tk_dice)
    print(tu_dice)
    '''