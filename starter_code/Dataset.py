from utils import load_case
from torch.utils.data import DataLoader, Dataset


# 读取三面的矩阵信息
# 预处理label 也是3面的，二值化#
# 网络设计
# 损失函数+优化器


def Getcasedata(case):  #读取某个病人的信息
    vol, seg = load_case(case)
    spacing = vol.affine  #仿射
    vol = vol.get_data()  # 获取体素数据（CT值）3维单通道
    seg = seg.get_data()
    return  vol, seg

    

class ImgDataset(Dataset):  #这个在图像数据的处理中很方便，而且必须要经过继承
    def __init__(self, x, y=None, transform=None, target_transform=None):
        self.x = x    #x放入data
        self.y = y    #y放入label
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):    #数据集的长度
        return len(self.x)

    def __getitem__(self, index):  #重写数据集的索引
        res_x = self.x[index]
        if self.transform is not None:
            res_x = self.transform(res_x)

        if self.y is not None:
            res_y = self.y[index]
            if self.target_transform is not None:
                res_y = self.target_transform(res_y)
            return res_x, res_y
        else:  # 如果没有标签那么只返回x
            return res_x

