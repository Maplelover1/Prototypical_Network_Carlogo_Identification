import os
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import math
import random
from scipy.stats import sem, t

def get_folder_in_directory(path):
    if not os.path.exists(path):
        return []
    
    all_items = os.listdir(path=path)
    folders = [item for item in all_items if os.path.isdir(os.path.join(path,item))]

    return folders
# 获取所有车标品类为划分做准备
# path = "data"
# folders = get_folder_in_directory(path)
# print(folders) 

# 划分数据集
car_logo_classes = ['Aston Martin', 'Audi', 'Benly', 
                    'BMW', 'buick', 'Cadillac','Chevrolet', 
                    'MINI', 'Renault', 'Rolls Royce', 'Volvo',
                    'Ferrari', 'Honda', 'Jaguar', 'Lamborhini', 'Land Rover', 
                    'Lexus', 'Lincoln', 'Maserati', 'Maybach', 'Mercedes-Benz', ]

TRAIN_CLASSES = ['Aston Martin', 'Audi', 'Benly', 
                    'BMW', 'buick', 'Cadillac','Chevrolet', 
                    'Ferrari', 'Honda', 'Jaguar', 'Lamborhini', 'Land Rover', 
                    'Lexus', 'Lincoln', 'Maserati', 'Maybach', 'Mercedes-Benz',]

TEST_CLASSES = ['MINI', 'Renault', 'Rolls Royce', 'Volvo']

ROOT_DIR = "data"
N_WAY = 4
K_SHOT = 1
Q_QUERY = 15
TEST_EPISODES = 1000

MODEL_PATH = "proto_car_logo_clean.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用的设备: {device}")
print(f"测试任务: {N_WAY}-Way {K_SHOT}-Shot")
print(f"测试品牌: {TEST_CLASSES}")

# Model、Loss等组件(从train文件复制)
# 特征提取模块
def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )
class ProtoNetFeatureExtractor(nn.Module):
    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super(ProtoNetFeatureExtractor, self).__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim)
        )
    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)

# 从 train_car_logos.py 复制过来，只取准确率
def prototypical_accuracy(support_embeddings, query_embeddings, query_labels, n_way, k_shot, q_query):
    dim = support_embeddings.size(-1)
    prototypes = support_embeddings.view(n_way, k_shot, dim).mean(dim=1)
    
    # 欧氏距离
    dists = torch.cdist(query_embeddings, prototypes)
    scores = -dists
    
    preds = scores.argmax(dim=1)
    acc = (preds == query_labels).float().mean()
    
    return acc.item()

# 数据加载器(只使用测试类别)
class TestCarLogoDataset:
    def __init__(self,root_dir,test_classes):
        self.root_dir = root_dir
        self.classes = test_classes
        self.data = {}

        # 测试不适用随机数据增强
        self.transform = transforms.Compose([
        transforms.Resize((84,84)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

        for cls in self.classes:
            cls_path = os.path.join(root_dir,cls)
            if not os.path.isdir(cls_path):
                raise FileNotFoundError(f"测试类别{cls}文件夹不存在，请检查路径")
        
            imgs = [os.path.join(cls_path,i) for i in os.listdir(cls_path) if i.lower().endswith(('.jpg','.png','jpeg'))]
            self.data[cls] = imgs 

        # 确保足够的图片进行测试
            if len(imgs) < K_SHOT + Q_QUERY:
                raise ValueError(f"类别{cls}图片数量不足{K_SHOT + Q_QUERY}张，无法进行测试")

        if len(self.classes) < N_WAY:
            raise ValueError(f"测试所需类别 N_WAY ({N_WAY}) 大于实际发现的有效测试类别数 ({len(self.classes)})。")
    
    def get_episode(self,n_way,k_shot,q_query):
        sampled_classes = random.sample(self.classes,n_way) #从测试集中随机抽取N个

        support_imgs = []
        query_imgs = []

        for cls in sampled_classes:
            cls_imgs = self.data[cls]
            cls_files = random.sample(cls_imgs,k_shot+q_query)

            s_files = cls_files[:k_shot]
            q_files = cls_files[k_shot:]

            support_imgs.extend([self.transform(Image.open(p).convert('RGB')) for p in s_files])
            query_imgs.extend([self.transform(Image.open(p).convert('RGB')) for p in q_files])

        support_imgs = torch.stack(support_imgs)
        query_imgs = torch.stack(query_imgs)

        query_labels = torch.arange(n_way).repeat_interleave(q_query)

        return support_imgs,query_imgs,query_labels
    
# 测试主循环
def test():
    # 初始化模型并加载权重
    model = ProtoNetFeatureExtractor().to(device=device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("成功加载模型权重")
    except FileNotFoundError:
        print("找不到吗模型文件")
        return
    
    # 初始化数据加载器
    try:
        dataset = TestCarLogoDataset(ROOT_DIR,TEST_CLASSES)
    except Exception as e:
        print(f"数据加载错误：{e}")
        return 
    
    model.eval()

    accuracies = []

    print("---------------------------------------")
    print(f"开始 {TEST_EPISODES} 次测试任务...")

    with torch.no_grad():
        for episode in range(TEST_EPISODES):
            support_img, query_img, query_labels = dataset.get_episode(N_WAY, K_SHOT, Q_QUERY)
            support_img = support_img.to(device)
            query_img = query_img.to(device)
            query_labels = query_labels.to(device)

            # 前向传播
            support_emb = model(support_img)
            query_emb = model(query_img)

            # 计算准确率
            acc = prototypical_accuracy(support_emb, query_emb, query_labels, N_WAY, K_SHOT, Q_QUERY)
            accuracies.append(acc)

    # 统计结果并计算置信区间
    mean_acc = np.mean(accuracies)
    std_err = sem(accuracies) #计算标准差

    # 计算95%置信区间
    confidence_level = 0.95
    h = std_err * t.ppf((1 + confidence_level) / 2, len(accuracies) - 1)

    print("\n---------------------------------------")
    print("测试结果")
    print(f"测试任务: {N_WAY}-Way {K_SHOT}-Shot")
    print(f"平均准确率: {mean_acc * 100:.2f} %")
    print(f"95% 置信区间: ± {h * 100:.2f} %")
    print("---------------------------------------")

if __name__ == "__main__":
    test()        




























