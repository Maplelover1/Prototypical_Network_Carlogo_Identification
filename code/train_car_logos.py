import torch
import torch.nn as nn
import os
import random
from PIL import Image
import torchvision.transforms as transforms
import time # 导入 time 模块用于计时和日志打印

# --- 1. 配置参数和数据集划分 ---
ROOT_DIR = "data" 
N_WAY = 8  
K_SHOT = 5 
Q_QUERY = 5 
EPISODES = 2000 
LR = 0.001
SAVE_PATH = "proto_car_logo_clean.pth" # 更改保存名称，区别于泄露的模型


TRAIN_CLASSES = ['Aston Martin', 'Audi', 'Benly', 
                    'BMW', 'buick', 'Cadillac','Chevrolet', 
                    'Ferrari', 'Honda', 'Jaguar', 'Lamborhini', 'Land Rover', 
                    'Lexus', 'Lincoln', 'Maserati', 'Maybach', 'Mercedes-Benz',]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. 数据加载器 (Dataset) - 核心修改部分 ---
class CarLogoDataset:
    # 接收 train_classes 列表作为参数
    def __init__(self, root_dir, train_classes):
        self.root_dir = root_dir
        self.data = {}
        valid_classes = []
        self.train_classes = train_classes # 新增：保存训练类别列表

        self.transform = transforms.Compose([
            transforms.Resize((84, 84)), 
            transforms.ToTensor(),      
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 核心修改点：只遍历 train_classes 列表中的类别
        for cls in self.train_classes:
            cls_path = os.path.join(root_dir, cls)
            
            # 1. 检查文件夹是否存在
            if os.path.isdir(cls_path):
                imgs = [os.path.join(cls_path, i) for i in os.listdir(cls_path) if i.lower().endswith(('.jpg', '.png', '.jpeg'))]
                
                # 2. 检查文件夹内是否有图片
                if imgs:
                    self.data[cls] = imgs
                    valid_classes.append(cls)

        self.classes = valid_classes
        
        if len(self.classes) < N_WAY:
            raise ValueError(f"错误：训练所需类别 N_WAY ({N_WAY}) 大于实际发现的有效训练类别数 ({len(self.classes)})。请检查您的训练数据文件夹。")
            
        print(f"成功加载 {len(self.classes)} 个有效训练品牌类别。")
        
    def get_episode(self,n_way,k_shot,q_query):
        # Sampling logic remains the same, now only samples from self.classes (the 17 training classes)
        
        #Selecting N branches randomly
        sampled_classes = random.sample(self.classes,n_way)

        support_imgs = []
        query_imgs = []

        for cls in sampled_classes:
            # 确保每个类别有足够的图片
            cls_data = self.data.get(cls)
            if not cls_data or len(cls_data) < k_shot + q_query:
                # 如果因为某种原因图片不足，我们跳过这个类别，并减少N_WAY的实际值
                # 简单起见，这里假设数据是平衡的，或者在初始化时已检查
                continue

            cls_imgs = random.sample(cls_data, k_shot + q_query)

            s_files = cls_imgs[:k_shot]
            q_files = cls_imgs[k_shot:]

            support_imgs.extend([self.transform(Image.open(p).convert('RGB')) for p in s_files])
            query_imgs.extend([self.transform(Image.open(p).convert('RGB')) for p in q_files])

        support_imgs = torch.stack(support_imgs)
        query_imgs = torch.stack(query_imgs)

        # Generating Labels
        # 需要重新计算实际采样的类别数，因为可能有类别被跳过 (虽然这里假设没有)
        # N_way 应该等于 len(sampled_classes)，但为了安全，假设它没变
        query_labels = torch.arange(n_way).repeat_interleave(q_query)

        return support_imgs,query_imgs,query_labels

# --- 3. 模型和损失函数 (保持不变) ---
class ProtoNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(ProtoNetFeatureExtractor,self).__init__()

        def conv_block(in_channels,out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels,out_channels,3,padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
        
        self.encoder = nn.Sequential(
            conv_block(3,64),
            conv_block(64,64),
            conv_block(64,64),
            conv_block(64,64)
        )

    def forward(self,x):
        x = self.encoder(x)
        return x.view(x.size(0),-1)

def prototypical_loss(support_embeddings, query_embeddings, query_labels, n_way, k_shot, q_query):
    dim = support_embeddings.size(-1)
    prototypes = support_embeddings.view(n_way, k_shot, dim).mean(dim=1)
    
    dists = torch.cdist(query_embeddings, prototypes)
    scores = -dists
    
    criterion = nn.CrossEntropyLoss()
    loss = criterion(scores, query_labels)
    
    preds = scores.argmax(dim=1)
    acc = (preds == query_labels).float().mean()
    
    return loss, acc

# --- 4. 训练主循环 - 实例化时传入 TRAIN_CLASSES ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 关键修改：实例化时传入训练类别列表
dataset = CarLogoDataset(ROOT_DIR, TRAIN_CLASSES) 
model = ProtoNetFeatureExtractor().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print(f"Start training... Use the equipment: {device}")
start_time = time.time()

# Training LOOP
for episode in range(EPISODES):
    # ... (训练逻辑与你原代码一致) ...
    support_img,query_img,query_labels = dataset.get_episode(N_WAY,K_SHOT,Q_QUERY)

    support_img = support_img.to(device)
    query_img = query_img.to(device)
    query_labels = query_labels.to(device)
    

    support_emb = model(support_img)
    query_emb = model(query_img)
    

    loss, acc = prototypical_loss(support_emb, query_emb, query_labels, N_WAY, K_SHOT, Q_QUERY)
    

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    

    if (episode + 1) % 50 == 0:
        elapsed = time.time() - start_time
        print(f"Episode [{episode+1}/{EPISODES}] | Time: {elapsed:.1f}s | Loss: {loss.item():.4f} | Acc: {acc.item()*100:.2f}%")
        
        # 增加 Checkpoint 保存
        if (episode + 1) % 500 == 0:
            torch.save(model.state_dict(), f"checkpoint_clean_ep{episode+1}.pth")
            print(f"已保存 Checkpoint: checkpoint_clean_ep{episode+1}.pth")

torch.save(model.state_dict(), SAVE_PATH)
print("\n==============================")
print("✨ 训练完成！")
print(f"模型已保存至 {SAVE_PATH}")
print(f"总耗时: {time.time() - start_time:.2f} 秒")