import torch
import torch.nn as nn
import os
import random
from PIL import Image
import torchvision.transforms as transforms


class CarLogoDataset:
    def __init__(self,root_dir):
        self.root_dir = root_dir
        self.classes = os.listdir(root_dir) #Get the names of all brand folders
        self.data = {} #Dictionary

        #Preprocessing, unify the image size, and convert to Tensor
        self.transform = transforms.Compose([
            transforms.Resize(84,84),
            transforms.ToTensor,
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        for cls in self.classes:
            cls_path = os.path.join(root_dir,cls)
            if os.path.isdir(cls_path):
                imgs = [os.path.join(cls_path, i) for i in os.listdir(cls_path) if i.endswith(('.jpg', '.png'))]
                self.data[cls] = imgs
        
    def get_episode(self,n_way,k_shot,q_query):
        """
        Construct a training task (Episode)
        n_way: How many branches are selected.
        k_shot: How many imgs are selected to be Support Set
        q_query: How many imgs are selected to be Query Set to calculate Loss
        """
        #Selecting N branches randomly
        sampled_classes = random.sample(self.classes,n_way)

        support_imgs = []
        query_imgs = []

        for cls in sampled_classes:
            cls_imgs = random.sample(self.data[cls],k_shot + q_query)

            s_files = cls_imgs[:k_shot]
            q_files = cls_imgs[k_shot:]

            support_imgs.extend([self.transform(Image.open(p).convert('RGB')) for p in s_files])
            query_imgs.extend([self.transform(Image.open(p).convert('RGB')) for p in q_files])

        # Support: (N * K, 3, 84, 84)
        # Query:   (N * Q, 3, 84, 84)
        support_imgs = torch.stack(support_imgs)
        query_imgs = torch.stack(query_imgs)

        # Generating Labels
        query_labels = torch.arange(n_way).repeat_interleave(q_query)

        return support_imgs,query_imgs,query_labels

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
        #input: (Batch_Size, 3, 84, 84)
        x = self.encoder(x)
        #x: (Batch_Size, 64, 5, 5)
        return x.view(x.size(0),-1)
    
#Prototypical Loss
def prototypical_loss(support_embeddings, query_embeddings, query_labels, n_way, k_shot, q_query):
    """
    Calculate Loss
    """
    # Prototypes
    # support_embeddings : (N * K, Dim)
    dim = support_embeddings.size(-1)
    prototypes = support_embeddings.view(n_way, k_shot, dim).mean(dim=1)
    
    # Euclidean distance
    dists = torch.cdist(query_embeddings, prototypes)
    
    # Loss
    scores = -dists
    
    criterion = nn.CrossEntropyLoss()
    loss = criterion(scores, query_labels)
    
    preds = scores.argmax(dim=1)
    acc = (preds == query_labels).float().mean()
    
    return loss, acc


#Configuration parameters
ROOT_DIR = "./data" 
N_WAY = 5   
K_SHOT = 5  
Q_QUERY = 5 
EPISODES = 1000 
LR = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = CarLogoDataset(ROOT_DIR)
model = ProtoNetFeatureExtractor().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print(f"Start training... Use the equipment: {device}")

# Training LOOP
for episode in range(EPISODES):
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
        print(f"Episode [{episode+1}/{EPISODES}] | Loss: {loss.item():.4f} | Acc: {acc.item()*100:.2f}%")
torch.save(model.state_dict(), 'proto_car_logo.pth')
print("训练完成！建议保存模型：torch.save(model.state_dict(), 'proto_car_logo.pth')")































