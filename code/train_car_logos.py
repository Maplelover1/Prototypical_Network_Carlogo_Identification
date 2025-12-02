import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
import os
import shutil
import time
import copy
from tqdm import tqdm

# 配置和路径
INPUT_ROOT_DIR = r'D:\Car_crawler\annotations' # 包含 Alfa Romeo, Bugatti 等品牌文件夹的父目录
DATA_TEMP_DIR = 'data_split_temp'               # 临时数据集划分目录 (将自动创建)
MODEL_NAME = 'resnet18'                         # 使用的模型
NUM_EPOCHS = 20                                 # 训练周期数
BATCH_SIZE = 32                                 # 批次大小
LEARNING_RATE = 0.001                           # 学习率
TRAIN_RATIO = 0.7                               # 训练集比例
VAL_RATIO = 0.15                                # 验证集比例
TEST_RATIO = 0.15                               # 测试集比例 (剩余部分)
SAVE_PATH = 'car_logo_classifier.pth'           # 模型保存路径





































