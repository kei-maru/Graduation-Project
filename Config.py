# config.py
import torch

# 数据集路径
TRAIN_FILE = 'weibo_dataset/train/usual_train.txt'
Augmented_train_data = 'weibo_dataset/train/augmented_train_data.json'
TEST_FILE = 'weibo_dataset/test/test_real/usual_test_labeled.txt/'

# 模型超参数
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 5e-5

# 设备配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
