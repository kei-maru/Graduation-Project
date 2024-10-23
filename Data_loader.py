# data_loader.py

import json
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from Config import MAX_LENGTH, BATCH_SIZE
import nlpaug.augmenter.word as naw

# 数据增强增强器（同义词替换）
augment_word = naw.RandomWordAug(action="delete")

# 引入情感词典
def load_emotion_dict():
    return {
        'angry': ['愤怒', '生气', '气愤', '恼火', '火大', '愠怒'],
        'happy': ['开心', '快乐', '高兴', '愉快', '喜悦'],
        'neutral': ['普通', '正常', '一般', '无感'],
        'surprise': ['惊讶', '震惊', '意外', '惊喜'],
        'sad': ['悲伤', '难过', '沮丧', '伤心', '失落', '痛苦'],
        'fear': ['害怕', '恐惧', '担忧', '紧张', '恐慌']
    }

# 使用情感词典为文本打情感标签
def get_emotion_feature(text, emotion_dict):
    for emotion, words in emotion_dict.items():
        if any(word in text for word in words):
            return emotion
    return 'neutral'  # 如果没有匹配，中立

def encode_emotion_feature(df):
    emotion_encoder = LabelEncoder()
    df['emotion_feature_encoded'] = emotion_encoder.fit_transform(df['emotion_feature'])
    return df, emotion_encoder

def label_to_id():
    return {
        'angry': 0,
        'happy': 1,
        'neutral': 2,
        'surprise': 3,
        'sad': 4,
        'fear': 5
    }
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return pd.DataFrame(data)


# 文本增强函数，n 表示增强次数
def augment_text(text, augmenter, n=1):
    augmented_texts = [augmenter.augment(text) for _ in range(n)]
    return augmented_texts


# 数据增强函数：针对少数类进行数据增强
def augment_fear_class(df, target_class, augmenter, target_size):
    class_df = df[df['label'] == target_class]
    augmented_texts = []

    # 计算需要的增强样本数量
    augment_count = target_size - len(class_df)

    if augment_count > 0:
        # 对每个文本进行数据增强
        for idx, row in class_df.iterrows():
            augmented_texts.extend(augment_text(row['content'], augmenter, n=augment_count // len(class_df) + 1))

        # 只取所需的数量
        augmented_texts = augmented_texts[:augment_count]

        # 创建增强后的 DataFrame
        augmented_df = pd.DataFrame({'content': augmented_texts, 'label': [target_class] * len(augmented_texts)})

        # 将增强数据添加回原始数据
        df = pd.concat([df, augmented_df])

    return df

# 保存增强后的数据
def save_augmented_data(df, folder_path, filename):
    # 如果文件夹不存在，则创建
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # 将增强后的 DataFrame 保存为 JSON 格式
    augmented_file_path = os.path.join(folder_path, filename)
    df.to_json(augmented_file_path, orient='records', force_ascii=False)
    print(f"增强后的数据保存至 {augmented_file_path}")


def preprocess_data(df):
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    # 检查是否有空的 'content' 列
    df = df[df['content'].notna()]  # 删除 'content' 列中为空的行
    df = df[df['content'].apply(lambda x: isinstance(x, str) and len(x.strip()) > 0)]  # 过滤空字符串或无效数据

    # 标签编码
    label = label_to_id()
    df['label'] = df['label'].apply(lambda x: label[x])

    # 添加情感特征列并编码
    emotion_dict = load_emotion_dict()
    df['emotion_feature'] = df['content'].apply(lambda x: get_emotion_feature(x, emotion_dict))
    df, emotion_encoder = encode_emotion_feature(df)  # 将情感特征编码为数值

    # 分词
    encodings = tokenizer(df['content'].tolist(),
                          padding='max_length',
                          truncation=True,
                          max_length=MAX_LENGTH,
                          return_tensors='pt')

    labels = torch.tensor(df['label'].values)

    # 将情感特征转换为 tensor
    emotion_features = torch.tensor(df['emotion_feature_encoded'].values).unsqueeze(1)  # 扩展维度以适配批量操作

    return encodings, labels, emotion_features


def get_train_valid_dataloaders(file_path, test_size=0.1, batch_size=BATCH_SIZE):
    df = load_data(file_path)

    # 获取 "angry" 类别的样本量，作为参考的目标大小
    target_size = len(df[df['label'] == 'sad'])

    # 只增强 "fear" 类别的数据
    df = augment_fear_class(df, 'fear', augment_word, target_size)

    # 保存增强后的数据集到新的文件夹
    augmented_folder = "weibo_dataset/train"
    save_augmented_data(df, augmented_folder, "augmented_train_data.json")

    # 从保存好的增强数据文件中读取
    augmented_data_path = os.path.join(augmented_folder, "augmented_train_data.json")
    df_augmented = load_data("weibo_dataset/train/usual_train.txt")
    #df_augmented = load_data(augmented_data_path)  # 读取增强后的数据

    # 划分训练集和验证集（10%作为验证集）
    train_df, valid_df = train_test_split(df_augmented, test_size=test_size, random_state=42)

    # 处理训练集
    train_encodings, train_labels, train_emotion_features = preprocess_data(train_df)
    train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels, train_emotion_features)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 处理验证集
    valid_encodings, valid_labels, valid_emotion_features = preprocess_data(valid_df)
    valid_dataset = TensorDataset(valid_encodings['input_ids'], valid_encodings['attention_mask'], valid_labels, valid_emotion_features)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader