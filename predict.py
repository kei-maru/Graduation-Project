import torch
import json
from transformers import BertTokenizer
from Config import DEVICE
from Model import build_model
from Data_loader import load_data, preprocess_data
import os

# 设置模型路径
MODEL_PATH = "saved_models/best_model_epoch_3.pt"


# 加载模型
def load_model(model_path):
    # 构建模型
    model = build_model(num_labels=6)  # 假设你的模型是一个多分类模型，有6个类别
    model.load_state_dict(torch.load(model_path))  # 加载模型参数
    model.to(DEVICE)
    model.eval()  # 设置为评估模式
    return model


# 预处理文本
def preprocess_text(text, tokenizer, max_length=128):
    inputs = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    return inputs['input_ids'].to(DEVICE), inputs['attention_mask'].to(DEVICE)


# 获取情感特征
def get_emotion_feature(text, emotion_dict):
    for emotion, words in emotion_dict.items():
        if any(word in text for word in words):
            return emotion
    return 'neutral'  # 如果没有匹配，中立


# 预处理情感特征并转换为张量
def preprocess_emotion_feature(text, emotion_dict):
    emotion = get_emotion_feature(text, emotion_dict)
    label_to_id = {'angry': 0, 'happy': 1, 'neutral': 2, 'surprise': 3, 'sad': 4, 'fear': 5}
    emotion_id = label_to_id.get(emotion, 2)  # 默认为 'neutral'，即 2
    return torch.tensor([emotion_id]).to(DEVICE)


# 进行情感分析
def predict_sentiment(text, model, tokenizer, emotion_dict):
    input_ids, attention_mask = preprocess_text(text, tokenizer)
    emotion_features = preprocess_emotion_feature(text, emotion_dict)

    # 确保 emotion_features 的形状是 (batch_size, 1)，用于拼接
    emotion_features = emotion_features.unsqueeze(1)  # 扩展维度

    with torch.no_grad():
        # 预测情感
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, emotion_features=emotion_features)

        # 直接获取 logits
        logits = outputs[0]  # 修改此行，直接获取输出元组的第一个元素

        predictions = torch.argmax(logits, dim=-1)
        return predictions.item()  # 返回预测的标签


# 加载测试数据
def load_test_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)  # 读取 JSON 文件
    return data


# 映射标签到情感
def label_to_emotion(label_id):
    emotion_map = {
        0: 'angry',
        1: 'happy',
        2: 'neutral',
        3: 'surprise',
        4: 'sad',
        5: 'fear'
    }
    return emotion_map.get(label_id, 'unknown')


# 主函数
if __name__ == "__main__":
    # 加载模型
    model = load_model(MODEL_PATH)

    # 加载BERT tokenizer（假设你使用了BERT作为模型）
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    # 加载情感词典
    emotion_dict = {
        'angry': ['愤怒', '生气', '气愤', '恼火', '火大', '愠怒'],
        'happy': ['开心', '快乐', '高兴', '愉快', '喜悦'],
        'neutral': ['普通', '正常', '一般', '无感'],
        'surprise': ['惊讶', '震惊', '意外', '惊喜'],
        'sad': ['悲伤', '难过', '沮丧', '伤心', '失落', '痛苦'],
        'fear': ['害怕', '恐惧', '担忧', '紧张', '恐慌']
    }

    # 读取 JSON 文件中的爬虫结果
    file_path = "Weibo_data.json"
    crawled_data = load_test_data(file_path)

    # 遍历每个爬取的文本，进行情感分析
    for item in crawled_data:
        sample_text = item.get('content', '')  # 获取每条内容文本
        if sample_text:  # 如果文本不为空
            sentiment_label_id = predict_sentiment(sample_text, model, tokenizer, emotion_dict)
            sentiment = label_to_emotion(sentiment_label_id)
            print(f"Text: {sample_text}\nPredicted sentiment: {sentiment}\n")
        else:
            print("Empty text, skipping...\n")