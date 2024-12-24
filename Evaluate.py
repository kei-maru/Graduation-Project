import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from Config import DEVICE
from Model import build_model
import os
import pandas as pd

label_to_id = {
    'angry': 0,
    'happy': 1,
    'neutral': 2,
    'surprise': 3,
    'sad': 4,
    'fear': 5
}

id_to_label = {v: k for k, v in label_to_id.items()}  # 反向映射

def load_best_model():
    # 读取记录最佳模型路径的文件
    best_model_file = os.path.join("saved_models", "best_model.txt")

    if not os.path.exists(best_model_file):
        raise FileNotFoundError("No best model file found. Please train the model first.")

    # 读取模型路径
    with open(best_model_file, "r") as f:
        model_path = f.readline().strip()

    # 加载模型
    model = build_model(num_labels=6)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()  # 切换到评估模式
    print(f"Loaded best model from {model_path}")
    return model

def plot_confusion_matrix(all_labels, all_preds, labels, title="Confusion Matrix"):

    cm = confusion_matrix(all_labels, all_preds, labels=labels)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.show()

def evaluate_model(model, loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            # 解包包含情感特征的所有四个值
            input_ids, attention_mask, labels, emotion_features = [x.to(DEVICE) for x in batch]

            # 前向传播时传入情感特征
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, emotion_features=emotion_features)
            logits = outputs  # 获取模型的logits
            predictions = torch.argmax(logits, dim=1)

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 使用 id_to_label，将数值标签转回文字标签
    all_preds = [id_to_label[pred] for pred in all_preds]
    all_labels = [id_to_label[label] for label in all_labels]

    # 计算准确率
    correct_predictions = sum([1 for true, pred in zip(all_labels, all_preds) if true == pred])
    accuracy = correct_predictions / len(all_labels)

    print(classification_report(all_labels, all_preds, target_names=list(id_to_label.values())))

    # 调用单独的混淆矩阵绘制函数
    #plot_confusion_matrix(all_labels, all_preds, labels=list(id_to_label.values()))

    return accuracy
