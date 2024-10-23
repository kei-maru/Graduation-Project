import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from Config import DEVICE

label_to_id = {
    'angry': 0,
    'happy': 1,
    'neutral': 2,
    'surprise': 3,
    'sad': 4,
    'fear': 5
}

id_to_label = {v: k for k, v in label_to_id.items()}  # 反向映射


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

    print(classification_report(all_labels, all_preds, target_names=list(id_to_label.values())))

    # 计算并绘制混淆矩阵
    cm = confusion_matrix(all_labels, all_preds, labels=list(id_to_label.values()))
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(id_to_label.values()),
                yticklabels=list(id_to_label.values()))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
