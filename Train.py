import torch
from torch.optim import AdamW
from Config import EPOCHS, LEARNING_RATE, DEVICE, BATCH_SIZE, TRAIN_FILE, TEST_FILE
from Data_loader import get_train_valid_dataloaders
from Evaluate import evaluate_model, load_best_model
from torch.nn import CrossEntropyLoss
from Model import build_model
import os
import argparse

# 保存模型的文件夹路径
MODEL_SAVE_PATH = "saved_models"


# 检查路径是否存在，如果不存在则创建
if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)


def train_model(model, train_loader, valid_loader, output_dir, epochs=EPOCHS, learning_rate=LEARNING_RATE):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    # 根据标签分布定义权重，数据量较少的类别赋予更高的权重
    class_weights = torch.tensor([0.8, 1.2, 1.0, 1.0, 1.8, 1.6]).to(DEVICE)
    # 使用加权的损失函数
    loss_fn = CrossEntropyLoss(weight=class_weights)

    best_accuracy = 0.0  # 用于存储最佳模型的准确率

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            input_ids, attention_mask, labels, emotion_features = [x.to(DEVICE) for x in batch]

            optimizer.zero_grad()
            # 调用模型的 forward 函数，传入情感特征
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, emotion_features=emotion_features)
            loss = loss_fn(outputs, labels)  # 手动计算损失
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}')

        # 验证模型并获取准确率
        accuracy = evaluate_model(model, valid_loader)

        # 如果当前模型表现更好，则保存模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_path = os.path.join(output_dir, f"best_model_epoch_{epoch + 1}.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved at epoch {epoch + 1} with accuracy {accuracy:.4f}")

            # 将最佳模型路径写入文件
            with open(os.path.join(output_dir, "best_model.txt"), "w") as f:
                f.write(best_model_path)


if __name__ == "__main__":
    # 从config.py中读取配置
    data_dir = TRAIN_FILE  # 从config.py中获取训练数据文件路径
    output_dir = "saved_models"  # 设定模型保存目录

    # 获取训练和验证数据加载器
    train_loader, valid_loader = get_train_valid_dataloaders(data_dir, batch_size=BATCH_SIZE)

    # 构建模型
    model = build_model(num_labels=6)
    model.to(DEVICE)

    # 训练模型并保存
    train_model(model, train_loader, valid_loader, output_dir, epochs=EPOCHS, learning_rate=LEARNING_RATE)

    # 加载并评估保存的最佳模型
    model = load_best_model()  # 加载最佳模型
    evaluate_model(model, valid_loader)