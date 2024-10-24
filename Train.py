import torch
from torch.optim import AdamW
from Config import EPOCHS, LEARNING_RATE, DEVICE
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
    # 使用 argparse 添加命令行参数解析
    parser = argparse.ArgumentParser(description="模型训练脚本")
    parser.add_argument('--data_dir', type=str, required=True, help='数据集的路径')
    parser.add_argument('--output_dir', type=str, required=True, help='模型保存路径')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='训练的epochs数量')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')

    args = parser.parse_args()

    # 确保输出路径存在
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 获取训练和验证数据加载器
    train_loader, valid_loader = get_train_valid_dataloaders(args.data_dir)

    # 构建扩展后的模型，使用build_model
    model = build_model(num_labels=6)
    model.to(DEVICE)

    # 训练模型并保存
    train_model(model, train_loader, valid_loader, args.output_dir, epochs=args.epochs)

    # 加载并评估保存的最佳模型
    model = load_best_model()  # 加载最佳模型
    evaluate_model(model, valid_loader)
