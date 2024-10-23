import torch
from torch.optim import AdamW
from Config import EPOCHS, LEARNING_RATE, DEVICE
from Data_loader import get_train_valid_dataloaders
from Evaluate import evaluate_model
from torch.nn import CrossEntropyLoss
from Model import build_model  # 修改为使用扩展后的模型

def train_model(model, train_loader, valid_loader, epochs=EPOCHS, learning_rate=LEARNING_RATE):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    # 根据标签分布定义权重，数据量较少的类别赋予更高的权重
    class_weights = torch.tensor([0.8, 1.2, 1.0, 1.0, 1.5, 1.6]).to(DEVICE)
    # 使用加权的损失函数
    loss_fn = CrossEntropyLoss(weight=class_weights)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            input_ids, attention_mask, labels, emotion_features = [x.to(DEVICE) for x in batch]  # 包括emotion_features

            optimizer.zero_grad()
            # 调用模型的 forward 函数，传入情感特征
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, emotion_features=emotion_features)
            loss = loss_fn(outputs, labels)  # 手动计算损失，之前的 outputs.loss 不再适用
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}')

        # 验证模型
        evaluate_model(model, valid_loader)

if __name__ == "__main__":
    train_loader, valid_loader = get_train_valid_dataloaders('weibo_dataset/train/usual_train.txt')

    # 构建扩展后的模型，使用build_model
    model = build_model(num_labels=6)  # 修改为使用扩展后的模型
    model.to(DEVICE)

    # 训练模型
    train_model(model, train_loader, valid_loader)
