import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# 加载数据
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return pd.DataFrame(data)


# 可视化函数：类别分布
def visualize_label_distribution(df):
    plt.figure(figsize=(8, 6))
    sns.countplot(x='label', data=df)
    plt.title("Label Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.show()


# 可视化函数：文本长度分布
def visualize_text_length_distribution(df):
    plt.figure(figsize=(8, 6))
    text_lengths = df['content'].apply(len)
    sns.histplot(text_lengths, kde=True)
    plt.title("Text Length Distribution")
    plt.xlabel("Text Length")
    plt.ylabel("Frequency")
    plt.show()


# 主函数：加载数据并调用可视化函数
if __name__ == "__main__":
    #file_path = 'weibo_dataset/train/usual_train.txt'
    file_path = 'weibo_dataset/train/augmented_train_data.json'

    # 加载数据
    df = load_data(file_path)

    # 可视化类别分布
    visualize_label_distribution(df)

    # 可视化文本长度分布
    visualize_text_length_distribution(df)
