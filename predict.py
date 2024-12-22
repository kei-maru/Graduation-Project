import sinaweibopy3
import urllib.error
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from urllib.parse import urlparse, parse_qs
from Data_loader import get_emotion_feature, load_emotion_dict, clean_data
import torch
from transformers import BertTokenizer
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
import jieba
import json
import pandas as pd

# 配置微博 API 的 APP_KEY, APP_SECRET 和 REDIRECT_URL
APP_KEY = '1484092473'
APP_SECRET = 'bc6bdeeaa3a230798666028b2364ef76'
REDIRECT_URL = 'https://api.weibo.com/oauth2/default.html'

# 模型路径
MODEL_PATH = "saved_models/best_model_epoch_3.pt"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setup_selenium_to_get_code(url):
    """通过 Selenium 自动完成授权并获取重定向的 URL"""
    driver = webdriver.Chrome(ChromeDriverManager().install())
    driver.get(url)
    input("请在浏览器中完成授权后按回车键...")  # 等待用户完成授权
    redirect_url = driver.current_url
    driver.quit()
    return redirect_url


def extract_weibo_id(url):
    """从 URL 提取微博 ID"""
    parsed_url = urlparse(url)
    path_parts = parsed_url.path.strip('/').split('/')
    if len(path_parts) > 1:
        return path_parts[-1]  # 提取 ID 部分
    raise ValueError("URL 格式不正确，无法提取微博 ID")
def fetch_long_id(client, short_id):
    """通过微博 API 将短 ID 转换为长整型 ID"""
    try:
        response = client.get.statuses__queryid(mid=short_id, type=1, isBase62=1)
        return response['id']
    except urllib.error.HTTPError as e:
        print("获取长整型 ID 时出错:", e.code)
        print("错误信息:", e.read().decode())
        return None


def fetch_comments(client, weibo_id, max_comments=900):
    """获取微博评论，处理分页"""
    comments = []
    page = 1  # 从第一页开始
    while len(comments) < max_comments:
        try:
            # 请求当前页的评论
            response = client.get.comments__show(id=weibo_id, page=page, count=100)
            comments.extend([comment['text'] for comment in response['comments']])

            # 如果返回的评论数少于 100，说明已经获取到所有评论
            if len(response['comments']) < 100:
                break

            page += 1  # 继续请求下一页
        except urllib.error.HTTPError as e:
            print("HTTP 错误:", e.code)
            print("错误信息:", e.read().decode())
            break

    # 限制返回评论的最大数量
    return comments[:max_comments]


def load_model(model_path):
    """加载预训练模型"""
    from Model import build_model  # 假设您的 `build_model` 定义在 Model.py 中
    model = build_model(num_labels=6)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def preprocess_text(text, tokenizer, max_length=128):
    """预处理文本：清洗文本并进行分词"""
    # 将文本转换为 DataFrame 格式以便清洗
    df = pd.DataFrame({'content': [text]})

    # 调用 clean_data 清洗文本
    cleaned_df = clean_data(df)

    # 获取清洗后的文本
    cleaned_text = cleaned_df['content'].values[0]
    print(cleaned_text)
    # 使用分词器进行分词处理
    inputs = tokenizer(
        cleaned_text,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    return inputs['input_ids'].to(DEVICE), inputs['attention_mask'].to(DEVICE)

def encode_emotion_feature(emotion, emotion_dict):
    """根据情感标签返回编码"""
    label_to_id = {emotion: idx for idx, emotion in enumerate(emotion_dict.keys())}
    return label_to_id.get(emotion, label_to_id['neutral'])  # 默认编码为 'neutral'



def predict_sentiment(text, model, tokenizer, emotion_dict):
    """预测文本情感"""
    # 预处理文本，获取 input_ids 和 attention_mask
    input_ids, attention_mask = preprocess_text(text, tokenizer)

    # 计算 emotion_features
    emotion = get_emotion_feature(text, emotion_dict)  # 从文本获取情感特征
    emotion_encoded = encode_emotion_feature(emotion, emotion_dict)  # 将情感特征编码
    emotion_features = torch.tensor([[emotion_encoded]], dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        # 将 emotion_features 一起传递给模型
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, emotion_features=emotion_features)
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()

    return predictions[0]



def analyze_sentiments(comments, model, tokenizer, emotion_dict):
    """分析评论的情感"""
    return [predict_sentiment(comment, model, tokenizer, emotion_dict) for comment in comments]


def visualize_sentiments(sentiments):
    """可视化情感分布（饼图）"""
    sentiment_labels = ['angry', 'happy', 'neutral', 'surprise', 'sad', 'fear']

    # 统计每种情感的数量
    counts = Counter(sentiments)
    labels = [sentiment_labels[i] for i in range(6)]
    values = [counts.get(i, 0) for i in range(6)]

    # 绘制饼图
    plt.figure(figsize=(8, 8))
    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90,
            colors=['#FF9999', '#66B3FF', '#99FF99', '#FFCC99', '#FFD700', '#C71585'])
    plt.title("Sentiment Distribution")
    plt.axis('equal')  # 保证饼图是圆形的
    plt.show()


def generate_wordcloud(comments):
    """生成词云"""
    text = ' '.join(jieba.cut(' '.join(comments)))
    wordcloud = WordCloud(font_path='simhei.ttf', background_color='white', width=800, height=400).generate(text)

    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Word Cloud of Comments")
    plt.show()


def main():
    # 初始化微博 API 客户端
    client = sinaweibopy3.APIClient(app_key=APP_KEY, app_secret=APP_SECRET, redirect_uri=REDIRECT_URL)
    url = client.get_authorize_url()

    # 使用 Selenium 获取授权后的 URL
    redirect_url = setup_selenium_to_get_code(url)
    code = parse_qs(urlparse(redirect_url).query).get('code', [None])[0]

    # 换取 Access Token
    result = client.request_access_token(code)
    client.set_access_token(result.access_token, result.expires_in)

    # 输入微博帖子 URL，提取微博 ID
    post_url = input("请输入微博帖子的网址：")
    short_id = extract_weibo_id(post_url)  # 从 URL 提取短 ID
    long_id = fetch_long_id(client, short_id)
    # 获取评论
    if long_id:
        comments = fetch_comments(client, long_id)

    if not comments:
        print("未获取到评论内容")
        return

    print(comments)
    # 加载模型和分词器
    model = load_model(MODEL_PATH)
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    # 分析评论情感
    emotion_dict = load_emotion_dict() #获取情感字典
    sentiments = analyze_sentiments(comments, model, tokenizer, emotion_dict)

    # 可视化情感分布和词云
    visualize_sentiments(sentiments)
    generate_wordcloud(comments)


if __name__ == "__main__":
    main()
