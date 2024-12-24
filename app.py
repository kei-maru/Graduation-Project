from flask import Flask, request, jsonify, redirect, render_template, session, url_for
import base64
from io import BytesIO
from selenium.webdriver.chrome import webdriver
import sinaweibopy3
import torch
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
from webdriver_manager.chrome import ChromeDriverManager
import jieba
import pandas as pd
from transformers import BertTokenizer
from urllib.parse import urlparse, parse_qs
from Data_loader import get_emotion_feature, load_emotion_dict, clean_data

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # 用于 session 加密

APP_KEY = '1484092473'
APP_SECRET = 'bc6bdeeaa3a230798666028b2364ef76'
REDIRECT_URL = 'http://127.0.0.1:5000/callback'

MODEL_PATH = "saved_models/best_model_epoch_3.pt"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        # 使用微博的 API 获取长 ID
        print('client, short_id: ',client,short_id)
        response = client.get.statuses__queryid(mid=short_id, type=1, isBase62=1)

        # 打印完整的 API 返回响应，帮助调试
        print(f"API 返回的响应: {response}")

        return response['id']
    except Exception as e:
        print(f"获取长整型 ID 时出错: {e}")
        return None


def fetch_comments(client, weibo_id, max_comments=900):
    """获取微博评论，处理分页"""
    comments = []
    page = 1
    while len(comments) < max_comments:
        try:
            # 请求微博评论，处理分页
            response = client.get.comments__show(id=weibo_id, page=page)
            #print(response)  # 打印每次请求返回的数据，帮助调试

            comments.extend([comment['text'] for comment in response['comments']])

            # 如果返回的评论数少于 100，说明所有评论都已获取
            if len(response['comments']) < 50:
                break

            page += 1  # 请求下一页评论

        except Exception as e:
            print(f"获取评论时出错: {e}")
            break
    print(page)

    # 返回不超过 max_comments 条评论
    return comments[:max_comments]


def load_model(model_path):
    """加载预训练模型"""
    from Model import build_model  # 假设模型定义在 Model.py 中
    model = build_model(num_labels=6)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def preprocess_text(text, tokenizer, max_length=128):
    """清洗文本并进行分词"""
    df = pd.DataFrame({'content': [text]})
    cleaned_df = clean_data(df)
    cleaned_text = cleaned_df['content'].values[0]
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
    input_ids, attention_mask = preprocess_text(text, tokenizer)
    emotion = get_emotion_feature(text, emotion_dict)
    emotion_encoded = encode_emotion_feature(emotion, emotion_dict)
    emotion_features = torch.tensor([[emotion_encoded]], dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, emotion_features=emotion_features)
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()

    return predictions[0]


def analyze_sentiments(comments, model, tokenizer, emotion_dict):
    """分析评论情感"""
    sentiments = [predict_sentiment(comment, model, tokenizer, emotion_dict) for comment in comments]
    # 将 sentiments 中的 numpy.int64 转换为 int
    return [int(s) for s in sentiments]  # 确保返回的是普通的 int 类型

@app.route('/')
def index():
    """根路由，自动重定向到登录路由"""
    access_token = session.get('access_token', None)

    # 打印 access_token，帮助调试
    print(f"Access token in session: {access_token}")

    if not access_token:
        print("Access token not found in session. Redirecting to login...")
        return redirect(url_for('login'))  # 自动重定向到 /login 路由

    return render_template('index.html')  # 返回主页 HTML 文件



@app.route('/login')
def login():
    """生成微博授权链接"""
    client = sinaweibopy3.APIClient(app_key=APP_KEY, app_secret=APP_SECRET, redirect_uri=REDIRECT_URL)
    authorize_url = client.get_authorize_url()
    return redirect(authorize_url)  # 跳转到微博授权页面


@app.route('/callback')
def callback():
    """微博授权回调页面，获取 code 并交换 access_token"""
    code = request.args.get('code')  # 获取微博返回的授权 code
    if not code:
        return "授权失败，未获取到 code"

    try:
        client = sinaweibopy3.APIClient(app_key=APP_KEY, app_secret=APP_SECRET, redirect_uri=REDIRECT_URL)
        result = client.request_access_token(code)  # 使用 code 获取 access_token
        access_token = result.access_token
        expires_in = result.expires_in  # 获取微博返回的 access_token 有效期

        # 打印 access_token 和 expires_in，帮助调试
        print(f"获得的 access_token: {access_token}")
        print(f"access_token 的有效期: {expires_in}秒")

        # 存储 access_token 和 expires_in 到 session 中
        session['access_token'] = access_token  # 使用 session 存储 access_token
        session['expires_in'] = expires_in  # 存储 expires_in

        # 打印 session，确保 access_token 和 expires_in 已存储
        print(f"session['access_token']: {session.get('access_token')}")
        print(f"session['expires_in']: {session.get('expires_in')}")

        # 授权成功后重定向到主页
        return redirect(url_for('index'))  # 授权成功后返回 index 页面

    except Exception as e:
        # 捕获异常并返回错误信息
        print(f"获取 access_token 时出错: {e}")
        return jsonify({"error": str(e)})




# API端点：获取评论和分析情感
@app.route('/fetch_comments', methods=['POST'])
def fetch_comments_endpoint():
    post_url = request.json.get('post_url')  # 获取请求中传递的微博 URL

    # 获取存储在 session 中的 access_token 和 expires_in
    access_token = session.get('access_token')
    expires_in = session.get('expires_in')

    if not access_token:
        return jsonify({"error": "未授权，请先登录"})  # 如果没有 access_token，返回错误
    if expires_in is None:
        return jsonify({"error": "未获取到有效期信息，请重新授权"})  # 如果 expires_in 为 None，返回错误

    # 初始化微博 API 客户端
    client = sinaweibopy3.APIClient(app_key=APP_KEY, app_secret=APP_SECRET, redirect_uri=REDIRECT_URL)
    client.set_access_token(access_token, expires_in)  # 使用从 session 中获取的 expires_in

    # 获取微博 ID 和评论
    short_id = extract_weibo_id(post_url)
    long_id = fetch_long_id(client, short_id)

    if not long_id:
        return jsonify({"error": "无法获取微博的长 ID"})

    comments = fetch_comments(client, long_id)
    print(comments)

    if not comments:
        return jsonify({"error": "未获取到评论内容"})

    # 加载模型和分词器
    model = load_model(MODEL_PATH)
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    # 获取情感字典
    emotion_dict = load_emotion_dict()

    # 分析评论情感
    sentiments = analyze_sentiments(comments, model, tokenizer, emotion_dict)
    print(sentiments)

    # 可视化情感分布（饼图）
    sentiment_labels = ['angry', 'happy', 'neutral', 'surprise', 'sad', 'fear']
    counts = Counter(sentiments)
    values = [counts.get(i, 0) for i in range(6)]

    fig, ax = plt.subplots()
    ax.pie(values, labels=sentiment_labels, autopct='%1.1f%%', startangle=90,
           colors=['#FF9999', '#66B3FF', '#99FF99', '#FFCC99', '#FFD700', '#C71585'])
    ax.axis('equal')

    # 转为图片流
    img_stream = BytesIO()
    plt.savefig(img_stream, format='png')
    img_stream.seek(0)
    plt.close(fig)  # 关闭图形，避免问题

    # 生成词云
    wordcloud = WordCloud(font_path='simhei.ttf', background_color='white', width=800, height=400).generate(
        ' '.join(comments))
    wordcloud_stream = BytesIO()
    wordcloud.to_image().save(wordcloud_stream, format='PNG')
    wordcloud_stream.seek(0)

    # 返回 Base64 编码的图像
    sentiment_img_base64 = base64.b64encode(img_stream.read()).decode('utf-8')
    wordcloud_img_base64 = base64.b64encode(wordcloud_stream.read()).decode('utf-8')

    return jsonify({
        'sentiment_img': sentiment_img_base64,
        'wordcloud_img': wordcloud_img_base64,
        'sentiments': sentiments,
        'comments': comments
    })

if __name__ == '__main__':
    app.run(debug=True)
