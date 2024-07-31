from flask import Flask, request, jsonify, render_template
import librosa
import numpy as np
import tensorflow as tf
from keras.models import load_model

# 假设你的模型和特征提取逻辑已经准备好了
# 加载你的模型
model = load_model("test_kick_1.h5")

cat_dict = [
    "Top",
    "Chest",
    "Signature",
    "Stomp",
    "Punchy",
    "808s",
    "Distorted",
    "Psy",
    "Big",
    "Hardstyle",
    "Stadium",
]

# 定义 Flask 应用
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1000 * 1000


# 定义一个函数来处理音频文件并返回预测结果
def process_audio(file):
    # 加载音频文件
    y, sr = librosa.load(file)
    # 提取特征
    n_fft = max(2048*3, min(2048, len(y)))
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft).T
    # 预测
    prediction = model.predict(tf.ragged.constant([mfccs]))
    return prediction


# 定义一个路由来处理音频文件上传请求
@app.route('/predict', methods=['POST'])
def predict():
    # 检查是否有文件上传
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    # 检查文件是否为音频文件
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file):
        # 处理音频文件并返回预测结果
        prediction = process_audio(file)
        item = np.argmax(prediction)
        cat_name = cat_dict[item]
        return jsonify({"prediction": cat_name, "probabilities" : prediction.tolist()})
    else:
        return jsonify({"error": "File type is not allowed or too large"}), 400


# 定义一个辅助函数来检查文件类型
def allowed_file(file):
    filename = file.filename
    ALLOWED_EXTENSIONS = {'wav', 'mp3', 'aif'}
    return '.' in filename and \
           filename.rsplit('.', 1)[-1].lower() in ALLOWED_EXTENSIONS \
           and file.stream.tell() < 10


# 定义一个路由来显示上传表单
@app.route('/')
def upload_form():
    return render_template('upload.html')


# 运行 Flask 应用
if __name__ == '__main__':
    app.run(debug=True)
