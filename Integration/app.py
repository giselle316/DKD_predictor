import os
import sys

# 获取当前文件的绝对路径并设置工作目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 切换到当前文件所在目录
os.chdir(current_dir)

print("当前工作目录:", os.getcwd())
print("app.py 所在目录:", current_dir)

# 检查模型文件是否存在
model_path = 'models/random_forest_model.joblib'
print(f"模型文件是否存在: {os.path.exists(model_path)}")
if os.path.exists('models'):
    print(f"models目录内容: {os.listdir('models')}")
else:
    print("models目录不存在")

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from utils.predict_image import predict_crescent, predict_fibrosis
from utils.predict_clinical import predict_clinical_risk

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    crescent_files = request.files.getlist('crescent_images')

    crescent_preds = []
    for file in crescent_files:
        if file and file.filename != '':
            path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
            file.save(path)
            pred, _ = predict_crescent(path)
            crescent_preds.append(pred)

    crescent_result = 1 if crescent_preds.count(1) > 3 else 0

    fibrosis_files = request.files.getlist('fibrosis_images')
    fibrosis_preds = []
    for file in fibrosis_files:
        if file and file.filename != '':
            path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
            file.save(path)
            pred, conf = predict_fibrosis(path)
            fibrosis_preds.append(pred)

    fibrosis_result = 0
    fibrosis_conf = 0
    if fibrosis_preds:
        fibrosis_result = 1 if fibrosis_preds.count(1) > len(fibrosis_preds) / 2 else 0
        fibrosis_conf = sum(fibrosis_preds) / len(fibrosis_preds)

    # 获取输入
    clinical_inputs = {
        'Crescent-shaped_changes': crescent_result,
        'Interstitial_fibrosis': fibrosis_result,
        'ePWV': float(request.form['ePWV']),
        'SII': float(request.form['SII']),
        '24h-UP': float(request.form['24h-UP']),
        'eGFR': float(request.form['eGFR'])
    }

    clinical_pred, clinical_conf = predict_clinical_risk(clinical_inputs)

    return render_template('result.html',
                           crescent=crescent_result,
                           fibrosis=fibrosis_result, fib_conf=fibrosis_conf,
                           clinical=clinical_pred, clinical_conf=clinical_conf)

if __name__ == '__main__':
    app.run(debug=True)