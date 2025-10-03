import numpy as np
import joblib

# 加载模型
model = joblib.load('models/random_forest_model.joblib')
label_encoders = joblib.load('models/label_encoders.joblib')
model_metadata = joblib.load('models/model_metadata.joblib')

FEATURE_COLUMNS = model_metadata['features']
THRESHOLD = model_metadata.get('optimal_threshold', 0.5)

def predict_clinical_risk(user_inputs):
    try:
        # 确保输入顺序和模型训练时一致
        x_input = [float(user_inputs[feature]) for feature in FEATURE_COLUMNS]
        x_input = np.array(x_input).reshape(1, -1)

        probas = model.predict_proba(x_input)[0]
        yes_prob = probas[1]
        prediction = 1 if yes_prob >= THRESHOLD else 0
        return prediction, round(yes_prob, 3)
    except Exception as e:
        return f"错误 Error: {str(e)}", 0.0
