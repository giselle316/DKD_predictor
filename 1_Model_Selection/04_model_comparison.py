import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_curve, auc, confusion_matrix, classification_report,
                             precision_recall_curve, average_precision_score,
                             accuracy_score, precision_score, recall_score, f1_score, roc_auc_score)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from collections import Counter
import warnings
from tabulate import tabulate
import os
from sklearn.ensemble import ExtraTreesClassifier, BaggingClassifier

warnings.filterwarnings('ignore')

# ===================== 配置参数 =====================
CONFIG = {
    'output_dir': r"results\4_comparison",
    'data_file': "建模及内部验证新-02-05.xlsx",
    'target': "1yearegfr",
    'categorical_features': ['Crescent-shaped_changes', 'Interstitial_fibrosis'],
    'numerical_features': ['ePWV', 'SII', '24h-UP', 'eGFR'],
    'test_size': 0.20,
    'random_state': 43,
    'cv_folds': 5
}

# 模型参数配置
MODEL_PARAMS = {
    "Logistic Regression": {
        "estimator": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        "params": {
            'classifier__C': [0.05, 0.1, 1.0],  # 扩展C范围
            'classifier__solver': ['liblinear', 'saga'],
            'classifier__penalty': ['l1']  # 添加正则化类型
        }
    },
    "Random Forest": {
    "estimator": RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        bootstrap=True,
        oob_score=True
    ),
    "params": {
        # 核心参数优化
        'classifier__n_estimators': [200],  # 增加树的数量
        'classifier__max_depth': [5],    # 更精细的深度控制
        'classifier__min_samples_split': [2],
        'classifier__min_samples_leaf': [2],

        # 特征采样策略优化
        'classifier__max_features': ['sqrt'],

        # 正则化参数
        'classifier__min_impurity_decrease': [0.05],

        # 类别权重策略
        'classifier__class_weight': [None],

        # 分裂标准
        'classifier__criterion': ['entropy'],

        # 增加随机性防止过拟合
        'classifier__max_samples': [0.9]
    }
    },

    # 尝试使用更先进的集成方法

    # # 极端随机森林
    # "Extra Trees": {
    #     "estimator": ExtraTreesClassifier(random_state=42, n_jobs=-1),
    #     "params": {
    #         'classifier__n_estimators': [200],
    #         'classifier__max_depth': [10],
    #         'classifier__min_samples_split': [10],
    #         'classifier__min_samples_leaf': [2],
    #         'classifier__max_features': ['log2']
    #     }
    # },

    "XGBoost": {
        "estimator": XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                                   random_state=42, scale_pos_weight=1),
        "params": {
            'classifier__n_estimators': [30, 40],  # 适当增加
            'classifier__max_depth': [1],  # 更浅的树
            'classifier__learning_rate': [0.01],
            'classifier__subsample': [0.1],  # 子采样
            'classifier__colsample_bytree': [0.1],
            'classifier__reg_alpha': [0],  # L1正则化
            'classifier__reg_lambda': [0.1]  # L2正则化
        }
    },
    "LightGBM": {
        "estimator": LGBMClassifier(random_state=42, class_weight='balanced'),
        "params": {
            'classifier__n_estimators': [10],
            'classifier__max_depth': [3],
            'classifier__learning_rate': [0.05],
            'classifier__num_leaves': [3],     # 与max_depth协调
            'classifier__min_child_samples': [10],  # 增加最小值
            'classifier__subsample': [0.05],       # 添加子采样
            'classifier__colsample_bytree': [0.7]
        }
    },
    "Naive Bayes": {
        "estimator": GaussianNB(),
        "params": {
            'classifier__var_smoothing': [0.5]
        }
    },
    "MLP": {
        "estimator": MLPClassifier(max_iter=1000, random_state=42, early_stopping=True),
        "params": {
            'classifier__hidden_layer_sizes': [(5,), (10,), (5, 3)],  # 更简单的结构
            'classifier__activation': ['relu', 'tanh'],
            'classifier__alpha': [0.01, 0.05, 0.1],  # 更强的正则化
            'classifier__learning_rate_init': [0.001, 0.005],
            'classifier__solver': ['adam'],
            'classifier__batch_size': [16, 32],  # 固定批量大小
            'classifier__early_stopping': [True]  # 强制早停
        }
    }
}

# ===================== 初始化设置 =====================
os.makedirs(CONFIG['output_dir'], exist_ok=True)
CV = StratifiedKFold(n_splits=CONFIG['cv_folds'], shuffle=True, random_state=CONFIG['random_state'])

# ===================== 数据加载和预处理 =====================
data = pd.read_excel(CONFIG['data_file'])
X = data[CONFIG['categorical_features'] + CONFIG['numerical_features']]
y = data[CONFIG['target']]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=CONFIG['test_size'], stratify=y, random_state=CONFIG['random_state']
)

print("\n类别分布情况:")
class_dist = y.value_counts()
print(class_dist)
print(f"正负类比例: {class_dist[1]}:{class_dist[0]} (约1:{round(class_dist[0] / class_dist[1], 2)})")
print("\n训练集类别分布:", Counter(y_train))
print("测试集类别分布:", Counter(y_test))

preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([('scaler', StandardScaler())]), CONFIG['numerical_features']),
    ('cat', Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))]), CONFIG['categorical_features'])
])


# ===================== 模型训练和评估 =====================
def calculate_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "ROC_AUC": roc_auc_score(y_true, y_prob),
        "PR_AUC": average_precision_score(y_true, y_prob),
        "CM": str(confusion_matrix(y_true, y_pred).tolist())  # 转换为字符串以便保存
    }


def find_best_threshold(y_true, y_prob):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    return thresholds[np.argmax(f1_scores)]


results = []
csv_results = []  # 专门用于CSV保存的结果

for name, model_info in MODEL_PARAMS.items():
    print(f"\n=== 训练 {name} ===")

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model_info["estimator"])
    ])

    # 网格搜索调参
    if model_info["params"]:
        grid_search = GridSearchCV(pipeline, param_grid=model_info["params"],
                                   cv=CV, scoring='roc_auc', n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print(f"最佳参数: {grid_search.best_params_}")
    else:
        best_model = pipeline.fit(X_train, y_train)

    # 预测和评估
    y_train_pred_prob = best_model.predict_proba(X_train)[:, 1]
    y_test_pred_prob = best_model.predict_proba(X_test)[:, 1]

    best_threshold = find_best_threshold(y_test, y_test_pred_prob)
    y_train_pred = (y_train_pred_prob >= best_threshold).astype(int)
    y_test_pred = (y_test_pred_prob >= best_threshold).astype(int)

    train_metrics = calculate_metrics(y_train, y_train_pred, y_train_pred_prob)
    test_metrics = calculate_metrics(y_test, y_test_pred, y_test_pred_prob)

    # 保存完整结果（用于可视化）
    result = {
        "Model": name,
        "Best_Threshold": best_threshold,
        "Train_Accuracy": train_metrics["Accuracy"],
        "Train_Precision": train_metrics["Precision"],
        "Train_Recall": train_metrics["Recall"],
        "Train_F1": train_metrics["F1"],
        "Train_ROC_AUC": train_metrics["ROC_AUC"],
        "Train_PR_AUC": train_metrics["PR_AUC"],
        "Train_CM": train_metrics["CM"],
        "Test_Accuracy": test_metrics["Accuracy"],
        "Test_Precision": test_metrics["Precision"],
        "Test_Recall": test_metrics["Recall"],
        "Test_F1": test_metrics["F1"],
        "Test_ROC_AUC": test_metrics["ROC_AUC"],
        "Test_PR_AUC": test_metrics["PR_AUC"],
        "Test_CM": test_metrics["CM"],
        "fpr_train": roc_curve(y_train, y_train_pred_prob)[0],
        "tpr_train": roc_curve(y_train, y_train_pred_prob)[1],
        "precision_train": precision_recall_curve(y_train, y_train_pred_prob)[0],
        "recall_train": precision_recall_curve(y_train, y_train_pred_prob)[1],
        "fpr_test": roc_curve(y_test, y_test_pred_prob)[0],
        "tpr_test": roc_curve(y_test, y_test_pred_prob)[1],
        "precision_test": precision_recall_curve(y_test, y_test_pred_prob)[0],
        "recall_test": precision_recall_curve(y_test, y_test_pred_prob)[1],
        "best_model": best_model
    }
    results.append(result)

    # 保存CSV结果（只包含可序列化的数据）
    csv_result = {
        "Model": name,
        "Best_Threshold": best_threshold,
        "Train_Accuracy": train_metrics["Accuracy"],
        "Train_Precision": train_metrics["Precision"],
        "Train_Recall": train_metrics["Recall"],
        "Train_F1": train_metrics["F1"],
        "Train_ROC_AUC": train_metrics["ROC_AUC"],
        "Train_PR_AUC": train_metrics["PR_AUC"],
        "Train_CM": train_metrics["CM"],
        "Test_Accuracy": test_metrics["Accuracy"],
        "Test_Precision": test_metrics["Precision"],
        "Test_Recall": test_metrics["Recall"],
        "Test_F1": test_metrics["F1"],
        "Test_ROC_AUC": test_metrics["ROC_AUC"],
        "Test_PR_AUC": test_metrics["PR_AUC"],
        "Test_CM": test_metrics["CM"]
    }
    csv_results.append(csv_result)

    # 输出结果
    print("\n训练集分类报告:\n", classification_report(y_train, y_train_pred))
    print("测试集分类报告:\n", classification_report(y_test, y_test_pred))

# ===================== 结果保存和可视化 =====================
# 保存结果到CSV
results_df = pd.DataFrame(csv_results)
results_file = os.path.join(CONFIG['output_dir'], "model_performance.csv")
results_df.to_csv(results_file, index=False)
print(f"\n模型性能结果已保存到: {results_file}")


# 可视化函数
def plot_curves(curve_type, title, filename, dataset_type="test"):
    plt.figure(figsize=(10, 8))
    for result in results:
        x = result[f"{'fpr' if curve_type == 'roc' else 'recall'}_{dataset_type}"]
        y = result[f"{'tpr' if curve_type == 'roc' else 'precision'}_{dataset_type}"]
        metric_key = f"{'Train' if dataset_type == 'train' else 'Test'}_{'ROC_AUC' if curve_type == 'roc' else 'PR_AUC'}"
        metric = result[metric_key]
        plt.plot(x, y, label=f'{result["Model"]} ({metric:.2f})')

    if curve_type == "roc":
        plt.plot([0, 1], [0, 1], 'k--')

    plt.xlabel('False Positive Rate' if curve_type == 'roc' else 'Recall')
    plt.ylabel('True Positive Rate' if curve_type == 'roc' else 'Precision')
    plt.title(f"{title} ({'Train' if dataset_type == 'train' else 'Test'} Set)")
    plt.legend(loc="lower right" if curve_type == "roc" else "lower left")
    plt.savefig(os.path.join(CONFIG['output_dir'], filename), dpi=300, bbox_inches='tight')
    plt.close()


# 绘制曲线
plot_curves("roc", "Receiver Operating Characteristic", "test_roc_curves.png")
plot_curves("pr", "Precision-Recall Curve", "test_pr_curves.png")
plot_curves("roc", "Receiver Operating Characteristic", "train_roc_curves.png", "train")
plot_curves("pr", "Precision-Recall Curve", "train_pr_curves.png", "train")

# 阈值分析图
plt.figure(figsize=(10, 8))
for result in results:
    precisions, recalls, thresholds = precision_recall_curve(y_test, result["best_model"].predict_proba(X_test)[:, 1])
    plt.plot(thresholds, precisions[:-1], label=f'{result["Model"]} Precision', linestyle='--')
    plt.plot(thresholds, recalls[:-1], label=f'{result["Model"]} Recall', linestyle='-')

plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision and Recall vs Threshold")
plt.legend(loc="lower left")
plt.grid(True)
plt.savefig(os.path.join(CONFIG['output_dir'], "threshold_analysis.png"), dpi=300, bbox_inches='tight')
plt.close()

# 输出汇总表格
table_data = []
for res in results:
    table_data.append([
        res["Model"],
        f"{res['Train_Accuracy']:.4f}", f"{res['Test_Accuracy']:.4f}",
        f"{res['Train_Precision']:.4f}", f"{res['Test_Precision']:.4f}",
        f"{res['Train_Recall']:.4f}", f"{res['Test_Recall']:.4f}",
        f"{res['Train_F1']:.4f}", f"{res['Test_F1']:.4f}",
        f"{res['Train_ROC_AUC']:.4f}", f"{res['Test_ROC_AUC']:.4f}",
        f"{res['Train_PR_AUC']:.4f}", f"{res['Test_PR_AUC']:.4f}",
        f"{res['Best_Threshold']:.4f}"
    ])

headers = ["Model", "Train Acc", "Test Acc", "Train Prec", "Test Prec",
           "Train Rec", "Test Rec", "Train F1", "Test F1",
           "Train ROC AUC", "Test ROC AUC", "Train PR AUC", "Test PR AUC", "Best Threshold"]

print("\n模型性能汇总:")
print(tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".4f"))
print(f"\n分析完成! 所有结果已保存到: {CONFIG['output_dir']}")

# %%
# =====================可视化=====================

print("\n=== Generating Improved Model Comparison Visualizations ===")

visualization_dir = os.path.join(CONFIG['output_dir'], "visualizations")
os.makedirs(visualization_dir, exist_ok=True)

model_names = [result["Model"] for result in results]
test_metrics = {
    "Accuracy": [result["Test_Accuracy"] for result in results],
    "Precision": [result["Test_Precision"] for result in results],
    "Recall": [result["Test_Recall"] for result in results],
    "F1": [result["Test_F1"] for result in results],
    "ROC_AUC": [result["Test_ROC_AUC"] for result in results],
    "PR_AUC": [result["Test_PR_AUC"] for result in results]
}

plt.figure(figsize=(14, 8))
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC', 'PR_AUC']
metric_data = [test_metrics[metric] for metric in metrics_to_plot]

box = plt.boxplot(metric_data, labels=metrics_to_plot, patch_artist=True)

colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99FF', '#FFD700']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

means = [np.mean(metric) for metric in metric_data]
for i, mean in enumerate(means):
    plt.plot(i + 1, mean, 'o', color='red', markersize=8, markeredgecolor='black')

plt.title('Distribution of Performance Metrics Across Models\n(Red dots indicate mean values)', fontsize=14, pad=20)
plt.ylabel('Score', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(visualization_dir, "metrics_distribution_boxplot.png"),
            dpi=300, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for i, result in enumerate(results):

    cm = eval(result["Test_CM"])
    cm_array = np.array(cm)

    im = axes[i].imshow(cm_array, interpolation='nearest', cmap='Blues', vmin=0, vmax=np.max(cm_array) * 1.1)
    axes[i].set_title(f'{result["Model"]}\nAccuracy: {result["Test_Accuracy"]:.3f}', fontsize=11)

    thresh = cm_array.max() / 2.
    for j in range(cm_array.shape[0]):
        for k in range(cm_array.shape[1]):
            axes[i].text(k, j, format(cm_array[j, k], 'd'),
                         ha="center", va="center", fontweight='bold',
                         color="white" if cm_array[j, k] > thresh else "black")

    axes[i].set_xticks([0, 1])
    axes[i].set_yticks([0, 1])
    axes[i].set_xticklabels(['Predicted Negative', 'Predicted Positive'])
    axes[i].set_yticklabels(['Actual Negative', 'Actual Positive'])
    axes[i].set_ylabel('True Label', fontsize=10)
    axes[i].set_xlabel('Predicted Label', fontsize=10)

plt.suptitle('Confusion Matrix Heatmaps by Model', fontsize=16, y=0.98)
plt.tight_layout()
plt.savefig(os.path.join(visualization_dir, "confusion_matrices.png"),
            dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(12, 10))
categories = metrics_to_plot
N = len(categories)

angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

ax = plt.subplot(111, polar=True)
plt.xticks(angles[:-1], categories, color='grey', size=10, ha='center')
ax.set_rlabel_position(30)
plt.yticks([0.7, 0.8, 0.9, 1.0], ["0.70", "0.80", "0.90", "1.00"], color="grey", size=8)
plt.ylim(0.65, 1.02)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
line_styles = ['-', '--', '-.', ':', '-', '--']

for i, model_name in enumerate(model_names):
    values = [test_metrics[metric][i] for metric in metrics_to_plot]
    values += values[:1]

    line = ax.plot(angles, values, linewidth=2.5, linestyle=line_styles[i],
                   label=model_name, color=colors[i], marker='o', markersize=6)

    ax.fill(angles, values, alpha=0.1, color=colors[i])

    for j, (angle, value) in enumerate(zip(angles[:-1], values[:-1])):
        if j % 2 == 0:
            ax.text(angle, value + 0.01, f'{value:.3f}',
                    ha='center', va='bottom', fontsize=8, color=colors[i],
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.7))

plt.title('Model Performance Radar Chart\n(Adjusted scale to show subtle differences)', size=14, y=1.08)
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(visualization_dir, "performance_radar_chart_improved.png"),
            dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(14, 8))

min_vals = [min(test_metrics[metric]) for metric in metrics_to_plot]
max_vals = [max(test_metrics[metric]) for metric in metrics_to_plot]
scaled_data = {}

for i, model_name in enumerate(model_names):
    scaled_values = []
    for j, metric in enumerate(metrics_to_plot):
        scaled_val = (test_metrics[metric][i] - min_vals[j]) / (max_vals[j] - min_vals[j] + 1e-8)
        scaled_values.append(scaled_val)
    scaled_data[model_name] = scaled_values

x_pos = range(len(metrics_to_plot))
for i, model_name in enumerate(model_names):
    plt.plot(x_pos, scaled_data[model_name], marker='o', linewidth=2.5,
             label=model_name, color=colors[i], linestyle=line_styles[i], markersize=8)

    for j, val in enumerate(scaled_data[model_name]):
        plt.text(j, val + 0.02, f'{test_metrics[metrics_to_plot[j]][i]:.3f}',
                 ha='center', va='bottom', fontsize=8, color=colors[i],
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

plt.xticks(x_pos, metrics_to_plot, rotation=45, ha='right')
plt.ylabel('Normalized Score (0-1 range)', fontsize=12)
plt.title('Model Performance Parallel Coordinates\n(Showing relative performance differences)', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(visualization_dir, "performance_parallel_coordinates.png"),
            dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(12, 8))

best_performance = {metric: max(test_metrics[metric]) for metric in metrics_to_plot}
gap_data = []

for metric in metrics_to_plot:
    gaps = [best_performance[metric] - value for value in test_metrics[metric]]
    gap_data.append(gaps)

gap_matrix = np.array(gap_data).T

im = plt.imshow(gap_matrix, cmap='Reds_r', aspect='auto', vmin=0, vmax=0.1)

for i in range(len(model_names)):
    for j in range(len(metrics_to_plot)):
        gap_val = gap_matrix[i, j]
        if gap_val > 0:
            text = plt.text(j, i, f'{gap_val:.4f}',
                            ha="center", va="center", color="w" if gap_val > 0.05 else "black",
                            fontweight='bold', fontsize=8)

plt.colorbar(im, label='Gap from Best Performance')
plt.xticks(range(len(metrics_to_plot)), metrics_to_plot, rotation=45, ha='right')
plt.yticks(range(len(model_names)), model_names)
plt.title('Performance Gap Heatmap\n(Smaller values are better)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(visualization_dir, "performance_gap_heatmap.png"),
            dpi=300, bbox_inches='tight')
plt.close()

print(f"All improved visualizations saved to: {visualization_dir}")
print("=== Model comparison visualization completed ===")

print("\n=== Performance Metrics Summary ===")
summary_df = pd.DataFrame({
    'Metric': metrics_to_plot,
    'Min': [min(test_metrics[metric]) for metric in metrics_to_plot],
    'Max': [max(test_metrics[metric]) for metric in metrics_to_plot],
    'Mean': [np.mean(test_metrics[metric]) for metric in metrics_to_plot],
    'Std': [np.std(test_metrics[metric]) for metric in metrics_to_plot],
    'Range': [max(test_metrics[metric]) - min(test_metrics[metric]) for metric in metrics_to_plot]
})

print(tabulate(summary_df, headers='keys', tablefmt='grid', floatfmt=".4f"))
# %%
# ===================== 单独绘制随机森林模型的ROC-AUC曲线 =====================
# 找到随机森林模型的结果
rf_result = None
for result in results:
    if result["Model"] == "Random Forest":
        rf_result = result
        break

if rf_result is not None:
    # 设置图像大小，增加宽度确保左侧显示完整
    plt.figure(figsize=(8, 6))

    # 计算测试集的ROC曲线数据
    fpr_test = rf_result["fpr_test"]
    tpr_test = rf_result["tpr_test"]
    roc_auc_test = rf_result["Test_ROC_AUC"]

    # 计算训练集的ROC曲线数据
    fpr_train = rf_result["fpr_train"]
    tpr_train = rf_result["tpr_train"]
    roc_auc_train = rf_result["Train_ROC_AUC"]

    # 绘制测试集ROC曲线
    plt.plot(fpr_test, tpr_test, label=f'Test ROC (AUC = {roc_auc_test:.2f})', lw=2)

    # 绘制训练集ROC曲线
    plt.plot(fpr_train, tpr_train, label=f'Train ROC (AUC = {roc_auc_train:.2f})', lw=2)

    # 绘制随机猜测线
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.5)')

    # 设置坐标轴范围，确保从0开始
    plt.xlim([-0.02, 1.0])  # 左侧稍微扩展一点
    plt.ylim([0.0, 1.05])

    # 添加标签和标题
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Random Forest ROC Curve (Train vs Test)', fontsize=14)  # 去掉fontweight='bold'
    plt.legend(loc='lower right', fontsize=11)

    # 添加网格
    plt.grid(True, alpha=0.3)

    # 调整布局，确保左侧不被裁剪
    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.9)

    # 保存图像，确保包含完整内容
    plt.savefig(os.path.join(CONFIG['output_dir'], "random_forest_roc_curve.png"),
                dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()
    plt.close()

    print("随机森林ROC-AUC曲线已单独保存!")
else:
    print("未找到随机森林模型结果")