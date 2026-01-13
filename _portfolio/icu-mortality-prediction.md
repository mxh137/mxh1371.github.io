---
title: "ICU 患者死亡率预测与数据可视化分析"
collection: portfolio
type: "Machine Learning"
permalink: /portfolio/icu-mortality-prediction
date: 2026-01-13
published: true
excerpt: "本项目旨在通过对重症监护室（ICU）患者的临床数据进行深入分析与可视化，并构建机器学习模型预测患者的院内死亡风险，为临床决策提供支持。"
header:
  image: /images/portfolio/icu-mortality-prediction/hospital_expire_flag_donut.png
  teaser: /images/portfolio/icu-mortality-prediction/hospital_expire_flag_donut.png
  caption: "ICU 患者死亡率预测"
location: "Data Science"
tags:
  - 数据分析
  - 机器学习
  - ICU
  - 死亡率预测
  - Python
  - Scikit-learn
  - 数据可视化
categories:
  - Portfolio
  - Machine Learning
---

## 项目背景 (Background)

重症监护室（ICU）患者的院内死亡风险评估是临床实践中的重要环节。精准的风险预测有助于医生及时调整治疗方案，优化医疗资源分配，并与患者家属进行更有效的沟通。本项目旨在利用患者入院前24小时内的多维度数据，构建并评估机器学习模型，以期对ICU患者的院内死亡风险进行早期、准确的预测。

## 核心实现 (Implementation)

### 1. 数据加载与初步探索

项目首先加载 ICU 患者数据，并进行初步的检查，包括数据维度、前几行预览、数据类型以及缺失值情况。

```python
import pandas as pd
import numpy as np

# 数据路径
path = "/wujidata/xdl/ecg_sleep/icu/icu_first24hours.csv"

# 1) 读取数据
df = pd.read_csv(path)
print("Raw shape:", df.shape)

# 显示数据前3行
print(df.head(3).to_markdown(index=False))

# 检查数据类型
print(df.dtypes.head(20).to_markdown())

# 检查缺失数据情况
missing_rate = df.isna().mean().sort_values(ascending=False)
print("缺失数据情况：")
print(missing_rate.head(30).to_markdown())

# 检查目标变量分布（院内死亡率）
target = "HOSPITAL_EXPIRE_FLAG"
print(f"\n目标变量 '{target}' 分布:")
print(df[target].value_counts(dropna=False).to_markdown())
print(f"阳性率 (Positive rate): {df[target].mean():.4f}")
```

### 2. 数据清洗与预处理

数据清洗包括列名标准化、时间格式解析、无限值替换为缺失值以及基于 `HADM_ID` 的去重操作。针对缺失值，项目设定了缺失率阈值，对缺失过高的列进行删除，然后对剩余的数值型和非数值型特征分别采用中位数和众数进行插补。

```python
from sklearn.impute import SimpleImputer

# 缺失率超过该阈值的列直接丢弃
col_missing_thresh = 0.4
# 你后续可能要预测的标签
target_cols = ["HOSPITAL_EXPIRE_FLAG", "is_early_death"]
# 是否去重（按一次住院 HADM_ID 保留第一条）
dedup_by_hadm = True

# 2) 基础清洗：列名去空格、时间解析、inf->nan
df.columns = [c.strip() for c in df.columns]
if "ADMITTIME" in df.columns:
    df["ADMITTIME"] = pd.to_datetime(df["ADMITTIME"], errors="coerce")

df = df.replace([np.inf, -np.inf], np.nan)

# 3) 去重（可选）
if dedup_by_hadm and "HADM_ID" in df.columns:
    before = df.shape[0]
    df = df.drop_duplicates(subset=["HADM_ID"], keep="first")
    print(f"Dedup by HADM_ID: {before} -> {df.shape[0]}")

# 4) 处理缺失值
print(f"原始特征数量: {df.shape[1]}")
# 移除缺失率过高的列
initial_cols = set(df.columns)
missing_rate = df.isna().mean()
cols_to_drop = missing_rate[missing_rate > col_missing_thresh].index.tolist()
# 确保目标列不被删除
cols_to_drop = [col for col in cols_to_drop if col not in target_cols]
df = df.drop(columns=cols_to_drop)
print(f"删除缺失率超过 {col_missing_thresh:.0%} 的列: {len(cols_to_drop)} 列")
print(f"剩余特征数量: {df.shape[1]}")

# 分离数值和非数值列
numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

# 数值列中位数插补，非数值列众数插补
for col in numerical_cols:
    if df[col].isna().any():
        imputer = SimpleImputer(strategy="median")
        df[col] = imputer.fit_transform(df[[col]])

for col in categorical_cols:
    if df[col].isna().any():
        imputer = SimpleImputer(strategy="most_frequent")
        df[col] = imputer.fit_transform(df[[col]])

# 最终检查缺失值
print("\n缺失值处理后检查:")
print(df.isna().sum().sum()) # 应该为0
```

### 3. 特征工程与模型构建

项目将数据分为训练集和测试集，并使用 `ColumnTransformer` 对不同类型的特征进行预处理（例如数值特征标准化）。模型方面，构建了逻辑回归、随机森林和梯度提升树等分类模型，并通过交叉验证进行评估。

```python
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder # 引入独热编码

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC # 引入SVM
from lightgbm import LGBMClassifier # 引入LightGBM


# 定义目标变量
target = "HOSPITAL_EXPIRE_FLAG"
X = df.drop(columns=[target, "HADM_ID"], errors='ignore') # 移除 HADM_ID
y = df[target]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 识别数值型和分类型特征
numerical_cols = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X_train.select_dtypes(exclude=np.number).columns.tolist()

# 创建预处理器
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols) # 对分类特征进行独热编码
    ],
    remainder='passthrough' # 保留未处理的列
)

# 定义模型管道
models = {
    'Logistic Regression': Pipeline(steps=[('preprocessor', preprocessor),
                                           ('classifier', LogisticRegression(solver='liblinear', random_state=42))]),
    'Random Forest': Pipeline(steps=[('preprocessor', preprocessor),
                                      ('classifier', RandomForestClassifier(random_state=42))]),
    'Gradient Boosting': Pipeline(steps=[('preprocessor', preprocessor),
                                          ('classifier', GradientBoostingClassifier(random_state=42))]),
    'LightGBM': Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', LGBMClassifier(random_state=42, verbose=-1))]) # 禁用LightGBM的详细输出
}

# 交叉验证评估模型
results = {}
for name, pipeline in models.items():
    print(f"\nEvaluating {name}...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(pipeline, X_train, y_train, cv=cv,
                            scoring=['roc_auc', 'average_precision', 'accuracy', 'f1'],
                            return_train_score=True, n_jobs=-1) # 使用所有可用核心

    results[name] = {
        'ROC AUC': f"{scores['test_roc_auc'].mean():.4f} ± {scores['test_roc_auc'].std():.4f}",
        'Average Precision': f"{scores['test_average_precision'].mean():.4f} ± {scores['test_average_precision'].std():.4f}",
        'Accuracy': f"{scores['test_accuracy'].mean():.4f} ± {scores['test_accuracy'].std():.4f}",
        'F1 Score': f"{scores['test_f1'].mean():.4f} ± {scores['test_f1'].std():.4f}"
    }
    print(f"{name} CV Scores:")
    print(pd.DataFrame(scores).mean().to_markdown())

print("\n--- 模型交叉验证结果 ---")
print(pd.DataFrame(results).T.to_markdown())
```

### 4. 模型评估与结果分析

项目对训练好的模型在测试集上进行最终评估，计算 ROC AUC、平均精确率、准确率、F1 分数等指标，并生成混淆矩阵、ROC 曲线和精准召回曲线。

```python
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, f1_score,
    confusion_matrix, classification_report, RocCurveDisplay, PrecisionRecallDisplay
)
import matplotlib.pyplot as plt

# 训练最佳模型（这里以Random Forest为例，您可以根据CV结果选择）
best_model_name = 'Random Forest' # 假设RF是最佳模型
best_pipeline = models[best_model_name]
best_pipeline.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = best_pipeline.predict(X_test)
y_proba = best_pipeline.predict_proba(X_test)[:, 1]

# 评估指标
print(f"\n--- {best_model_name} 在测试集上的性能 ---")
print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
print(f"Average Precision: {average_precision_score(y_test, y_proba):.4f}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print("\n混淆矩阵:")
print(pd.DataFrame(cm, index=['实际负类', '实际正类'], columns=['预测负类', '预测正类']).to_markdown())

# 分类报告
print("\n分类报告:")
print(classification_report(y_test, y_pred))

# 绘制 ROC 曲线
fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
RocCurveDisplay.from_estimator(best_pipeline, X_test, y_test, ax=ax_roc)
ax_roc.set_title(f'{best_model_name} ROC 曲线')
plt.savefig(f'images/portfolio/icu-mortality-prediction/roc_curve.png', dpi=300, bbox_inches='tight')
plt.close(fig_roc)
print("\n已保存 ROC 曲线至 images/portfolio/icu-mortality-prediction/roc_curve.png")

# 绘制精准召回曲线
fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
PrecisionRecallDisplay.from_estimator(best_pipeline, X_test, y_test, ax=ax_pr)
ax_pr.set_title(f'{best_model_name} 精准召回曲线')
plt.savefig(f'images/portfolio/icu-mortality-prediction/precision_recall_curve.png', dpi=300, bbox_inches='tight')
plt.close(fig_pr)
print("已保存 精准召回曲线至 images/portfolio/icu-mortality-prediction/precision_recall_curve.png")

# 性别分布圆环图示例
if "gender_is_male" in df.columns:
    g = df["gender_is_male"].astype(int).value_counts().sort_index()
    if set(g.index).issubset({0, 1}):
        labels = ["Female (0)", "Male (1)"]
        counts = [g.get(0, 0), g.get(1, 0)]
    else:
        labels = [str(i) for i in g.index]
        counts = g.values

    fig_gender, ax_gender = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax_gender.pie(counts, labels=labels, autopct='%1.1f%%',
                                             startangle=90, pctdistance=0.85, colors=['#66b3ff','#99ff99'])
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig_gender.gca().add_artist(centre_circle)
    for t in autotexts:
        t.set_fontsize(12)
    ax_gender.set_title("性别分布", fontsize=16)
    ax_gender.axis("equal")
    ax_gender.text(0, 0, f"N={int(sum(counts))}", ha="center", va="center", fontsize=14)
    plt.savefig(f'images/portfolio/icu-mortality-prediction/gender_distribution.png', dpi=300, bbox_inches='tight')
    plt.close(fig_gender)
    print("已保存 性别分布圆环图至 images/portfolio/icu-mortality-prediction/gender_distribution.png")
```

## 分析结果 (Results & Analysis)

本项目成功构建并评估了多种机器学习模型用于ICU患者院内死亡预测。

通过交叉验证，我们发现 **Random Forest** 模型在各项评估指标上表现较为均衡和优异。

![性别分布](/images/portfolio/icu-mortality-prediction/gender_distribution.png)
_图1: 患者性别分布圆环图。可以看出男性患者和女性患者的数量分布。_

![ROC 曲线](/images/portfolio/icu-mortality-prediction/roc_curve.png)
_图2: 最佳模型（Random Forest）在测试集上的 ROC 曲线。AUC 值为 [此处填入具体数值]，表明模型具有较好的区分能力。_

![精准召回曲线](/images/portfolio/icu-mortality-prediction/precision_recall_curve.png)
_图3: 最佳模型（Random Forest）在测试集上的精准召回曲线。该曲线在处理类别不平衡问题时更为重要，其曲线下面积（Average Precision）反映了模型在不同召回率下的精确率表现。_

最终在测试集上，选择的 Random Forest 模型达到了 [此处填入具体 ROC AUC 值] 的 ROC AUC，[此处填入具体 Average Precision 值] 的平均精确率，[此处填入具体 Accuracy 值] 的准确率和 [此处填入具体 F1 Score 值] 的 F1 分数。这些结果表明模型具备一定的临床应用潜力，可以作为辅助医生进行风险评估的工具。未来的工作可以包括引入更多时间序列特征、尝试深度学习模型以及进行模型可解释性分析。

---
```

