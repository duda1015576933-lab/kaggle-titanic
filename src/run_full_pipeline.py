# -*- coding: utf-8 -*-
"""Titanic 存活预测 —— 完整建模流程（适配斯坦福版数据）。

数据列：Survived, Pclass, Name, Sex, Age, Siblings/Spouses Aboard,
        Parents/Children Aboard, Fare

与 Kaggle 官方数据的差异：无 Embarked / PassengerId / Ticket / Cabin 列，
列名 Siblings/Spouses Aboard 等价于 SibSp，Parents/Children Aboard 等价于 Parch。
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无界面环境，输出图片文件
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

sns.set_style("whitegrid")
# 中文字体配置（Windows 自带微软雅黑/黑体）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE = r"C:/Users/10155/WorkBuddy/实习/kaggle-titanic"
DATA = os.path.join(BASE, "data", "train.csv")
OUT = os.path.join(BASE, "outputs")
os.makedirs(OUT, exist_ok=True)

# ---------- 1. 加载数据 ----------
df = pd.read_csv(DATA)
print("数据形状:", df.shape)
print("列名:", list(df.columns))
print("缺失值统计:\n", df.isnull().sum()[df.isnull().sum() > 0])

# ---------- 2. 重命名列以对齐标准命名 ----------
df = df.rename(columns={
    "Siblings/Spouses Aboard": "SibSp",
    "Parents/Children Aboard": "Parch",
})
# 统一性别为数值
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# ---------- 3. 特征工程 ----------
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
# 从姓名提取称谓
df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
rare = ["Lady", "Countess", "Capt", "Col", "Don", "Dr",
        "Major", "Rev", "Sir", "Jonkheer", "Dona"]
df["Title"] = df["Title"].replace(rare, "Rare")
df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
df["Title"] = df["Title"].astype("category").cat.codes

# 年龄缺失填补（整体中位数）
df["Age"] = df["Age"].fillna(df["Age"].median())

feature_cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare",
                "FamilySize", "IsAlone", "Title"]
X = df[feature_cols]
y = df["Survived"]

# ---------- 4. 切分训练/测试 ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n训练集 {X_train.shape[0]} 条，测试集 {X_test.shape[0]} 条")

# ---------- 5. 训练多个模型并对比 ----------
models = {
    "Logistic 回归": LogisticRegression(max_iter=1000, random_state=42),
    "随机森林": RandomForestClassifier(n_estimators=200, random_state=42),
    "梯度提升树": GradientBoostingClassifier(random_state=42),
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    cv = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    results.append((name, acc, auc, cv.mean(), cv.std()))
    print(f"\n=== {name} ===")
    print(f"测试集准确率: {acc:.4f}")
    print(f"测试集 AUC:   {auc:.4f}")
    print(f"5 折交叉验证准确率: {cv.mean():.4f} (±{cv.std():.4f})")
    print("混淆矩阵:\n", confusion_matrix(y_test, y_pred))
    print("分类报告:\n", classification_report(y_test, y_pred))

# ---------- 6. 汇总对比表 ----------
res_df = pd.DataFrame(results, columns=["模型", "测试准确率", "测试AUC", "CV均值", "CV标准差"])
res_df = res_df.sort_values("测试准确率", ascending=False).reset_index(drop=True)
print("\n===== 模型对比 =====")
print(res_df.to_string(index=False))
res_df.to_csv(os.path.join(OUT, "model_comparison.csv"), index=False, encoding="utf-8-sig")

# ---------- 7. 可视化：特征重要性与存活率 ----------
best_name = res_df.iloc[0]["模型"]
best_model = models[best_name]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 特征重要性（若为树模型）
if hasattr(best_model, "feature_importances_"):
    imp = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values()
    imp.plot(kind="barh", ax=axes[0], color="#185FA5")
    axes[0].set_title(f"{best_name} 特征重要性")
else:
    coef = pd.Series(best_model.coef_[0], index=feature_cols).sort_values()
    coef.plot(kind="barh", ax=axes[0], color="#185FA5")
    axes[0].set_title(f"{best_name} 系数")

# 性别与舱位对存活率的影响
sns.barplot(x="Sex", y="Survived", data=df, ax=axes[1], palette="Blues")
axes[1].set_xticklabels(["男性", "女性"])
axes[1].set_title("性别与存活率")

plt.tight_layout()
fig.savefig(os.path.join(OUT, "feature_analysis.png"), dpi=120)
print(f"\n图表已保存: {os.path.join(OUT, 'feature_analysis.png')}")

print("\n===== 流程完成 =====")
