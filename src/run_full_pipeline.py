# -*- coding: utf-8 -*-
"""Titanic 存活预测 —— 官方数据完整建模流程。

数据：Kaggle 官方 train.csv（891 条）/ test.csv（418 条）
流程：数据概览 → 缺失值处理 → 特征工程 → 多模型训练与交叉验证
     → 生成提交文件 → 导出处理后数据

运行：python src/run_full_pipeline.py
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

sns.set_style("whitegrid")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE = r"C:/Users/10155/WorkBuddy/实习/kaggle-titanic"
DATA_DIR = os.path.join(BASE, "data")
OUT_DIR = os.path.join(BASE, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42


# ============================================================
# 1. 数据加载与概览
# ============================================================
def load_data():
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    return train, test


def overview(train):
    print("=" * 55)
    print("数据概览")
    print("=" * 55)
    print(f"训练集形状: {train.shape}")
    print(f"目标变量分布:\n{train['Survived'].value_counts().to_string()}")
    mis = train.isnull().sum()
    mis = mis[mis > 0]
    print(f"\n缺失值:\n{mis.to_string()}")


# ============================================================
# 2. 数据清洗与特征工程（train / test 统一处理）
# ============================================================
def preprocess(df):
    """缺失值处理 + 特征工程。返回处理后的 DataFrame。"""
    df = df.copy()

    # --- 缺失值处理 ---
    # Age：按「性别 + 舱位等级」分组中位数填补（比整体中位数更精细）
    df["Age"] = df.groupby(["Sex", "Pclass"])["Age"].transform(
        lambda x: x.fillna(x.median())
    )
    # Embarked：用众数填补
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    # Fare：中位数填补
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    # Cabin：缺失率高达 77%，不直接填补，而是提取两个新特征
    df["HasCabin"] = df["Cabin"].notna().astype(int)
    df["Deck"] = df["Cabin"].str[0].fillna("U")  # 首字母 = 甲板层，U 表示未知
    df = df.drop(columns=["Cabin", "Ticket"])  # 丢弃原始 Cabin/Ticket

    # --- 特征工程 ---
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    # 从姓名提取称谓
    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    rare_titles = ["Lady", "Countess", "Capt", "Col", "Don", "Dr",
                   "Major", "Rev", "Sir", "Jonkheer", "Dona"]
    df["Title"] = df["Title"].replace(rare_titles, "Rare")
    df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    df = df.drop(columns=["Name"])

    # 类别变量数值编码
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})
    df["Deck"] = df["Deck"].astype("category").cat.codes
    df["Title"] = df["Title"].astype("category").cat.codes

    return df


FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked",
            "FamilySize", "IsAlone", "Title", "HasCabin", "Deck"]


# ============================================================
# 3. 模型训练与评估
# ============================================================
def build_models():
    return {
        "Logistic 回归": LogisticRegression(max_iter=1000, random_state=SEED),
        "支持向量机": make_pipeline(
            StandardScaler(),
            CalibratedClassifierCV(SVC(random_state=SEED), cv=3),
        ),
        "随机森林": RandomForestClassifier(n_estimators=300, random_state=SEED),
        "梯度提升树": GradientBoostingClassifier(random_state=SEED),
    }


def evaluate_all(models, X, y):
    """用 5 折分层交叉验证评估所有模型，返回结果表。"""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for name, model in models.items():
        cv_acc = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
        cv_auc = cross_val_score(model, X, y, cv=skf, scoring="roc_auc")
        rows.append({
            "模型": name,
            "CV准确率均值": round(cv_acc.mean(), 4),
            "CV准确率标准差": round(cv_acc.std(), 4),
            "CV-AUC均值": round(cv_auc.mean(), 4),
        })
    return pd.DataFrame(rows).sort_values("CV准确率均值", ascending=False).reset_index(drop=True)


# ============================================================
# 主流程
# ============================================================
def main():
    train, test = load_data()
    overview(train)

    # 预处理（训练集保留 Survived 与 PassengerId）
    train_id = train["PassengerId"]
    test_id = test["PassengerId"]
    train_pp = preprocess(train)
    test_pp = preprocess(test)

    X = train_pp[FEATURES]
    y = train_pp["Survived"]

    print("\n" + "=" * 55)
    print("特征矩阵")
    print("=" * 55)
    print(f"特征数量: {len(FEATURES)}")
    print(f"特征列表: {FEATURES}")

    # 训练并评估
    models = build_models()
    result_df = evaluate_all(models, X, y)
    print("\n" + "=" * 55)
    print("模型交叉验证对比（5 折分层）")
    print("=" * 55)
    print(result_df.to_string(index=False))
    result_df.to_csv(os.path.join(OUT_DIR, "model_comparison.csv"),
                     index=False, encoding="utf-8-sig")

    # 用 CV 最优模型生成提交文件
    best_name = result_df.iloc[0]["模型"]
    best_model = models[best_name]
    best_model.fit(X, y)
    test_pred = best_model.predict(test_pp[FEATURES])

    submission = pd.DataFrame({
        "PassengerId": test_id,
        "Survived": test_pred,
    })
    submission_path = os.path.join(OUT_DIR, "submission.csv")
    submission.to_csv(submission_path, index=False)
    print(f"\n最优模型: {best_name}")
    print(f"提交文件已生成: {submission_path}（{len(submission)} 条预测）")
    print(f"预测存活人数占比: {submission['Survived'].mean():.1%}")

    # 导出处理后的数据
    train_out = pd.concat([train_id, train_pp], axis=1)
    test_out = pd.concat([test_id, test_pp], axis=1)
    train_out.to_csv(os.path.join(OUT_DIR, "train_processed.csv"),
                     index=False, encoding="utf-8-sig")
    test_out.to_csv(os.path.join(OUT_DIR, "test_processed.csv"),
                    index=False, encoding="utf-8-sig")
    print("处理后数据已导出: train_processed.csv / test_processed.csv")

    # 可视化：特征重要性 + 关键特征与存活率
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 特征重要性（取梯度提升树）
    imp_model = models["梯度提升树"]
    imp_model.fit(X, y)
    imp = pd.Series(imp_model.feature_importances_, index=FEATURES).sort_values()
    imp.plot(kind="barh", ax=axes[0, 0], color="#185FA5")
    axes[0, 0].set_title("特征重要性（梯度提升树）")

    sns.barplot(x="Sex", y="Survived", data=train_pp, ax=axes[0, 1], color="#185FA5")
    axes[0, 1].set_xticklabels(["男性", "女性"])
    axes[0, 1].set_title("性别与存活率")

    sns.barplot(x="Pclass", y="Survived", data=train_pp, ax=axes[1, 0], color="#185FA5")
    axes[1, 0].set_title("舱位等级与存活率")

    sns.barplot(x="Title", y="Survived", data=train_pp, ax=axes[1, 1], color="#185FA5")
    axes[1, 1].set_title("称谓与存活率")

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "feature_analysis.png"), dpi=120)
    print(f"图表已保存: {os.path.join(OUT_DIR, 'feature_analysis.png')}")

    print("\n===== 全流程完成 =====")


if __name__ == "__main__":
    main()
