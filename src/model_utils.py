# -*- coding: utf-8 -*-
"""模型训练与评估模块。

封装特征编码、模型训练、交叉验证与结果输出等流程，
供 notebooks/ 中的建模脚本调用。
"""
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def encode_categorical(df, columns):
    """将分类变量转为数值编码（类别码）。

    Args:
        df: 待编码的 DataFrame。
        columns: 需要编码的列名列表。

    Returns:
        编码后的 DataFrame。
    """
    df = df.copy()
    for col in columns:
        df[col] = df[col].astype("category").cat.codes
    return df


def train_logistic(X, y, random_state=42):
    """训练 Logistic 回归模型。

    Args:
        X: 特征矩阵。
        y: 标签。
        random_state: 随机种子。

    Returns:
        训练好的模型。
    """
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X, y)
    return model


def train_random_forest(X, y, n_estimators=100, random_state=42):
    """训练随机森林模型。

    Args:
        X: 特征矩阵。
        y: 标签。
        n_estimators: 树的数量。
        random_state: 随机种子。

    Returns:
        训练好的模型。
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state
    )
    model.fit(X, y)
    return model


def evaluate(model, X, y, cv=5):
    """评估模型：交叉验证准确率 + 混淆矩阵 + 分类报告。

    Args:
        model: 已训练模型。
        X: 特征矩阵。
        y: 标签。
        cv: 交叉验证折数。

    Returns:
        (cv_scores, conf_matrix, report) 三元组。
    """
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    y_pred = model.predict(X)
    conf_matrix = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred)
    return cv_scores, conf_matrix, report


def save_submission(model, test_df, feature_columns, output_path="../outputs/submission.csv"):
    """生成 Kaggle 提交文件。

    Args:
        model: 已训练模型。
        test_df: 测试集 DataFrame（需含 PassengerId 与特征列）。
        feature_columns: 用于预测的特征列名列表。
        output_path: 输出 CSV 路径。
    """
    X_test = test_df[feature_columns]
    pred = model.predict(X_test)
    submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": pred,
    })
    submission.to_csv(output_path, index=False)
    print(f"提交文件已生成：{output_path}（{len(submission)} 条预测）")
    return submission
