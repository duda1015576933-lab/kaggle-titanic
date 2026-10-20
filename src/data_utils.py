# -*- coding: utf-8 -*-
"""数据加载与清洗模块。

提供 Titanic 数据的读取、缺失值处理、特征工程等可复用函数，
供 notebooks/ 中的分析脚本调用。
"""
import pandas as pd
import numpy as np

DATA_DIR = "../data"


def load_data():
    """加载训练集与测试集。

    Returns:
        (train_df, test_df): 两个 DataFrame。若 data/ 目录下无 CSV 文件则抛出 FileNotFoundError。
    """
    import os
    train_path = os.path.join(DATA_DIR, "train.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            "未找到 train.csv，请先从 Kaggle 下载数据放入 data/ 目录。"
        )
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def summarize_missing(df):
    """统计各列缺失值数量与占比。

    Args:
        df: 待检查的 DataFrame。

    Returns:
        DataFrame，含缺失数量与缺失比例两列，按缺失比例降序排列。
    """
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    summary = pd.DataFrame({
        "missing_count": missing,
        "missing_pct": missing_pct.round(2),
    })
    return summary[summary["missing_count"] > 0].sort_values(
        "missing_count", ascending=False
    )


def impute_age(df, by_group=True):
    """填补年龄缺失值。

    默认按「性别 + 舱位等级」分组的中位数填补；关闭 by_group 时用整体中位数。

    Args:
        df: 含 Age 列的 DataFrame。
        by_group: 是否按分组中位数填补。

    Returns:
        填补后的 DataFrame（就地修改并返回）。
    """
    df = df.copy()
    if by_group:
        df["Age"] = df.groupby(["Sex", "Pclass"])["Age"].transform(
            lambda x: x.fillna(x.median())
        )
    else:
        df["Age"] = df["Age"].fillna(df["Age"].median())
    return df


def build_features(df):
    """构造衍生特征：家庭规模、是否独行、称谓。

    Args:
        df: 含 SibSp、Parch、Name 列的 DataFrame。

    Returns:
        新增 FamilySize、IsAlone、Title 三列后的 DataFrame。
    """
    df = df.copy()
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    # 从姓名中提取称谓（Mr / Mrs / Miss / Master / 其他）
    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    rare_titles = [
        "Lady", "Countess", "Capt", "Col", "Don", "Dr",
        "Major", "Rev", "Sir", "Jonkheer", "Dona",
    ]
    df["Title"] = df["Title"].replace(rare_titles, "Rare")
    df["Title"] = df["Title"].replace(
        {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}
    )
    return df
