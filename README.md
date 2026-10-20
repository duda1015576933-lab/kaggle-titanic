# Kaggle Titanic 存活预测

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

泰坦尼克号乘客存活预测 —— 入门级二分类建模项目，完整走通「数据清洗 → 特征工程 → 建模 → 评估」全流程。

## 项目背景

Titanic 是 Kaggle 公认的入门竞赛（[链接](https://www.kaggle.com/competitions/titanic)）。给定 891 名乘客的个人信息（年龄、性别、舱位等级、票价、登船港口等），预测其是否在沉船事故中存活。

## 数据

- 训练集：891 条样本，12 个特征（含目标变量 `Survived`）
- 测试集：418 条样本（用于提交预测结果）
- 数据来源：Kaggle `train.csv` / `test.csv`

## 分析流程

1. **数据概览与清洗**：缺失值统计与填补（年龄用中位数、登船港口用众数等）、异常值处理
2. **探索性分析（EDA）**：各特征与存活率的关系可视化
3. **特征工程**：分类变量编码、新特征构造（家庭规模、称谓等）
4. **建模**：Logistic 回归、随机森林、梯度提升树
5. **评估**：交叉验证、混淆矩阵、准确率与 ROC-AUC

## 结果

- 最佳模型（梯度提升树）5 折交叉验证准确率：**82.1% (±1.3%)**
- Logistic 回归测试集 AUC：0.858；三模型对比详见 `outputs/model_comparison.csv`
- 关键发现：性别是存活率最强预测因子（女性存活率 74% vs 男性 19%），舱位等级次之

> 注：当前使用斯坦福公开版数据（887 条）完成全流程验证；接入 Kaggle 官方数据（891 训练 + 418 测试）后可提交排行榜。

## 技术栈

- Python：pandas、numpy、scikit-learn、matplotlib、seaborn

## 项目结构

```
kaggle-titanic/
├── data/            # 训练/测试数据（不提交，见 .gitignore）
├── notebooks/       # 探索性分析与建模（Jupyter）
├── src/             # 可复用脚本（数据清洗、建模函数）
├── outputs/         # 生成的图表与预测结果
├── README.md
├── requirements.txt
└── .gitignore
```

## 运行方式

```bash
pip install -r requirements.txt
jupyter notebook notebooks/
```

## 作者

黄文彦（cosmohwang@163.com）
