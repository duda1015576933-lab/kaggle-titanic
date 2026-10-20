# Titanic 存活预测 —— 数据处理与建模说明

> 本文档记录本项目的数据来源、清洗策略、特征工程与建模方法，便于复现与面试回顾。

## 一、数据来源

- 数据：Kaggle 竞赛 [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) 官方数据
- 训练集 `train.csv`：891 条样本，12 个字段（含目标变量 `Survived`）
- 测试集 `test.csv`：418 条样本，11 个字段（无目标变量，用于生成提交预测）

## 二、字段说明

| 字段 | 含义 | 类型 |
|------|------|------|
| PassengerId | 乘客编号 | 标识 |
| Survived | 是否存活（0=死亡，1=存活） | 目标变量 |
| Pclass | 舱位等级（1/2/3 等） | 有序类别 |
| Name | 姓名 | 文本 |
| Sex | 性别 | 类别 |
| Age | 年龄 | 数值 |
| SibSp | 同行的兄弟姐妹/配偶数 | 数值 |
| Parch | 同行的父母/子女数 | 数值 |
| Ticket | 船票编号 | 文本 |
| Fare | 票价 | 数值 |
| Cabin | 船舱号 | 文本 |
| Embarked | 登船港口（S/C/Q） | 类别 |

## 三、数据清洗策略

| 字段 | 缺失数 | 处理方式 | 理由 |
|------|--------|---------|------|
| Age | 177 | 按「性别 + 舱位等级」分组中位数填补 | 年龄与性别、舱位强相关，分组中位数比整体中位数更精确 |
| Embarked | 2 | 众数填补 | 缺失极少，众数（S 港）是合理近似 |
| Fare | 1（测试集） | 中位数填补 | 单值缺失，中位数稳妥 |
| Cabin | 687（77%） | 不填补，转为两个新特征 | 缺失率过高，直接填补会引入噪声 |

## 四、特征工程

1. **HasCabin**：是否有船舱记录（Cabin 非空 = 1）。有船舱的乘客多为高舱位，存活率更高。
2. **Deck**：船舱号首字母 = 甲板层，缺失记为 U（Unknown）。不同甲板层对应不同舱位等级与位置。
3. **FamilySize**：`SibSp + Parch + 1`，家庭规模。独行者与大家庭的存活率存在差异。
4. **IsAlone**：是否独自出行（FamilySize == 1）。
5. **Title**：从姓名提取称谓（Mr/Mrs/Miss/Master/Rare）。称谓隐含性别、年龄与社会地位信息。

最终进入模型的 12 个特征：

```
Pclass, Sex, Age, SibSp, Parch, Fare, Embarked,
FamilySize, IsAlone, Title, HasCabin, Deck
```

## 五、建模与评估

- 评估方式：**5 折分层交叉验证**（`StratifiedKFold`），保证每折中存活/死亡比例与总体一致
- 指标：准确率（Accuracy）与 ROC-AUC
- 模型对比：

| 模型 | CV 准确率 | CV 标准差 | CV-AUC |
|------|----------|----------|--------|
| 梯度提升树（GradientBoosting） | **83.95%** | ±2.47% | **0.884** |
| 支持向量机（SVM + 标准化） | 83.39% | ±1.06% | 0.865 |
| 随机森林（RandomForest） | 81.93% | ±1.11% | 0.872 |
| Logistic 回归 | 80.02% | ±2.20% | 0.859 |

> 注：SVM 对特征尺度敏感，使用 `StandardScaler` 标准化后性能从 68.9% 提升至 83.4%，说明特征标准化对基于距离/间隔的模型至关重要。

## 六、核心结论

1. **性别是最强预测因子**：女性存活率约 74%，男性约 19%（海上救援遵循"妇孺优先"）。
2. **舱位等级显著**：一等舱存活率最高，三等舱最低，与票价、社会地位强相关。
3. **最优模型为梯度提升树**，5 折交叉验证准确率 83.95%、AUC 0.884。

## 七、复现方式

```bash
pip install -r requirements.txt
python src/run_full_pipeline.py
```

脚本会自动完成数据加载 → 清洗 → 特征工程 → 建模 → 交叉验证 → 生成提交文件全流程。

## 八、产出文件

| 文件 | 说明 |
|------|------|
| `outputs/submission.csv` | Kaggle 提交文件（418 条预测） |
| `outputs/train_processed.csv` | 处理后的训练数据（891 条 × 含新增特征） |
| `outputs/test_processed.csv` | 处理后的测试数据（418 条 × 含新增特征） |
| `outputs/model_comparison.csv` | 四模型交叉验证对比表 |
| `outputs/feature_analysis.png` | 特征重要性与关键特征可视化 |
