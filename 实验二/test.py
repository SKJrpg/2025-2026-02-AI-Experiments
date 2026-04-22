#步骤 3：实验所需模块导入
# 数据处理库：处理表格数据、数值计算
import pandas as pd
import numpy as np
# 数据集划分：将数据分成训练集（训练模型）和测试集（验证模型）
from sklearn.model_selection import train_test_split
# 数据预处理：特征标准化（统一量纲）
from sklearn.preprocessing import StandardScaler
# 回归算法模型：线性回归、随机森林回归、XGBoost 回归
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
# 分类算法模型：逻辑回归（二分类）
from sklearn.linear_model import LogisticRegression
# 模型评估指标：回归和分类专用指标
from sklearn.metrics import mean_squared_error, r2_score # 回归指标（MSE、R²）
from sklearn.metrics import accuracy_score, confusion_matrix # 分类指标
# 数据可视化：绘制图表，直观查看结果
import matplotlib.pyplot as plt
import seaborn as sns
# 忽略无关警告（避免代码运行时出现多余警告，不影响结果）
import warnings
warnings.filterwarnings('ignore')
# 设置可视化样式：显示中文、取消负号乱码（必加，否则图表中文乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#任务二
#步骤2：数据集加载
from sklearn.datasets import load_diabetes
# 加载糖尿病数据集
diabetes = load_diabetes()
# 转换为DataFrame 格式（类似Excel 表格），便于查看和处理数据
diabetes_df = pd.DataFrame(data=diabetes.data,
columns=diabetes.feature_names)
# 添加目标变量（病情指标）列，便于后续分析
diabetes_df['target'] = diabetes.target
print('数据集形状（样本数×特征数）：', diabetes_df.shape) # 查看数据量
print('数据集缺失值检查：', diabetes_df.isnull().sum()) # 查看是否有缺失值（均为0）
# 提取特征矩阵X（所有特征）和目标变量y（病情指标），为后续模型训练做准备
X = diabetes_df.drop('target', axis=1) # 去掉病情列，剩下的都是特征
y = diabetes_df['target'] # 病情列作为目标变量
# 简单查看特征与病情的相关性（热力图，直观易懂）
plt.figure(figsize=(10, 8))
# 计算特征间的相关系数（取值范围：-1 到1）
# 1.0 = 完全正相关（同增同减）
# -1.0 = 完全负相关（此消彼长）
# 0 = 不相关
correlation_matrix = diabetes_df.corr()
# 绘制热力图
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix,
annot=True, # 显示数字
cmap='coolwarm', # 红色=正相关，蓝色=负相关
fmt='.2f', # 保留2 位小数
center=0, # 颜色中点设为0
vmin=-1, vmax=1) # 固定范围-1 到1
plt.title('糖尿病数据集特征相关性热力图\n(红色=正相关，蓝色=负相关，颜色越深相关性越强)')
plt.show()

#步骤3：数据预处理
# 初始化标准化器（用于统一特征量纲，让算法训练更准确）
scaler = StandardScaler()
# 对特征矩阵X 进行标准化处理（只标准化特征，不标准化目标变量y）
X_scaled = scaler.fit_transform(X)
# 查看标准化效果
print('标准化前特征均值（示例）：', X.iloc[:,0].mean())
print('标准化后特征均值（示例）：', round(X_scaled[:,0].mean(), 2))
# 标准化后均值接近0

#步骤4：数据集划分
# 数据集划分：训练集70%，测试集30%
# random_state=42：保证每次运行代码，划分结果都一样（便于复现）
X_train, X_test, y_train, y_test = train_test_split(
X_scaled, y, test_size=0.3, random_state=42
)
# 打印划分后的数据量
print('训练集样本数：', X_train.shape[0])
print('测试集样本数：', X_test.shape[0])

#步骤5：回归模型训练
# 1. 训练线性回归模型
lr = LinearRegression()
lr.fit(X_train, y_train) # 用训练集训练模型
print('线性回归模型训练完成！')
# 2. 训练随机森林回归模型
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
print('随机森林回归模型训练完成！')
# 3. 训练XGBoost 回归模型
xgb_reg = xgb.XGBRegressor(random_state=42,
objective='reg:squarederror')
xgb_reg.fit(X_train, y_train)
print('XGBoost 回归模型训练完成！')

#步骤6：回归模型评估
# 定义评估函数
def evaluate_model(model, model_name):
    y_pred = model.predict(X_test) # 用模型预测测试集病情指标
    mse = mean_squared_error(y_test, y_pred) # 计算MSE
    r2 = r2_score(y_test, y_pred) # 计算R²
    print(f'\n{model_name}评估结果：')
    print(f'均方误差（MSE）：{mse:.4f}（越小越好）')
    print(f'决定系数（R²）：{r2:.4f}（越接近1 越好）')
    return y_pred
# 评估三个模型，得到预测值（用于后续绘图）
y_pred_lr = evaluate_model(lr, '线性回归')
y_pred_rf = evaluate_model(rf, '随机森林回归')
y_pred_xgb = evaluate_model(xgb_reg, 'XGBoost 回归')
# 可视化：只绘制预测值与真实值对比（直观查看预测效果）
plt.figure(figsize=(15, 5))
# 线性回归预测图
plt.subplot(1, 3, 1)
plt.scatter(y_test, y_pred_lr, alpha=0.6, color='steelblue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(),
y_test.max()], 'r--') # 完美预测线
plt.xlabel('真实病情指标')
plt.ylabel('预测病情指标')
plt.title('线性回归预测值vs 真实值')
# 随机森林回归预测图
plt.subplot(1, 3, 2)
plt.scatter(y_test, y_pred_rf, alpha=0.6, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(),
y_test.max()], 'r--')
plt.xlabel('真实病情指标')
plt.ylabel('预测病情指标')
plt.title('随机森林回归预测值vs 真实值')
# XGBoost 回归预测图
plt.subplot(1, 3, 3)
plt.scatter(y_test, y_pred_xgb, alpha=0.6, color='orange')
plt.plot([y_test.min(), y_test.max()], [y_test.min(),
y_test.max()], 'r--')
plt.xlabel('真实病情指标')
plt.ylabel('预测病情指标')
plt.title('XGBoost 回归预测值vs 真实值')
plt.tight_layout()
plt.show()

#任务三
#步骤2：数据集加载与探索
# 加载威斯康星乳腺癌数据集
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
# 转换为DataFrame 格式，便于查看
cancer_df = pd.DataFrame(data=cancer.data,
columns=cancer.feature_names)
cancer_df['TARGET'] = cancer.target # 添加肿瘤类型列
# 打印核心数据
print('数据集形状（样本数×特征数）：', cancer_df.shape)
print('缺失值检查：', cancer_df.isnull().sum()) # 无缺失值
print('肿瘤类型分布：',
cancer_df['TARGET'].value_counts().rename({0:'恶性', 1:'良性'}))
# 提取特征和目标变量
X_c = cancer_df.drop('TARGET', axis=1)
y_c = cancer_df['TARGET']

#步骤3：数据预处理
# 标准化特征（统一量纲，提升模型训练效果）
scaler_c = StandardScaler()
X_c_scaled = scaler_c.fit_transform(X_c)

#步骤4：数据集划分
# 训练集70%，测试集30%，固定随机种子，保证可复现
X_c_train, X_c_test, y_c_train, y_c_test = train_test_split(
X_c_scaled, y_c, test_size=0.3, random_state=42
)
print('训练集样本数：', X_c_train.shape[0])
print('测试集样本数：', X_c_test.shape[0])

#步骤5：逻辑回归模型训练
# 初始化逻辑回归模型（max_iter=1000，保证模型能收敛）
lr_classifier = LogisticRegression(random_state=42, max_iter=1000)
# 用训练集训练模型
lr_classifier.fit(X_c_train, y_c_train)
print('逻辑回归分类模型训练完成！')

#步骤6：逻辑回归模型评估
# 用测试集预测
y_c_pred = lr_classifier.predict(X_c_test)
# 计算核心评估指标（准确率）
acc = accuracy_score(y_c_test, y_c_pred)
print(f'逻辑回归模型准确率：{acc:.4f}（越接近1 越好）')
# 绘制混淆矩阵（直观查看预测对错的数量）
cm = confusion_matrix(y_c_test, y_c_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
xticklabels=['恶性', '良性'],
yticklabels=['恶性', '良性'])
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title(f'逻辑回归混淆矩阵（准确率={acc:.4f}）')
plt.show()
# 混淆矩阵解读
print('\n 混淆矩阵解读：')
print('左上角数字：真实为恶性、预测为恶性的样本数（预测正确）')
print('右上角数字：真实为恶性、预测为良性的样本数（预测错误）')
print('左下角数字：真实为良性、预测为恶性的样本数（预测错误）')
print('右下角数字：真实为良性、预测为良性的样本数（预测正确）')