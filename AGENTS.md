# AI 通识课实验仓库 — AGENTS.md

## 仓库定位

大一非 CS 专业学生的人工智能通识课实验作业。三个独立实验，彼此无依赖。

## 重要事实（不然容易踩坑）

- **README.md 已过时** — 只列了实验一和实验二，漏掉了实验三。不要把它当完整目录索引。
- **没有 `.gitignore`** — `git status` 会看到 `*.png` 等生成物（图片在 `data/` 下）。如果你在修改或清理，注意不要误删生成的图片。
- **没有 `requirements.txt`** — 依赖：`pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`。安装方式：
  ```bash
  pip install pandas numpy scikit-learn xgboost matplotlib seaborn
  ```
- **没有测试框架、没有 lint、没有 type check** — 代码直接 `python 实验二/test.py` 运行即可。
- **没有 CI/CD** — 仓库仅用于课程提交，不部署。
- **所有文档和注释使用中文** — 实验报告和代码注释都是中文。如果你补充内容，保持中文。

## 实验一览

| 实验 | 目录 | 内容 | 状态 |
|------|------|------|------|
| 实验一 | `实验一/` | 北京/南京/广州逐小时温度数据分析清洗、统计、可视化 | 已完成，有报告+PDF |
| 实验二 | `实验二/` | 经典 ML：线性回归、随机森林、XGBoost 回归 + 逻辑回归分类 | 已完成，有代码+报告 |
| 实验三 | `实验三/` | 深度学习：手写数字识别（FCNN + CNN） | 已完成，有代码+报告 |

### 实验一 — 温度数据处理

- 数据文件：`data/Beijing_cleaned.csv`, `Beijing_filled.csv` 等（北京、南京、广州各两个文件，cleaned=清洗后，filled=填充缺失后）
- 核心报告：`实验1.md`（包含分析结论 + 绘图代码）
- 输出图片：`data/myplot.png`
- 其他文件：`Test.docx`（可能是原始数据或草稿，不关键）

### 实验二 — 经典机器学习

- **唯一可运行的 Python 代码**：`实验二/test.py`
- 直接运行：`python 实验二/test.py`
- 数据集：`sklearn.datasets.load_diabetes`（回归）+ `load_breast_cancer`（分类）
- 模型：LinearRegression → RandomForestRegressor → XGBRegressor → LogisticRegression（按顺序执行）
- 标准化：StandardScaler，训练/测试 7:3 划分，`random_state=42`
- matplotlib 字体配置 `SimHei` + `axes.unicode_minus = False` 解决中文乱码（Windows 需要安装 SimHei 字体，否则自行修改）
- 输出图片保存在 `data/` 下

### 实验三 — 深度学习（手写数字识别）

- **可运行代码**：`实验三/test.py`
- 使用 `d2l` Anaconda 环境（路径 `D:\Anaconda_envs\envs\d2l\python.exe`）
- 框架：PyTorch，包含 FCNN 和 CNN 两种模型
- 数据集：MNIST（通过 torchvision 自动下载到 `data/MNIST/`）
- 训练 5 个 epoch，使用 Adam 优化器，CrossEntropyLoss
- 输出图片保存在 `data/` 下
- 报告：`实验三/深度学习——手写数字识别实验报告.md`

## 处理建议

- **添加 `.gitignore`**：建议忽略 `__pycache__/`, `*.pyc`, `data/MNIST/`（数据集自动下载，无需提交）。实验结果图片（`data/*.png`）可根据需要决定是否提交。

## Git 信息

- 作者：苏凯军
- 提交记录中 `REBUILD` 是为重新组织仓库结构做的重置，不影响内容。
