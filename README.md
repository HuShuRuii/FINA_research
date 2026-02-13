# 机器学习学习项目

这是一个用于学习机器学习的项目环境。

## 环境配置步骤

### 1. 创建虚拟环境（推荐）

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 Jupyter Notebook（可选）

如果你想使用 Jupyter Notebook：

```bash
python -m ipykernel install --user --name=ml_env
```

### 4. 验证安装

运行以下命令验证主要库是否安装成功：

```python
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns

print("所有库安装成功！")
print(f"NumPy版本: {np.__version__}")
print(f"Pandas版本: {pd.__version__}")
print(f"Scikit-learn版本: {sklearn.__version__}")
```

## 项目结构

```
machine_learning_gogogo/
├── data/              # 数据文件目录
├── notebooks/         # Jupyter Notebook文件
├── src/              # 源代码目录
├── models/           # 保存的模型文件
├── requirements.txt  # 依赖包列表
└── README.md         # 项目说明
```

## 开始学习

1. 在 `notebooks/` 目录中创建你的第一个学习笔记
2. 在 `data/` 目录中存放数据集
3. 在 `src/` 目录中编写可重用的代码模块

## 常用命令

- 启动 Jupyter Notebook: `jupyter notebook`
- 启动 Jupyter Lab: `jupyter lab`
- 退出虚拟环境: `deactivate`

## 学习资源推荐

- [Scikit-learn 官方文档](https://scikit-learn.org/stable/)
- [Pandas 官方文档](https://pandas.pydata.org/)
- [NumPy 官方文档](https://numpy.org/)

祝学习愉快！🚀
