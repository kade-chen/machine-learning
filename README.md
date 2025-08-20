# Wondercloud 机器学习

## 项目简介

简要介绍项目背景和目标，比如：

> 本项目基于出租车行程数据，训练了一个线性回归模型用于预测行程总费用。模型通过 Google Cloud Vertex AI 进行部署，实现在线预测和模型解释。

---

## 环境准备

- Python 3.11+
- Google Cloud SDK 已安装并配置
- 服务账号 JSON 文件（需要 Vertex AI 权限）

---

## 安装依赖

```bash
pip install -r requirements.txt
python3.11 -m pip install python-dotenv
python3.11 -m pip install scikit-learn
python3.11 -m pip install matplotlib
python3.11 -m pip install pandas
python3.11 -m pip install db-dtypes
python3.11 -m pip install google-cloud-bigquery-storage
python3.11 -m pip install seaborn
```

###
- 
- credentials
- dataset
- data preprocess
- training
- upload gcs
- model upload
- create endpoint
- deploy endpoint


2222222