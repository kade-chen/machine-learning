import random
import string
import subprocess


# load the required libraries start
import pickle

import matplotlib.pyplot as plt
# load the required libraries
import pandas as pd
import seaborn as sns
from google.cloud import aiplatform, storage
from google.cloud.aiplatform_v1.types import SampledShapleyAttribution
from google.cloud.aiplatform_v1.types.explanation import ExplanationParameters
from google.cloud.bigquery import Client

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from google.oauth2 import service_account
# load the required libraries end

PROJECT_ID = "genai-specialization-468108"  # 把your-project-id替换成你自己的项目ID字符串
LOCATION1 = "us"  # 这是区域，通常保持这个或你需要
LOCATION = "us-central1"  # 这是区域，通常保持这个或你需要
BUCKET_URI = f"gs://kade-gcs" 

# Generate a uuid of length 8
def generate_uuid():
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=8))


UUID = generate_uuid()

print(UUID,PROJECT_ID,LOCATION)

##执行完注释掉了
# cmd = f"gsutil mb -l {LOCATION} -p {PROJECT_ID} {BUCKET_URI}"

# result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

# print("stdout:", result.stdout)
# print("stderr:", result.stderr)

# The following two lines are only necessary to run once.
# Comment out otherwise for speed-up.

### pip install google-cloud-bigquery-storage 安装完走BigQuery Storage API 会更快
service_account_key_path="/Users/kade.chen/go-kade-project/github/mcenter/etc/kade-poc.json"
credentials = service_account.Credentials.from_service_account_file(
    service_account_key_path,
    scopes=['https://www.googleapis.com/auth/cloud-platform']
)

client = Client(
    project=PROJECT_ID,
    location=LOCATION1,
    credentials=credentials,
)

query = """select 
taxi_id, trip_start_timestamp, 
trip_seconds, trip_miles, trip_total, 
payment_type, pickup_community_area, 
dropoff_community_area 
from `bigquery-public-data.chicago_taxi_trips.taxi_trips`
where 
trip_start_timestamp >= '2018-05-12' and 
trip_end_timestamp <= '2018-05-18' and
trip_seconds > 60 and trip_seconds < 6*60*60 and
trip_miles > 0 and
trip_total > 3 and
pickup_community_area is not NULL and 
dropoff_community_area is not NULL"""

job = client.query(query)  # 使用 BigQuery 客户端提交SQL查询任务
df = job.to_dataframe()   # 将查询结果转成 Pandas DataFrame 方便数据分析

print(df.shape)           # 输出数据维度（行数和列数）
df.columns                # 查看DataFrame所有列名
df.head()                 # 显示前5条数据，快速预览内容
df.dtypes                 # 查看每列的数据类型
df.info()                 # 输出DataFrame详细信息（非空值数量、类型、内存）
df.describe().T           # 对数值列做统计描述并转置，方便查看均值、标准差等指标

# 你要做一个机器学习任务，目标是预测 trip_total （出租车行程总费用），这个字段就是你的目标变量（target）。
target = "trip_total"
	# •	payment_type（付款类型，比如现金、信用卡等）
	# •	pickup_community_area（上车地点所在社区区域编号）
	# •	dropoff_community_area（下车地点所在社区区域编号）
categ_cols = ["payment_type", "pickup_community_area", "dropoff_community_area"]
	# •	trip_seconds（行程时间，单位是秒）
	# •	trip_miles（行程距离，单位是英里）
num_cols = ["trip_seconds", "trip_miles"]


	# •	直方图（Histogram）：看数据的分布情况，比如集中在哪些值，是否偏斜，是否有多峰等。
	# •	箱线图（Boxplot）：看数据的分布情况和异常值（离群点）。
# 遍历数值型特征列 + 目标列
for i in num_cols + [target]:
    # 创建一个 1 行 2 列的图像区域（figsize=(12, 4) 表示整个图宽 12 英寸，高 4 英寸）
    _, ax = plt.subplots(1, 2, figsize=(12, 4))
    
    # 绘制直方图（Histogram）
    df[i].plot(kind="hist", bins=100, ax=ax[0])  # bins=100 表示分成 100 个柱子
    ax[0].set_title(str(i) + " -Histogram")     # 设置直方图标题
    
    # 绘制箱型图（Boxplot）
    df[i].plot(kind="box", ax=ax[1])             # 直接在右侧子图画箱线图
    ax[1].set_title(str(i) + " -Boxplot")        # 设置箱型图标题
    
    # 显示当前图像
    plt.show()

# trip_seconds 是行程时间，单位是秒。为了分析更方便，把它换算成小时：
df["trip_hours"] = round(df["trip_seconds"] / 3600, 2)
# 画出这个新列的箱线图（box plot），可以看出行程时间的分布情况和异常值
df["trip_hours"].plot(kind="box")
plt.show()
     
# 用里程数除以小时数，得到平均速度（英里/小时）：
df["trip_speed"] = round(df["trip_miles"] / df["trip_hours"], 2)
# 同样画箱线图查看速度的分布和异常值。
df["trip_speed"].plot(kind="box")
plt.show()


# generate a pairplot for 10K samples
# 用 Seaborn 的 pairplot，随机抽样1万条数据，画出几个数值变量之间的两两关系散点图矩阵：
try:
    sns.pairplot(
        # trip_seconds（行程时间）和 trip_miles（行程里程）是线性相关的（一般时间越长，距离越远）
        # 这两个变量和目标变量 trip_total（行程总价）也有一定的关系
        data=df[["trip_seconds", "trip_miles", "trip_total", "trip_speed"]].sample(
            10000
        )
    )
    plt.show()
except Exception as e:
    print(e)
     

# 过滤掉 trip_total <= 3 的数据（总价太低可能异常）
df = df[df["trip_total"] > 3]

# trip_miles 限制在0到300英里之间，过滤异常里程
df = df[(df["trip_miles"] > 0) & (df["trip_miles"] < 300)]

# 行程时间至少1分钟
df = df[df["trip_seconds"] >= 60]

# 乘车时间最多2小时
df = df[df["trip_hours"] <= 2]

# 行驶速度限制在每小时70英里以内，剔除异常速度
df = df[df["trip_speed"] <= 70]

# 重置索引
df.reset_index(drop=True, inplace=True)

# 查看过滤后数据的形状（行数，列数）
print(df.shape)

# 对分类列循环，打印类别数量，画出类别比例柱状图
for i in categ_cols:
    print(f"Unique values in {i}:", df[i].nunique())
    df[i].value_counts(normalize=True).plot(kind="bar", figsize=(10, 4))
    plt.title(i)
    plt.show()

for i in categ_cols:
    plt.figure(figsize=(10, 4))
    sns.boxplot(x=i, y=target, data=df)
    plt.xticks(rotation=45)
    plt.title(i)
    plt.show()

# 这段代码是对分类变量（categ_cols 中的列）分别画箱线图（boxplot）。
# 箱线图显示不同类别对应的目标变量（trip_total）的分布情况，
# 可以帮助我们观察不同类别之间目标变量的差异和异常值。

df = df[df["trip_total"] < 3000].reset_index(drop=True)
# 这行代码过滤掉目标变量 trip_total 超过3000的极端大值，防止异常值影响分析，
# 并且重置索引。

df = df[df["payment_type"].isin(["Credit Card", "Cash"])].reset_index(drop=True)
# 只保留付款方式为“Credit Card”或“Cash”的数据，排除其他付款方式的样本。

df["payment_type"] = df["payment_type"].apply(
    lambda x: 0 if x == "Credit Card" else (1 if x == "Cash" else None)
)
# 将付款方式编码为数字：Credit Card编码为0，Cash编码为1，
# 方便机器学习模型使用。

df["trip_start_timestamp"] = pd.to_datetime(df["trip_start_timestamp"])
# 将trip_start_timestamp列转换为pandas的日期时间格式，方便日期相关操作。

df["dayofweek"] = df["trip_start_timestamp"].dt.dayofweek
df["hour"] = df["trip_start_timestamp"].dt.hour
# 提取日期中的星期几（0=周一，6=周日）和小时数，生成新的特征列。

_, ax = plt.subplots(1, 2, figsize=(10, 4))
df[["dayofweek", "trip_total"]].groupby("dayofweek").trip_total.sum().plot(
    kind="bar", ax=ax[0]
)
ax[0].set_title("Sum of trip_total")
df[["dayofweek", "trip_total"]].groupby("dayofweek").trip_total.mean().plot(
    kind="bar", ax=ax[1]
)
ax[1].set_title("Avg. of trip_total")
plt.show()
# 这部分代码绘制按星期几聚合的 trip_total 总和和平均值的柱状图，
# 用于观察不同星期几的业务量和平均消费情况。

_, ax = plt.subplots(1, 2, figsize=(10, 4))
df[["hour", "trip_total"]].groupby("hour").trip_total.sum().plot(kind="bar", ax=ax[0])
ax[0].set_title("Sum of trip_total")
df[["hour", "trip_total"]].groupby("hour").trip_total.mean().plot(kind="bar", ax=ax[1])
ax[1].set_title("Avg. of trip_total")
plt.show()
# 同样的，绘制按小时聚合的 trip_total 总和和平均值柱状图，
# 用来分析一天中不同时间段的业务量和平均消费。

df["dayofweek"] = df["dayofweek"].apply(lambda x: 0 if x in [5, 6] else 1)
# 将星期几特征二值化：周末（星期六、星期日）编码为0，工作日编码为1。

df["hour"] = df["hour"].apply(lambda x: 0 if x in [23, 0, 1, 2, 3, 4, 5, 6, 7] else 1)
# 将小时特征二值化：深夜和凌晨时间段（23点到7点）编码为0，白天编码为1。

df.describe().T
# 对整个数据框（DataFrame）做统计描述，显示各列的计数、均值、标准差、最小值、25%、50%、75%分位数和最大值，
# 帮助快速了解数据的整体分布和范围。

cols = [
    "trip_seconds",             # 行程时间（秒）
    "trip_miles",               # 行程距离（英里）
    "payment_type",             # 付款方式（编码后）
    "pickup_community_area",    # 上车社区区域编号
    "dropoff_community_area",   # 下车社区区域编号
    "dayofweek",                # 星期几（0=工作日，1=周末）
    "hour",                     # 小时段（0=夜间，1=白天）
    "trip_speed",               # 平均速度（英里/小时）
]
x = df[cols].copy()            # 选取这些特征列，作为模型输入
y = df[target].copy()          # 目标变量（出租车行程总价）

# 按75%训练集、25%测试集划分数据，random_state保证每次划分一致
X_train, X_test, y_train, y_test = train_test_split(
    x, y, train_size=0.75, test_size=0.25, random_state=13
)
X_train.shape, X_test.shape     # 查看训练集和测试集形状

reg = LinearRegression()        # 创建线性回归模型对象
reg.fit(X_train, y_train)       # 用训练集训练模型

y_train_pred = reg.predict(X_train)  # 训练集预测值
train_score = r2_score(y_train, y_train_pred)  # 训练集R²得分
train_rmse = root_mean_squared_error(y_train, y_train_pred)  # 训练集RMSE

y_test_pred = reg.predict(X_test)    # 测试集预测值
test_score = r2_score(y_test, y_test_pred)     # 测试集R²得分
test_rmse = root_mean_squared_error(y_test, y_test_pred)   # 测试集RMSE

print("Train R2-score:", train_score, "Train RMSE:", train_rmse)  # 输出训练集性能
print("Test R2-score:", test_score, "Test RMSE:", test_rmse)      # 输出测试集性能

coef_df = pd.DataFrame({"col": cols, "coeff": reg.coef_})   # 创建系数表，列名和对应系数
coef_df.set_index("col").plot(kind="bar")                  # 画出特征系数的柱状图

FILE_NAME = "model.pkl"         # 模型文件名
with open(FILE_NAME, "wb") as file:
    pickle.dump(reg, file)      # 将训练好的模型序列化保存为文件

BLOB_PATH = "taxicab_fare_prediction/"                     # 云存储路径前缀
BLOB_NAME = BLOB_PATH + FILE_NAME                           # 云存储完整文件路径

bucket = storage.Client(
    project=PROJECT_ID,
    credentials=credentials,
).bucket(BUCKET_URI[5:])            # 连接云存储桶（去除 "gs://"）
blob = bucket.blob(BLOB_NAME)                               # 创建存储对象
blob.upload_from_filename(FILE_NAME)                        # 上传本地模型文件到云存储


MODEL_DISPLAY_NAME = "[your-model-display-name]"  
# 定义模型显示名称，如果未设置，则用默认名称

if MODEL_DISPLAY_NAME == "[your-model-display-name]":
    MODEL_DISPLAY_NAME = "kade_taxi_fare_prediction_model"  
# 默认名称赋值

ARTIFACT_GCS_PATH = f"{BUCKET_URI}/{BLOB_PATH}"  
# 模型文件在 Google Cloud Storage 中的路径

exp_metadata = {"inputs": {"Input_feature": {}}, "outputs": {"Predicted_taxi_fare": {}}}  
# 解释器元数据，定义输入和输出的名称（可任意命名）

aiplatform.init(project=PROJECT_ID, location=LOCATION,credentials=credentials,)  
# 初始化 Vertex AI 客户端，指定项目和区域
model = aiplatform.Model.upload(
    display_name=MODEL_DISPLAY_NAME,                      # 模型显示名称
    artifact_uri=ARTIFACT_GCS_PATH,                        # 模型文件路径
    serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest",  # 使用的预测容器镜像（sklearn CPU版）
    explanation_metadata=exp_metadata,                     # 解释元数据,请求和响应参数
    explanation_parameters=ExplanationParameters(          # 解释参数，使用Sampled Shapley值解释
        sampled_shapley_attribution=SampledShapleyAttribution(path_count=25)
    ),
    upload_request_timeout=7200,  # 设置超时为1800秒（30分钟）
    # serving_container_deployment_timeout=7200,  # 设置容器部署超时为1800秒（30分钟）
    # serving_container_startup_probe_timeout_seconds=7200,  # 设置容器启动探针超时为1800秒（30分钟）
    # serving_container_health_probe_timeout_seconds=7200,  # 设置容器健康探针超时为1800秒（30分钟）
)
model.wait()  
# 等待模型上传完成

print(model.display_name)  
print(model.resource_name)  
# 打印模型显示名和资源名

ENDPOINT_DISPLAY_NAME = "[your-endpoint-display-name]"  
# 定义端点显示名称，未设置则用默认值

if ENDPOINT_DISPLAY_NAME == "[your-endpoint-display-name]":
    ENDPOINT_DISPLAY_NAME = "kade_taxi_fare_prediction_endpoint"  
# 赋默认端点名称

endpoint = aiplatform.Endpoint.create(
    display_name=ENDPOINT_DISPLAY_NAME, project=PROJECT_ID, location=LOCATION
)  
# 创建 Vertex AI 端点用于模型部署

print(endpoint.display_name)  
print(endpoint.resource_name)  
# 打印端点名称和资源名

DEPLOYED_MODEL_NAME = "[your-deployed-model-name]"  
# 定义部署模型的名称，未设置则用默认

if DEPLOYED_MODEL_NAME == "[your-deployed-model-name]":
    DEPLOYED_MODEL_NAME = "taxi_fare_prediction_deployment"  
# 赋默认部署名

MACHINE_TYPE = "n1-standard-2"  
# 设定部署模型的机器类型（虚拟机规格）

model.deploy(
    endpoint=endpoint,                                     # 部署到指定端点
    deployed_model_display_name=DEPLOYED_MODEL_NAME,      # 部署模型名称
    machine_type=MACHINE_TYPE,         
    deploy_request_timeout=7200,  # 设置超时为1800秒（30分钟）                    # 使用的机器类型
    traffic_split={"0": 100} 
)  
model.wait()  
# 等待模型部署完成

print(model.display_name)  
print(model.resource_name)  
# 再次打印模型名称和资源名

endpoint.list_models()  
# 列出端点上已部署的模型列表

test_json = {"instances": [X_test.iloc[0].tolist(), X_test.iloc[1].tolist()]}  
# 准备两个测试样本（测试集的前两条），转换成列表格式，作为请求payload

def plot_attributions(attrs):
    """
    接受特征归因值，绘制条形图表示各特征的重要性
    """
    rows = {"feature_name": [], "attribution": []}
    for i, val in enumerate(features):  
        rows["feature_name"].append(val)  
        rows["attribution"].append(attrs["Input_feature"][i])  
    attr_df = pd.DataFrame(rows).set_index("feature_name")  
    attr_df.plot(kind="bar")  
    plt.show()  
    return

features = X_train.columns.to_list()  
# 获取训练集的特征名称列表

def explain_tabular_sample(
    project: str, location: str, endpoint_id: str, instances: list  
):
    """
    调用Vertex AI端点，发送预测请求并获取特征归因解释结果，打印并绘制
    """
    aiplatform.init(project=project, location=location, credentials=credentials,)  
    # 初始化客户端
    endpoint = aiplatform.Endpoint(endpoint_id)
    response = endpoint.explain(instances=instances)  
    # 调用端点explain方法，传入测试样本，返回解释和预测结果

    print("#" * 10 + "Explanations" + "#" * 10)  
    for explanation in response.explanations:  
        print(" explanation")  
        attributions = explanation.attributions  # 取出归因信息

        for attribution in attributions:  
            print("  attribution")  
            print("   baseline_output_value:", attribution.baseline_output_value)  
            print("   instance_output_value:", attribution.instance_output_value)  
            print("   output_display_name:", attribution.output_display_name)  
            print("   approximation_error:", attribution.approximation_error)  
            print("   output_name:", attribution.output_name)  
            output_index = attribution.output_index  
            for output_index in output_index:  
                print("   output_index:", output_index)  

            plot_attributions(attribution.feature_attributions)  
            # 绘制该样本的特征归因条形图

    print("#" * 10 + "Predictions" + "#" * 10)  
    for prediction in response.predictions:  
        print(prediction)  
        # 打印模型预测结果

    return response

test_json = [X_test.iloc[0].tolist(), X_test.iloc[1].tolist()]  
# 准备测试样本列表格式

prediction = explain_tabular_sample(PROJECT_ID, LOCATION, endpoint.resource_name, test_json)  
# 调用解释函数，发送请求，获得并打印解释及预测

# ##########Explanations##########
#  explanation
#   attribution
#    baseline_output_value: 3.8885688650039567 基线预测值，比如模型在没有输入特征时的默认输出（类似“起点”）。
#    instance_output_value: 7.835568915449318  当前样本模型的预测值。
#    output_display_name: 模型输出的可读名称，这里是空的。
#    approximation_error: 1.0767001030476328e-18  归因计算的近似误差，越小越好，表明解释结果准确。
#    output_name: Predicted_taxi_fare   模型输出的名字，这里是 Predicted_taxi_fare，即出租车费用预测
#    output_index: -1
#  explanation
#   attribution
#    baseline_output_value: 3.8885688650039567
#    instance_output_value: 12.469287764089437
#    output_display_name: 
#    approximation_error: 3.860452983579596e-19
#    output_name: Predicted_taxi_fare
#    output_index: -1
# ##########Predictions##########
# 7.835568915449318
# 12.46928776408944