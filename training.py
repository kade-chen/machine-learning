import datapreprocess as datep
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error
import pandas as pd
# load the required libraries start
import pickle

def train_model():
    
    df = datep.PreprocessData()

    target = "trip_total"

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

if __name__ == "__main__":
    # CreateImportDataset(bqclient,os.getenv("PROJECT_ID_Genai"), os.getenv("DATASET_ID"), os.getenv("TABLE_ID"))
    train_model() 