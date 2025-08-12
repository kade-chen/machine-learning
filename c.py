import pandas as pd
import matplotlib.pyplot as plt
from google.cloud import aiplatform
from google.cloud.aiplatform_v1.types import SampledShapleyAttribution
from google.cloud.aiplatform_v1.types.explanation import ExplanationParameters
from google.oauth2 import service_account

# 1. 初始化配置
PROJECT_ID = "genai-specialization-468108"  # 你的项目ID
LOCATION = "us-central1"                     # 你的区域
endpoint_id = "projects/601797833546/locations/us-central1/endpoints/1824875941294243840"  # 替换成你的endpoint完整资源名

service_account_key_path = "/Users/kade.chen/go-kade-project/github/mcenter/etc/kade-poc.json"  # 替换成你的服务账户JSON路径
credentials = service_account.Credentials.from_service_account_file(
    service_account_key_path,
    scopes=["https://www.googleapis.com/auth/cloud-platform"],
)

# 2. 定义特征名，用于画图
features = ["trip_seconds", "trip_miles", "payment_type", "pickup_community_area", "dropoff_community_area", "dayofweek", "hour", "trip_speed"]

def plot_attributions(attrs):
    """
    绘制特征归因条形图
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    
    rows = {"feature_name": [], "attribution": []}
    for i, val in enumerate(features):
        rows["feature_name"].append(val)
        rows["attribution"].append(attrs["Input_feature"][i])
    attr_df = pd.DataFrame(rows).set_index("feature_name")
    attr_df.plot(kind="bar")
    plt.show()

def explain_tabular_sample(project: str, location: str, endpoint_id: str, instances: list):
    """
    调用Vertex AI Endpoint做预测和解释
    """
    aiplatform.init(project=project, location=location, credentials=credentials)
    endpoint = aiplatform.Endpoint(endpoint_id)
    
    # # Undeploy the model
    # endpoint.undeploy_all()

    # # Delete the endpoint resource
    # endpoint.delete()
    # endpoint.wait()
    response = endpoint.explain(instances=instances)

    print("#" * 10 + "Explanations" + "#" * 10)
    for explanation in response.explanations:
        attributions = explanation.attributions
        for attribution in attributions:
            print("Baseline output value:", attribution.baseline_output_value)
            print("Instance output value:", attribution.instance_output_value)
            print("Output display name:", attribution.output_display_name)
            print("Approximation error:", attribution.approximation_error)
            plot_attributions(attribution.feature_attributions)

    print("#" * 10 + "Predictions" + "#" * 10)
    for prediction in response.predictions:
        print(prediction)
    return response

# 3. 准备测试样本，示例两条数据
test_samples = [
    [600, 2.5, 0, 8, 10, 1, 1, 30.0],
    [1200, 5.0, 1, 12, 15, 0, 0, 25.0]
]


# 4. 调用解释函数
explain_tabular_sample(PROJECT_ID, LOCATION, endpoint_id, test_samples)