import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_stacked_bar(csv_file: str, features: str = ""):
    """
    读取 CSV 文件并绘制选定特征的分块条形图（堆叠条形图）。
    
    参数:
    - csv_file: 输入的 CSV 文件路径
    - features: 逗号分隔的特征名（为空则绘制所有特征）
    """
    # 读取 CSV 文件
    df = pd.read_csv(csv_file)

    # 移除非特征列（假设前两列为 'Filename' 和 'Identity'）
    feature_columns = df.columns[2:]

    # 解析需要绘制的特征列
    if features:
        selected_features = [f.strip() for f in features.split(",") if f.strip() in feature_columns]
    else:
        selected_features = feature_columns  # 为空时选择所有特征

    # 统计每个特征的 -1 和 1 的数量
    data = []
    for feature in selected_features:
        counts = df[feature].value_counts(normalize=True) * 100  # 计算百分比
        data.append({
            "Feature": feature,
            "Positive": counts.get(1, 0),
            "Negative": counts.get(-1, 0)
        })

    # 转换数据格式以适配 seaborn
    plot_df = pd.DataFrame(data).melt(id_vars=["Feature"], var_name="Category", value_name="Percentage")

    # 绘制堆叠条形图
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x="Feature", y="Percentage", hue="Category", data=plot_df, palette=["salmon", "steelblue"])

    # 美化图表
    plt.xlabel("Feature")
    plt.ylabel("Percentage (%)")
    plt.title("Stacked Bar Chart of Selected Features")
    plt.legend(title="Category", labels=["Negative (-1)", "Positive (1)"])
    plt.xticks(rotation=30, ha="right")  # 旋转 X 轴标签以防止重叠
    plt.ylim(0, 100)  # 设定 y 轴范围
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    # 显示图表
    plt.show()

if __name__ == "__main__":
    input_csv = "data-bin/celeba/id_label.csv"
    plot_stacked_bar(input_csv)
