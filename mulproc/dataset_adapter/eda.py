import h5py
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm


def plot_stacked_bar_from_h5(h5_file: str, features: str = ""):
    """
    读取 HDF5 文件并绘制选定特征的分块条形图（堆叠条形图）。

    参数:
    - h5_file: 输入的 HDF5 文件路径
    - features: 逗号分隔的特征名（为空则绘制所有特征）
    """
    # 读取 HDF5 文件
    with h5py.File(h5_file, "r") as f:
        # 获取所有数据集的名称
        all_datasets = list(f.keys())
        # 排除 'images' 和 'identity' 数据集，保留其余的数据集作为标签特征
        feature_columns = [
            name for name in all_datasets if name not in ["image", "identity"]
        ]
        # 解析需要绘制的特征列
        if features:
            selected_features = [
                f.strip() for f in features.split(",") if f.strip() in feature_columns
            ]
        else:
            selected_features = feature_columns  # 为空时选择所有特征
        # 统计每个特征的正负样本数量
        data = []
        for feature in tqdm(selected_features, desc="Processing", unit="feature"):
            # 获取当前特征的数据
            feature_data = f[feature][:]
            num_pos = feature_data.sum() / len(feature_data) * 100
            num_neg = 100 - num_pos
            data.append(
                {
                    "Feature": feature,
                    "Positive": num_pos,
                    "Negative": num_neg,
                }
            )
        # 转换数据格式以适配 seaborn
        plot_df = pd.DataFrame(data).melt(
            id_vars=["Feature"], var_name="Category", value_name="Percentage"
        )
    # 绘制堆叠条形图
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        x="Feature",
        y="Percentage",
        hue="Category",
        data=plot_df,
        palette=["salmon", "steelblue"],
    )
    # 美化图表
    plt.xlabel("Feature")
    plt.ylabel("Percentage (%)")
    plt.title("Stacked Bar Chart of Selected Features")
    plt.legend(title="Category", labels=["Negative (-1)", "Positive (1)"])
    plt.xticks(rotation=30, ha="right")  # 旋转 X 轴标签以防止重叠
    plt.ylim(0, 100)  # 设定 y 轴范围
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    # 在 y=50 处绘制淡红色的虚线并添加标签
    plt.axhline(y=50, color="red", linestyle="--", linewidth=1)  # 更粗的淡红色虚线
    plt.text(
        x=len(selected_features),
        y=50,
        s="50%",
        color="red",
        va="center",
        ha="left",
        fontsize=12,
    )

    # 显示图表
    plt.show()


if __name__ == "__main__":
    input_h5 = "data-bin/celeba/split/train.h5"  # 请替换为你的 H5 文件路径
    plot_stacked_bar_from_h5(input_h5)
