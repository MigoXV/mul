import pandas as pd

# 读取两个CSV文件
label_df = pd.read_csv('data-bin/celeba/label.csv')
identity_df = pd.read_csv('data-bin/celeba/identity.csv')

# 将identity_df的第二列与label_df合并，假设第一列（filename）是索引列
merged_df = label_df.merge(identity_df[['filename', 'identity']], on='filename', how='left')

# 将identity列插入到label的第一列和第二列之间
merged_df = merged_df[['filename', 'identity'] + label_df.columns[1:].tolist()]

# 将结果保存到新的CSV文件
merged_df.to_csv('data-bin/celeba/id_label.csv', index=False)
