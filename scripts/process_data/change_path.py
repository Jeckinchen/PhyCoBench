import pandas as pd

# 读取CSV文件
input_csv_path = 'phys101_fall_captions.csv'  # 请将路径替换为您的CSV文件路径
output_csv_path = 'phys101_fall_captions_v2.csv'  # 修改后的CSV文件保存路径

# 读取CSV文件
df = pd.read_csv(input_csv_path)

# 修改file_path列中的路径
df['file_path'] = df['file_path'].str.replace('/mnt/merchant/yongfan/data', '/data/oss_bucket_0/yongfan/data')

# 保存修改后的CSV文件
df.to_csv(output_csv_path, index=False)

print(f'文件已保存到 {output_csv_path}')
