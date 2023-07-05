import pandas as pd

# 读取CSV文件，设置分隔符为分号（;）
data = pd.read_csv('bank-full.csv', sep=';')

# 将离散型属性转换为数值型
data['job'] = pd.factorize(data['job'])[0]
data['marital'] = pd.factorize(data['marital'])[0]
data['education'] = pd.factorize(data['education'])[0]
data['default'] = pd.factorize(data['default'])[0]
data['housing'] = pd.factorize(data['housing'])[0]
data['loan'] = pd.factorize(data['loan'])[0]
data['contact'] = pd.factorize(data['contact'])[0]
data['month'] = pd.factorize(data['month'])[0]
data['poutcome'] = pd.factorize(data['poutcome'])[0]
data['y'] = pd.factorize(data['y'])[0]

# 计算属性的相关性
correlation_matrix = data.corr()

# 保存文件
correlation_matrix.to_excel('/Users/weihongbi/Desktop/correlation_matrix.xlsx', index=False)

# 打印相关性矩阵
print(correlation_matrix)




