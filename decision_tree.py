import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# 读取CSV文件，设置分隔符为分号（;）
data = pd.read_csv('bank-full.csv', sep=';')

# 选择属性
selected_features = ['duration', 'poutcome']
X = data[selected_features]
y = data['y']

# 对离散型属性进行Label编码
label_encoder = LabelEncoder()
#X['housing'] = label_encoder.fit_transform(X['housing'])
#X['contact'] = label_encoder.fit_transform(X['contact'])
#X['month'] = label_encoder.fit_transform(X['month'])
X['poutcome'] = label_encoder.fit_transform(X['poutcome'])
#X['loan'] = label_encoder.fit_transform(X['loan'])

# 检查缺失值
missing_values = X.isnull().sum()
# 删除包含缺失值的记录
X = X.dropna()

# 重置索引
X = X.reset_index(drop=True)
y = y[X.index]

# 建立决策树模型
clf = DecisionTreeClassifier()


# 进行5折交叉验证，并打印每次的MCC结果
# cv_scores = cross_val_score(clf, X, y, cv=5, scoring=scoring)
# for i, score in enumerate(cv_scores):
#     print(f"Fold {i+1}: MCC = {score:.4f}")

# 打印平均MCC结果
# print(f"Average MCC: {cv_scores.mean():.4f}")

# 建立随机森林模型
#clf = RandomForestClassifier()

# 定义评估指标为MCC
scoring = make_scorer(matthews_corrcoef)

# 进行5折交叉验证，并打印每次的MCC结果
cv_scores = cross_val_score(clf, X, y, cv=5, scoring=scoring)
for i, score in enumerate(cv_scores):
    print(f"Fold {i+1}: MCC = {score:.4f}")

# 打印平均MCC结果
print(f"Average MCC: {cv_scores.mean():.4f}")