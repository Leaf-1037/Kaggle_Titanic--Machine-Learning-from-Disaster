import pandas as pd #数据分析
import numpy as np #科学计算
from pandas import Series,DataFrame

train = pd.read_csv("./train.csv")
print(train.info())

import matplotlib.pyplot as plt


# Training

data_train=train
from sklearn.ensemble import RandomForestRegressor


### 使用 RandomForestClassifier 填补缺失的年龄属性
def set_missing_ages(df):
    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].values
    unknown_age = age_df[age_df.Age.isnull()].values

    # y即目标年龄
    y = known_age[:, 0]

    # X即特征属性值
    X = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1::])

    # 用得到的预测结果填补原缺失数据
    df.loc[(df.Age.isnull()), 'Age'] = predictedAges

    return df, rfr


def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()), 'Cabin'] = "Yes"
    df.loc[(df.Cabin.isnull()), 'Cabin'] = "No"
    return df

# 填充年龄数据
nn={'Capt':'Rareman', 'Col':'Rareman','Don':'Rareman','Dona':'Rarewoman',
    'Dr':'Rareman','Jonkheer':'Rareman','Lady':'Rarewoman','Major':'Rareman',
    'Master':'Master','Miss':'Miss','Mlle':'Rarewoman','Mme':'Rarewoman',
    'Mr':'Mr','Mrs':'Mrs','Ms':'Rarewoman','Rev':'Mr','Sir':'Rareman',
    'the Countess':'Rarewoman'}

data_train['Title']=data_train['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
data_train['Child']=data_train['Age'].apply(lambda x:1 if x <= 12 else 0)
data_train['Mother']=data_train['Title'].apply(lambda x:1 if x == 'Mrs' else 0)
data_train['MixedStatus']=data_train['Sex'].apply(lambda x:1 if x == 'female' else 0) + data_train['Pclass']
data_train['FamilySize'] = data_train['SibSp']+data_train['Parch']
data_train.Age.groupby([data_train.Title,data_train.Sex]).describe()
data_train.Title = data_train.Title.map(nn)
Tit=['Mr','Miss','Mrs','Master','Girl','Rareman','Rarewoman']
for i in Tit:
    data_train.loc[(data_train.Age==999)&(data_train.Title==i),'Age']=data_train.loc[data_train.Title==i,'Age'].median()

data_train, rfr = set_missing_ages(data_train)
data_train = set_Cabin_type(data_train)




print(data_train)

dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')
df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
print(df)



import sklearn.preprocessing as preprocessing
scaler = preprocessing.StandardScaler()
age_scale_param = scaler.fit(df['Age'].values.reshape(-1, 1))
df['Age_scaled'] = scaler.fit_transform(df['Age'].values.reshape(-1, 1), age_scale_param)
fare_scale_param = scaler.fit(df['Fare'].values.reshape(-1, 1))
df['Fare_scaled'] = scaler.fit_transform(df['Fare'].values.reshape(-1, 1), fare_scale_param)
print()
print("scaled train data is like...")
print(df)





from sklearn import linear_model

# 用正则取出我们要的属性值
train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Sex_.*|Pclass_.*|Child|Mother|MixedStatus|FamilySize')
train_np = train_df.values

# y即Survival结果
y = train_np[:, 0]

# X即特征属性值
X = train_np[:, 1:]

print("train_data is like...")
print(X)


# fit到LogisticRegression之中
clf = linear_model.LogisticRegression(C=1.0, penalty='l2', tol=1e-6, solver='liblinear')
clf.fit(X, y)

clf

# Model 2
from sklearn.ensemble import BaggingRegressor
bagging_clf = BaggingRegressor(clf,n_estimators=20,max_samples=0.8,max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=-1)
bagging_clf.fit(X,y)

# Model 3
from sklearn.svm import SVC
svc = SVC(C=0.71,kernel='rbf',max_iter=20000)
svc.fit(X,y)

# Model rfr
from sklearn.ensemble import RandomForestClassifier  # 随机森林模型分类器
rnd_clf = RandomForestClassifier(random_state=1
                            ,n_estimators=10 # 要构建的多少颗决策树
                            ,min_samples_leaf=1
                            ,min_samples_split=2
                            )

rnd_clf.fit(X,y)


# Predicting

# Pre-processing
data_test = pd.read_csv("./test.csv")
data_test.loc[ (data_test.Fare.isnull()), 'Fare' ] = 0
# 接着我们对test_data做和train_data中一致的特征变换
# 首先用同样的RandomForestRegressor模型填上丢失的年龄
tmp_df = data_test[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
null_age = tmp_df[data_test.Age.isnull()].values
# 根据特征属性X预测年龄并补上
X = null_age[:, 1:]
predictedAges = rfr.predict(X)
data_test.loc[ (data_test.Age.isnull()), 'Age' ] = predictedAges

data_test['Title']=data_test['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
data_test['Child']=data_test['Age'].apply(lambda x:1 if x <= 12 else 0)
data_test['Mother']=data_test['Title'].apply(lambda x:1 if x == 'Mrs' else 0)
data_test['MixedStatus']=data_test['Sex'].apply(lambda x:1 if x == 'female' else 0) + data_test['Pclass']
data_test['FamilySize'] = data_test['SibSp']+data_test['Parch']

data_test = set_Cabin_type(data_test)
dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix= 'Pclass')

df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'].values.reshape(-1, 1), age_scale_param)
df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'].values.reshape(-1, 1), fare_scale_param)
df_test


# prediction

test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
test_df = df_test.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Sex_.*|Pclass_.*|Child|Mother|MixedStatus|FamilySize')
test_X = test_df.values
print("test_data is like...")
print(test_X)
predictions = clf.predict(test_X)
predictions_bagging = bagging_clf.predict(test_X)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].values, 'Survived':predictions.astype(np.int32)})
result_bagging = pd.DataFrame({'PassengerId':data_test['PassengerId'].values, 'Survived':predictions_bagging.astype(np.int32)})
result.to_csv("./logistic_regression_predictions.csv", index=False)
result_bagging.to_csv("./bagging_regression_predictions.csv", index=False)

predictions_svm = svc.predict(test_X)
result_svm = pd.DataFrame({'PassengerId':data_test['PassengerId'].values, 'Survived':predictions_svm.astype(np.int32)})
result.to_csv("./svm_predictions.csv", index=False)

predictions_rf = rnd_clf.predict(test_X)
result_rf = pd.DataFrame({'PassengerId':data_test['PassengerId'].values, 'Survived':predictions_rf.astype(np.int32)})
result.to_csv("./rf_pred.csv", index=False)

# Result Analysis

