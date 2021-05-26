## (0) Problem
# Definition of Input, Output : pandas data frame, Numeric
# Is it a classification? regression? classification
# How many number of labels? NA, Bronze, Silver, Gold
# How the data looks like? 271116 * 15

## (1) Feature
# 사용한 feature는 'Name', 'Sex', 'Age', 'Height', 'Weight', 'Team', 'NOC', 'Games', Year', 'Season', 'City', 'Sport', 'Event'이며
# 'Age', "Height", "Weight" 변수들은 na 값이 있어 각 값들의 평균값으로 대체 하였고
# 'Sex', 'Team', 'NOC', 'Games', 'Year', 'Season', 'City', 'Sport', 'Event' feature들과 'Medal' 레이블에 Label인코딩을 실행하였다.
# 각 feature들의 범위가 다르므로 Min-Max 정규화를 통해 모든 피쳐의 값을 0,1 범위로 조정했다.

## (2) Model
# 랜덤 포레스트 모델을 설정한 이유 : label이 binary가 아닌 multi label이고 Overfitting이 잘되지 않고 학습 시간이 
# 빠르기 때문에 랜덤포레스트를 설정하였다. 
import pandas as pd
import numpy as np

athlete = pd.read_csv("C:/machinelearning/athlete_events.csv")

athlete.info()

# 피쳐 이름 확인
athlete.columns.values
# label 개수  Bronze : 13295, Silver : 13116, Gold : 13372, NA = 231333
athlete['Medal'].value_counts()
athlete['Medal'].isnull().sum()


# na 값 확인  age, height, weight에 na값이 많이 있다.
athlete.isnull().sum()


# 각 팩터형 변수의 수준 확인
athlete['Name'].value_counts()
athlete['Team'].value_counts()
athlete['NOC'].value_counts()
athlete['Games'].value_counts()
athlete['Year'].value_counts()
athlete['Season'].value_counts()
athlete['City'].value_counts()
athlete['Sport'].value_counts()
athlete['Event'].value_counts()


## 전처리
# ID 변수 제거
athlete = athlete.drop(['ID'], axis = 1)

# 결측값을 각 값들의 평균값으로 대체
athlete['Age'].replace(to_replace = np.nan, value = athlete.Age.mean(), inplace = True)
athlete['Height'].replace(to_replace = np.nan, value = athlete.Height.mean(), inplace = True)
athlete['Weight'].replace(to_replace = np.nan, value = athlete.Weight.mean(), inplace = True)


# 'Sex', 'Team', 'NOC', 'Games', 'Year', 'Season', 'City', 'Sport', 'Event' feature들과 
# 'Medal' 레이블에 Label인코딩을 실행
# 각 feature들의 범위가 다르므로 Min-Max 정규화를 통해 모든 피쳐의 값을 0,1 범위로 조정
from sklearn import preprocessing

features_le = ['Name', 'Sex', 'Team', 'NOC', 'Games', 'Year', 'Season', 'City', 'Sport', 'Event', 'Medal']
le_class = preprocessing.LabelEncoder()
for i in features_le:
    athlete[i] = le_class.fit_transform(athlete[i])


# feature의 정규화를 위해 label을 따로 저장
y = athlete['Medal']
athlete = athlete.drop(['Medal'], axis=1)

# 모든 피처를 0, 1범위로 조정
min_max_scaler = preprocessing.MinMaxScaler()
athlete_scale = min_max_scaler.fit_transform(athlete)
athlete = pd.DataFrame(athlete_scale, columns=athlete.columns)



## (3)
# K-fold 교차 검증을 하기전에 pd.concat함수를 이용하여 label과 df을 다시 합친다.
# K-fold 교차검증을 하기 위해 kf모델을 만든다. 
# for문을 이용하여 train과 test셋을 나누고 모델에 학습하여 fold 별로 f1 score를 구한다음
# 마지막에 f1 score의 평균을 구한다.
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


features = athlete.columns.values
athlete = pd.concat([athlete, y], axis=1)


# KFold 10교차 검증
kf = KFold(n_splits = 10, shuffle=True)

f1_array = []
fold_idx = 1


# KFold 데이터 나누기
for train_idx, test_idx in kf.split(athlete):
    train_d, test_d = athlete.iloc[train_idx], athlete.iloc[test_idx]
    
    train_y = train_d['Medal']  # train셋 label
    train_x = train_d[features] # train셋 features
    
    test_y = test_d['Medal'] # test셋 label
    test_x = test_d[features] # test셋 features
    
    model = RandomForestClassifier()
    
    model.fit(train_x, train_y)
    pred_y = model.predict(test_x)

    f1 = round(f1_score(test_y, pred_y, average = "weighted"),2)  # f1 score 구하기
    f1_array.append(f1)
    print('fold {}: F1 = {}'.format(fold_idx, f1))
    fold_idx += 1


print("Total (Average) F1 = {}".format(round(np.average(f1_array),2)))


## (4) 
# Model parameter engineering
from sklearn.model_selection import GridSearchCV

params = {'n_estimators' : [10,100],
          'max_depth' : [6,8,10,12],
          'min_samples_leaf' : [8, 12, 18],
          'min_samples_split' : [8, 16, 20]
          }


rf = RandomForestClassifier()
grid_cv = GridSearchCV(rf, param_grid = params, cv = 3, scoring= f1_score, n_jobs = -1)
grid_cv.fit(train_x, train_y)

print(gird_cv.best_score_)
print(grid_cv.best_params_)
