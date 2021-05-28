## (0) Problem
# Definition of Input, Output : pandas data frame, Numeric
# Is it a classification? regression? classification
# How many number of labels? Bronze : 13295, Silver : 13116, Gold : 13372, NA = 231333
# How the data looks like? 271116 * 15

## (1) Feature
# 사용한 feature는 'Name', 'Sex', 'Age', 'Height', 'Weight', 'Team', 'NOC', 'Games', Year', 'Season', 'City', 'Sport', 'Event'이며
# 'Age', "Height", "Weight" 변수들은 na 값이 있어 각 값들의 평균값으로 대체 하였고
# 'Sex', 'Team', 'NOC', 'Games', 'Year', 'Season', 'City', 'Sport', 'Event' feature들에 Label인코딩을 실행하였다.


## (2) Model
# 랜덤 포레스트 모델을 설정한 이유 : label이 binary가 아닌 multi label이고 
# Overfitting이 잘되지 않고 학습 시간이 빠르기 때문에 랜덤포레스트를 설정하였다.
# 


## (3) Measure
# KFold 함수를 이용해 kf변수에 저장한다음 for문을 이용하여 train, test셋을 나누어 10번 실행했다.
# random_state를 사용해 변수를 고정했다.
# 평가지표로 F1 스코어를 사용했다.
# f1_array[]를 만들어 각 fold에서의 f1값을 저장하고 평균값을 구해 10-fold에서 나온 f1 score의 평균 값을 구했다.



## (4) Model parameter engineering
# label의 불균형이 심하여 다운 샘플링을 진행했다.
# grid search를 이용하여 최적의 하이퍼 파라메터를 구했다.
# n_estimators와 max_features는 모델 성능의 영향을 주는 파라미터이고 min_samples_split, min_samples_leaf는 모델의 과적합을 방지하기 위한 파라미터이다.
# 최적 하이퍼 파라메터 {max_depth = 12, min_samples_leaf = 2, min_samples_split = 8, n_estimators = 100, max_features = 0.75}

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
# from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample

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


# 'Name', 'Sex', 'Team', 'NOC', 'Games', 'Year', 'Season', 'City', 'Sport', 'Event' feature들에 label 인코딩을 실행
# Medal에 label 인코딩을 실행
features_le = ['Name', 'Sex', 'Team', 'NOC', 'Games', 'Year', 'Season', 'City', 'Sport', 'Event', 'Medal']

le_class = preprocessing.LabelEncoder()
for i in features_le:
    athlete[i] = le_class.fit_transform(athlete[i])



# 다운 샘플링
medal_na = athlete[athlete.Medal == 3]
medal_gold = athlete[athlete.Medal == 1]
medal_silver = athlete[athlete.Medal == 2]
medal_bronze = athlete[athlete.Medal == 0]
medal_na_downsampled = resample(medal_na, replace = False, n_samples = len(medal_silver), random_state = 0)


athlete = pd.concat([medal_na_downsampled, medal_silver, medal_gold, medal_bronze])

y = athlete['Medal']
athlete = athlete.drop(['Medal'], axis=1)


features = athlete.columns.values
athlete = pd.concat([athlete, y], axis=1)


# KFold 10교차 검증
kf = KFold(n_splits = 10, shuffle=True)

f1_array = []
fold_idx = 1

# # KFold 데이터 나누어서 검정하기
# for train_idx, test_idx in kf.split(athlete):
#     train_d, test_d = athlete.iloc[train_idx], athlete.iloc[test_idx]
    
#     train_y, train_x = train_d['Medal'], train_d[features]  
    
#     test_y, test_x = test_d['Medal'], test_d[features] 
    

#     model = RandomForestClassifier(random_state = 0, n_jobs = -1)
#     model.fit(train_x, train_y)
#     pred_y = model.predict(test_x)

#     f1 = round(f1_score(test_y, pred_y, average = "weighted"),2)  
#     f1_array.append(f1)
#     print('fold {}: F1 = {}'.format(fold_idx, f1))
#     fold_idx += 1


# print("Total (Average) F1 = {}".format(round(np.average(f1_array),2)))


# params = {'n_estimators' : [10,50,100],
#           'max_depth' : [6,8,10,12],
#           'min_samples_leaf' : [2, 8, 16, 20],
#           'min_samples_split' : [8, 12, 18],
#           'max_features' : [0.1, 0.5, 0.75, 1.0]
#           }
# rf = RandomForestClassifier(random_state = 0, n_jobs = -1)
# ## n_jobs = -1로 설정해 모든 프로세스를 사용하도록 고려 
# grid_cv = GridSearchCV(rf, param_grid = params, scoring = 'f1_weighted', cv = 3, n_jobs = -1)
# grid_cv.fit(train_x, train_y)

# print('최적 하이퍼 파라미터:', grid_cv.best_params_)
# print('최고 예측 정확도: {:.2f}'.format(grid_cv.best_score_))



## 최적의 하이퍼 파라메터를 이용해 k-fold 실행
for train_idx, test_idx in kf.split(athlete):
    train_d, test_d = athlete.iloc[train_idx], athlete.iloc[test_idx]
    
    train_y, train_x = train_d['Medal'], train_d[features]  # train셋 label, features
    
    test_y, test_x = test_d['Medal'], test_d[features] # test셋 label, features
    
    model = RandomForestClassifier(max_depth = 12, min_samples_leaf = 2, min_samples_split = 8, 
                                   n_estimators = 100, max_features = 0.75, random_state = 0, n_jobs = -1)
    
    model.fit(train_x, train_y)
    pred_y = model.predict(test_x)

    f1 = round(f1_score(test_y, pred_y, average = "weighted"),2)  # f1 score 구하기
    f1_array.append(f1)
    print('fold {}: F1 = {}'.format(fold_idx, f1))
    fold_idx += 1


print("Total (Average) F1 = {}".format(round(np.average(f1_array),2)))

