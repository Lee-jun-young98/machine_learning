# Definition of Input, Output : Categorical data type, Numeric
# Is it a classification? regression? classification
# How many number of labels? NA, Bronze, Silver, Gold
# How the data looks like? 271116 * 15

# 모든 피처를 사용했으며 label은 Medal이다.
# 

import pandas as pd
import numpy as np

athlete = pd.read_csv("C:/machinelearning/athlete_events.csv")

athlete.info()

# label 개수  Bronze : 13295, Silver : 13116, Gold : 13372, NA = 231333
athlete['Medal'].value_counts()
athlete['Medal'].isnull().sum()

# 피쳐 이름
features = athlete.iloc[:,1:-1].columns.values

# na 값 확인  age, height, weight에 na값이 많이 있다.
athlete.isnull().sum()


# 각 팩터형 변수 확인
athlete['Name'].value_counts()
athlete['Team'].value_counts()
athlete['NOC'].value_counts()
athlete['Games'].value_counts()
athlete['Year'].value_counts()
athlete['Season'].value_counts()
athlete['City'].value_counts()
athlete['Event'].value_counts()



# 결측값을 각 값들의 평균값으로 대체하였다.
athlete["Age"].replace(to_replace = np.nan, value = athlete.Age.mean(), inplace = True)
athlete["Height"].replace(to_replace = np.nan, value = athlete.Height.mean(), inplace = True)
athlete["Weight"].replace(to_replace = np.nan, value = athlete.Weight.mean(), inplace = True)

# ID feature 제거
athlete = athlete.drop(["ID"], axis = 1)

from sklearn import preprocessing

# Season 변수와 Sex 변수, Medal 레이블을에 LabelEncoder를 진행 
le_class = preprocessing.LabelEncoder()
athlete['Medal'] = le_class.fit_transform(athlete['Medal'])
athlete['Season'] = le_class.fit_transform(athlete['Season'])
athlete['Sex'] = le_class.fit_transform(athlete['Sex'])


pd.get_dummies(athlete)

from sklearn.model_selection import KFold

features = ['{}'.format(x) for x in range(0,137111)]
kf = KFold(n_splits = 10, shuffle=True)

accrs = []
fold_idx = 1

for train_idx, test_idx in kf.split(x):
    print('fold {}'.format(fold_idx))
    train_d, test_d = x.iloc[train_idx], y.iloc[test_idx]
    
    
    