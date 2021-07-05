# machine_learning


# Purpose
- 수업시간에 배운 내용으로 데이터 분석을 해보자! -> athlete



# (0) Problem
- Definition of Input, Output : pandas data frame, Numeric
- Is it a classification? regression? classification
- How many number of labels? Bronze : 13295, Silver : 13116, Gold : 13372, NA = 231333
- How the data looks like? 271116 * 15

# (1) Feature
- 사용한 feature는 'Name', 'Sex', 'Age', 'Height', 'Weight', 'Team', 'NOC', 'Games', Year', 'Season', 'City', 'Sport', 'Event'이며
- 'Age', "Height", "Weight" 변수들은 na 값이 있어 각 값들의 평균값으로 대체 하였고
- 'Sex', 'Team', 'NOC', 'Games', 'Year', 'Season', 'City', 'Sport', 'Event' feature들에 Label인코딩을 실행하였다.


# (2) Model
- 랜덤 포레스트 모델을 설정한 이유 : label이 binary가 아닌 multi label이고 
- Overfitting이 잘되지 않고 학습 시간이 빠르기 때문에 랜덤포레스트를 설정하였다.
 


# (3) Measure
- KFold 함수를 이용해 kf변수에 저장한다음 for문을 이용하여 train, test셋을 나누어 10번 실행했다.
- random_state를 사용해 변수를 고정했다.
- 평가지표로 F1 스코어를 사용했다.
- f1_array[]를 만들어 각 fold에서의 f1값을 저장하고 평균값을 구해 10-fold에서 나온 f1 score의 평균 값을 구했다.



# (4) Model parameter engineering
- label의 불균형이 심하여 다운 샘플링을 진행했다.
- grid search를 이용하여 최적의 하이퍼 파라메터를 구했다.
- n_estimators와 max_features는 모델 성능의 영향을 주는 파라미터이고 min_samples_split, min_samples_leaf는 모델의 과적합을 방지하기 위한 파라미터이다.
- 최적 하이퍼 파라메터 {max_depth = 12, min_samples_leaf = 2, min_samples_split = 8, n_estimators = 100, max_features = 0.75}
