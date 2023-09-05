---
title: "랜덤포레스트와 SHAP"
layout: single
# toc: true
categories: 
- "[개인 프로젝트] ML"
breadcrumbs: false
---

* 코로나-19로 대학생 중도탈락 크게 증가하여, 대학생의 중도탈락 예측 요인에 대한 연구가 필요함. 관련하여 기존 선행연구는 대부분 모수 기반의 통계 기법을 활용하였으나, 이와 같이 전통적 통계 기법은 많은 변인을 모형에 투입하는 다중공선성 발생 우려가 있고, 자유도 제약 등의 문제로부터 자유롭지 못하다는 한계가 있다. 이에 기계학습 기법을 활용해 대학생의 중도탈락 예측요인을 살펴보고자 한다.

* 교육학에서는 학습자에 대한 이해가 중요한 만큼 변수의 상대적 중요도에 대한 정보를 제공하는 랜덤포레스트 모형이 가장 빈번히 활용되고 있어, 본 연구에서도 랜덤포레스트를 사용하였다. 아울러, 종속 변인에 대한 기여도를 보다 안정적으로 도출하고자 가중 평균을 사용하는 SHAP 지수를 산출하였다.

* 타겟: 중도탈락(자퇴 의도) 
* 피처: '성별', '주관적계층의식', '코로나학번여부', '거주형태(자택 거주 여부)', '코로나스트레스', '사회적실재감', '인지적실재감', '교수실재감', '교수_학생 비대면 상호작용', '교수_학생 대면 상호작용', '학생_학생 비대면 상호작용', '학생_학생 대면 상호작용', '대학조직 커뮤니케이션에 대한 인식', '대학의 LMS지원에 대한 인식', '대학의 비대면 서비스 지원에 대한 인식', '전공계열_1(인문)', '전공계열_2(사회)', '전공계열_3(이공계)', '전공계열_4(학제간융합전공 등 기타)'

```python
# 데이터 전처리를 통해 최종 데이터셋 df 생성

# 변수별 분포 확인
plt.rc('font', family='NanumBarunGothic')

fig, ax = plt.subplots(ncols=4, nrows=6, figsize=(24, 36))
columns = df.columns.to_list()
count = 0

for row in range(6):
  for col in range(4):
    sns.kdeplot(data=df[columns[count]], ax=ax[row][col])
    ax[row][col].set_title(columns[count])
    count += 1
    if count == len(columns):
      break
```

```python
# 라이브러리 불러오기
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
```

```python
# 데이터를 피처, 타겟 분리 
features = df.drop(columns= ["전출", "자퇴", "대학몰입"], axis=1)
target = df["자퇴"]
```

```python
# 학습, 테스트 데이터셋 분리 
random_seed = 42
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.3, shuffle=True random_state=random_seed)
```

```python
# 학습 
classifier = RandomForestClassifier(random_state=random_seed)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
print(f"예측 정확도: {accuracy_score(y_test, y_pred)}")
```

```python
# 교차검증 예시1 
from sklearn.model_selection import StratifiedKFold

classifier = RandomForestClassifier(random_state=random_seed)
skfold = StratifiedKFold(n_splits=5)

n_iter = 0
cv_accuracy = []

for train_idx, test_idx in skfold.split(features, target):
  x_train, x_test = features.values[train_idx], features.values[test_idx]
  y_train, y_test = target.values[train_idx], target.values[test_idx]
  classifier.fit(x_train, y_train)
  pred = classifier.predict(x_test)

  n_iter += 1
  accuracy = np.round(accuracy_score(y_test, pred), 4)
  train_size = x_train.shape[0]
  test_size = x_test.shape[0]
  print(f"n_iter: {n_iter}교차 검증 정확도: {accuracy}, 학습데이터 크기: {train_size}, 검증데이터 크기: {test_size} ")
  print(f"검증 세트 인덱스: {test_idx}")
  cv_accuracy.append(accuracy)

print(f"교차 검증별 정확도: {np.round(cv_accuracy, 4)}")
print(f"평균 검증 정확도: {np.round(np.mean(cv_accuracy), 4)}")
```

```python
# 교차검증 예시2 
from sklearn.model_selection import cross_val_score, cross_validate

classifier = RandomForestClassifier(random_state=random_seed)
scores = cross_val_score(classifier, data.values, target.values, scoring="accuracy", cv=5)
print(f"교차 검증별 정확도: {np.round(scores, 4)}")
print(f"평균 검증 정확도: {np.round(np.mean(scores), 4)}")
```

```python
# 그리드 서치 
from sklearn.model_selection import GridSearchCV

grid = {
    'n_estimators' : [100,200],
    'max_depth' : [6,8,10,12],
    'min_samples_leaf' : [3,5,6],
    'min_samples_split' : [2,3,5],
    'max_features': [5, 6, 7]
}

classifier_grid = GridSearchCV(classifier, param_grid = grid, scoring="accuracy", n_jobs=-1, verbose =1)

classifier_grid.fit(x_train, y_train)

print("최고 평균 정확도 : {}".format(classifier_grid.best_score_))
print("최고의 파라미터 :", classifier_grid.best_params_)
```
    Fitting 5 folds for each of 216 candidates, totalling 1080 fits
    최고 평균 정확도 : 0.9196218487394958
    최고의 파라미터 : {'max_depth': 8, 'max_features': 7, 'min_samples_leaf': 3, 'min_samples_split': 2, 'n_estimators': 100}

```python
# 변수 중요도 
classifier_best_estimator = classifier_grid.best_estimator_
print(classifier_best_estimator.feature_importances_, '\n')

# 시리즈로 만들어 인덱스 붙이기
ser = pd.Series(classifier_best_estimator.feature_importances_, index=features.columns)

# 내림차순 정렬 
top = ser.sort_values(ascending=False)
print(top)
```

```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,6))
plt.title('Feature Importances Top 15')
sns.barplot(x=top, y=top.index)
plt.show()
```  
<p align="center"><img src="/assets/images/rf_feature_importance.png" title="feature importance"/></p>

```python
# SHAP 라이브러리 설치
!pip install shap
import shap
```

```python
explainer = shap.TreeExplainer(classifier_best_estimator)
shap_values = explainer.shap_values(x_test)
rf_resultX = pd.DataFrame(shap_values[1], columns = features.columns.to_list())
vals = np.abs(rf_resultX.values).mean(0)
shap_importance = pd.DataFrame(list(zip(data.columns, vals)), columns=['col_name', 'feature_importance_vals'])
shap_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
print(shap_importance)
```

```python
# 결과 시각화
shap.summary_plot(shap_values, features, class_names=["persistence", "neutral", "droupout"])
```  
<p align="center"><img src="/assets/images/shap.png" title="feature importance"/></p>