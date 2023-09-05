---
title: "비대면 교육환경에서 대학생의 대학 몰입 예측요인 탐색을 위한 다중회귀분석 및 상호작용효과 분석"
layout: single
toc: true
categories: 
  - "[논문] Moderated multiple regression"
breadcrumbs: false
---
  
* This contains a summary of a journal paper which is under review at *Korean Journal of Educational Research* (September 1, 2023). Python code for data anlysis is also presented. 

### Abstract 

**1. Objective**
  * This study aims to explore the factors predicting college students' institutional commitment in the online learning environment, focusing on educational presence(social presence, cognitive presence, teaching presence). In addition, this study investigated the moderating effects of ‘being students in the COVID-19 cohort’, who started their college life during the pandemic, on the relationship between educational presence and institutional commitment.  

**2. Theoretical Frameworks**
  * This study adopted Tinto(1975, 1993)'s interactionalist theory of college student departure and the concept of educational presence(Garrison et al., 2000) as theoretical frameworks.  

**3. Methods**  
  * Data were collected in October 2021 via online survey with 864 students in A university located in Seoul, Korea. 
  * For data analysis, multiple regression analysis and analysis of moderating effects were conducted using Python 3.10.12. 

**4. Results** 
  * The main findings of this study are as follows: cognitive presence, teaching presence, subjective social status, the non-face-to-face interactions between students, the awareness of organizational communication in university, awareness of support for the online learning management system(LMS), and awareness of support for online services and programs were found to predict institutional commitment. 
  * However, there were no moderating effects of ‘being students in the COVID-19 cohort’ on the relationship between educational presence and institutional commitment.  
  <p align="center"><img src="/assets/images/ic_graph_moderation.png" title="moderation graph"/></p>

**5. Conclusion & Implications** 
  *  Accordingly, this study suggested practical implications to improve institutional commitment in the online learning environment, including support for cognitive presence and teaching presence.

### Code 

```python
# 라이브러리 설치 및 불러오기 
import pandas as pd
from sklearn import datasets
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

!pip install linearmodels
import linearmodels.iv.model as lm
```

```python
# 한글 폰트 관련 
!sudo apt-get install -y fonts-nanum
!sudo fc-cache -fv
!rm ~/.cache/matplotlib -rf
```

```python
# 경고메시지 무시하기 
import warnings
warnings.filterwarnings('ignore')
```

```python
# 데이터 불러오기 
df = pd.read_csv("dataset.csv")
df.info()
```

```python
# 변수별 분포 확인
plt.rc('font', family='NanumBarunGothic')

fig, ax = plt.subplots(ncols=3, nrows=6, figsize=(15, 30))
columns = df.columns.to_list()
count = 0

for row in range(6):
  for col in range(3):
    sns.kdeplot(data=df[columns[count]], ax=ax[row][col])
    ax[row][col].set_title(columns[count])
    count += 1
    if count == len(columns):
      break
```

```python
# skewness 확인 -> 문제 없음
columns = df.columns.to_list()
for column in columns:
  if abs(df[column].skew()) > 2 :
     print(f"{column} skew: {df[column].skew()}")
```

```python
# 더미변수 생성
df_dummy = pd.get_dummies(data = df, columns = ["전공계열"], drop_first = False)
```

```python
# 다중회귀분석
lr = smf.ols(f'대학몰입1 ~ 성별더미 + 주관적계층의식 + 코로나학번더미 + 전공계열_2 + 전공계열_3+ 전공계열_4 + 자택더미 + 코로나스트레스 + 사회적실재감 + 인지적실재감 + 교수실재감 + 교수_학생비대면 + 교수_학생대면 + 학생_학생비대면 + 학생_학생대면 + 조직커뮤니케이션 + LMS지원 + 비대면서비스지원', data=df_dummy).fit()
lr.summary()
```

```python
# 조절회귀분석 (상호작용효과)
for iv in ["사회적실재감", "인지적실재감", "교수실재감"]:
  mo_lr = smf.ols(f'대학몰입1 ~ 성별더미 + 주관적계층의식 + 코로나학번더미 + 전공계열_2 + 전공계열_3+ 전공계열_4 + 자택더미 + 코로나스트레스 + 사회적실재감 + 인지적실재감 + 교수실재감 + 교수_학생비대면 + 교수_학생대면 + 학생_학생비대면 + 학생_학생대면 + 조직커뮤니케이션 + LMS지원 + 비대면서비스지원 + {iv}*코로나학번더미', data=df_dummy).fit()
  print(mo_lr.summary())
```

