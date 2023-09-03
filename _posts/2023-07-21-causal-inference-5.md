---
title: "[2-3] 인과추론 관점에서의 회귀분석"
layout: single
toc: true
categories: 
- Causal inference
---

### 1. 어떤 종류의 Selection bias를 다룰 수 있을까?  
<p><img src="/assets/images/(un)observable.png" title="Selection on observables, unobservables"/></p>
그림 출처: 인과추론의 데이터과학. (2022, June 15)

* Selection on Unobservables Strategies
    * Randomized Controlled Trial, Quasi-Experiment, Instrumetal Variable
    * Random assignment, 적절한 research design을 통해 관찰 가능하지 않은 교란 요인들에 의한 selection biaS 문제까지 해결하고자 하는 전략. 아래 전략보다 좀 더 powerful 하다는 특징. 
* Selection on Observables Strategies
    * Designed Regression / Matching 
    * 관찰 가능한 변수들에 의해서만 처치 집단과 통제 집단이 선택된다는 가정 하에, selection bias를 모두 설명하고자 하는 전략 

### 2. 어떻게 관찰 가능한 변수들에 의해서만 통제, 처치 집단의 균형을 맞출 수 있을까? 
1. Regression adjustment  
    * 통제 변수의 활용을 통해서 selection bias를 설명하고자 함
2. Matching 
    * 두 집단이 서로 비교 가능할 수 있도록, 관찰 가능한 변수들의 값이 서로 유사한 데이터들끼리 매칭
3. Weighting 
    * 처치를 받을 확률의 역수 만큼을 각 데이터에 가중치를 부여함으로써 결과적으로 random assignment와 비슷하게 처치를 받을 확률이 같아지도록 만드는 방법

### 3. 인과추론 관점에서의 회귀분석
* 인과추론 관점에서 regression의 진정한 역할은 R-square의 높은 값을 강조하는 기존 관점과 달리 selection bias를 야기하는 confounding factor를 통제하는 데 있다. 그 중심에는 control variable 통제변수가 있음. 
* 즉, 인과추론이 목적이라면, control variable이 selection bias를 잘 통제하는 것이 중요하다. 

* 아래 식에서, selection bias가 0이 되어야 causal effect를 추정할 수 있음. 만약 우리가 selection bias를 모두 설명할 수 있는 변수를 알고 있다면, selection bias를 설명할 수 있는 control variable과 selection bias 간의 관계에 대한 functional form에 대한 가정이 필요함. 즉, selection bias가 control variable의 변화에 linear하게 비례할 수 있다고 가정함. 

<p><img src="/assets/images/regression.png" title="regression"/></p>
그림 출처: 인과추론의 데이터과학. (2022, June 15)

### 4. Control variable로 selection bias를 제거하기 위한 두 가지 가정
1. Selection bias를 모두 설명할 수 있는 control variable을 알고 있어야 한다. 
2. 그 관계가 어떠한 functional form을 가져야 한다. 
    * Regression을 통해서도 인과관계를 분석할 수 있게 됨. 흔히, causal inference가 가능한 assumption을 identity assumption이라고 부름. 
    * 따라서, 회귀분석에서의 인과추론 관점에서 identity assumption은 conditional independence라고 볼 수 있음. 
    * Control variable이 conditioning된 상태에서, 원인변수인 X여부에 상관없이 error term인 E의 평균값이 동일해야 함. 즉, Control variable이 conditioning된 상태에서 원인변수인 X와 error term 간의 상관관계가 없어야 함. 
    * 일반적인 회귀분석과 달리 중요한 건 R2이 아님. X가 0일 때, 1일 때의 차이(=selection bias)를 control variable이 얼마나 잘 설명하느냐 그 여부가 causal inference의 가능 여부를 결정함. 

### 5. 최종 정리
1. Regression 식의 우항(오른쪽 항)에 있는 모든 독립변수들이 동일한 역할을 하는 것이 아님. 인관관계를 분석하고자 하는 원인 변수와 나머지 통제 변수의 역할을 명확히 구분해야 함. 구분하는 목적은 통제변수의 역할을 판단하기 위해서임
2. 통제변수의 역할은 selection bias를 얼마나 잘 설명하는가에 있음 
3. 통제변수에 대해서는 인과적인 효과로 해석하지 않도록 주의해야 함. 


### References
인과추론의 데이터과학. (2022, June 15). [Bootcamp 2-3] 인과추론 관점에서의 회귀분석 [Video]. YouTube. [https://www.youtube.com/watch?v=6zQlPFdPBaI&list=PLKKkeayRo4PV_6-nbBgmUNOSpG1OO49M3&index=6][1]

[1]: https://www.youtube.com/watch?v=6zQlPFdPBaI&list=PLKKkeayRo4PV_6-nbBgmUNOSpG1OO49M3&index=6