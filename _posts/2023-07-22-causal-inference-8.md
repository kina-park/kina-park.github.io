---
title: "[3-2] 준실험 분석방법론"
layout: single
# toc: true
categories: 
- "[관심분야 공부] Causal inference"
breadcrumbs: false
---

### 1. Quasi-Experiment method
* 비교 가능한 control group을 찾는 것이 앞서 살펴본 Quasi-Experiment design이었다면, 이렇게 찾은 control group을 활용해서 어떻게 counterfactual을 유추할 수 있을 지에 대한 방법이 Quasi-Experiment method. 디자인이 먼저고, 그 디자인을 활용해서 method를 적용할 수 있음. 즉, Quasi-Experiment method는 비교 가능한 control group을 통해서 우리가 원하는 counterfactual을 approximate하는 것이라고 볼 수 있음
<p><img src="/assets/images/atet.png" title="ATET"/></p>
그림 출처: 인과추론의 데이터과학. (2022, June 17)

* 만약, treatment가 없었다면 있었을 잠재적 결과인 counterfactual을 추론하는 게 핵심. 그래서 대부분의 인과추론은 위 그림의 표에서 treatment 그룹 내 에서의 효과인 ATET임. 만약, 전체 샘플로 확장하기 위해서 ATE를 구하고자 한다면, 처치 그룹과 통제 그룹이 완전히 비교 가능해서 서로 역할을 바꿔도 결과가 동일할 것이다 라는 좀 더 강한 추가적인 가정이 있어야, ATET가 ATEU를 포함하는 ATE로 확장할 수 있음

* Causal Inference 관점에서 데이터 구조의 이해를 재해석해 보자. 만약, counterfactual을 시간 관점에서 구분해 보면, 데이터 구조를 이해하고, 여러가지 방법론을 이해하는 것이 수월함. treatment가 없었다면 있었을 counterfactual에서 시간에 따라 변하지 않는 counterfactual과 시간에 따라 변하는 counterfactual로 구분할 수 있음. 만약, 시간에 따라 변하지 않는 counterfactual이 있었다면, treatment가 실제로 없었을 때의 그 과거의 값에 거의 동일함  
<p><img src="/assets/images/counterfactual_time.png" title="time_invariant/varying"/></p>
그림 출처: 인과추론의 데이터과학. (2022, June 17)

### 2. Data structure based on causal inference 
* Causal inference 관점에서 데이터 구조를 이해하는 첫 번째 포인트
    * treatment 전후의 데이터를 관찰할 수 있는 longitudinal 데이터가 어떤 treatment를 받은 특정 시점의 cross sectional 데이터보다 인과추론 관점에서는 훨씬 더 유리한 측면이 있음
* Causal inference 관점에서 데이터 구조를 이해하는 두 번째 포인트
    * treatment 전후의 데이터를 모두 활용할 때에도, treatment 그룹 내에서의 전후 데이터만 있는 경우도 있고, 전후의 control 집단 데이터까지 있는 경우도 있음. 전자를 time series data하고, 후자는 패널 데이터라고 함. 따라서, 인과추론 관점에서는 treatment 집단의 전후만 있는 time series data 보다 control 집단의 전후도 있는 패널 데이터가 훨씬 유리하다고 볼 수 있음

### 3. Research design & Data structure  
<p><img src="/assets/images/map.png" title="Research design & Data structure"/></p>
그림 출처: 인과추론의 데이터과학. (2022, June 17)

### 4. Difference-in-Differences (DID) 
* Treatment가 없을 때 시간에 따라 변했던 counterfactual은 control group에서의 시간에 따라 변하는 정도만 가지고 추정
1. Parallel trend assumption
    * control group과 treatment group이 모든 면에서 다 비교 가능(비슷)한 것을 요구하지 않음. DID가 성립하기 위해서는 control group과 treatment group이 시간에 따라 변하는 정도만 비교 가능하면 됨. 따라서, 기존의 가정에 비해 느슨한 가정

2. 아래 그림에서 보면, 만약 treatment 집단 내에서 treatment가 없었다면(counterfactual) 통제 그룹의 평균 추세와 마찬가지로 0.5씩 증가했을 것이라 가정

3. DID를 통해서 구할 수 있는 것은 ATET. 시간에 따라 변하는 추세만 비슷하면 됨. 만약, ATE로 확장하기 위해서는 Parallel trend assumption 보다 더 강한 treatment group과 control group이 모든 면에서 비교 가능하다(두 그룹의 역할을 그대로 스위치 했어도 동일하다)는 가정이 충족되어야 함

4. 구한 것이 ATET 인지, ATE인지 명확히 구분하는 것이 중요함. 연구에 따라서는 ATET만으로 충분한 research context도 있음

5. Parallel trend assumption이 가능하지 않다면, 매칭을 활용할 수 있음

### 5. Synthetic Control
* 만약, 매칭 또한 불가능하다면 Synthetic Control. Synthetic Control은 control 그룹의 조합을 통해서 treatment 그룹의 counterfactual을 예측하고자 하는 것. 장점 중 하나는 Parallel trend assumption을 만족하지 않더라도, control unit을 잘 조합하면 treatment unit을 잘 예측할 수 있음 

### 6. 여러 상황 가정하기 
* 'DID + Matching’이 잘 결합되기도 함: 다양한 control 집단 중에서 그나마 Parallel trend assumption이 성립되는 집단을 매칭해서, 그 집단만을 대상으로 분석을 하는 DID. 즉, 여기서 매칭은 Parallel trend assumption을 위한 보조적인 수단  
<p><img src="/assets/images/did_matching.png" title="DID + Matching"/></p>
그림 출처: 인과추론의 데이터과학. (2022, June 17)

* 그럼에도 불구하고 사실상 Firm2도 Parallel trend assumption를 만족하지 않음. 그런데, 만약 Firm2와 Firm3를 조합해서 가상의 통제 집단을 만들었더니, counterfactual을 잘 구성할 수 있는 상황이라면..?  
<p><img src="/assets/images/pararell_trend_assumption.png" title="DID + Matching"/></p>
그림 출처: 인과추론의 데이터과학. (2022, June 17)

* control 그룹은 없고, treatment 그룹의 전후만 있는 데이터 구조라면?  
    * Interrupted Time-Series Analysis / time-series forecasting 

### References 
인과추론의 데이터과학. (2022, June 17). [Bootcamp 3-2] 준실험 분석방법론 [Video]. YouTube. [https://www.youtube.com/watch?v=aPv_xLzBw1w&list=PLKKkeayRo4PV_6-nbBgmUNOSpG1OO49M3&index=9][1]

[1]: https://www.youtube.com/watch?v=aPv_xLzBw1w&list=PLKKkeayRo4PV_6-nbBgmUNOSpG1OO49M3&index=9