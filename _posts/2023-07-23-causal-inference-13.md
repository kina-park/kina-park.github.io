---
title: "[4-3] 회귀 불연속"
layout: single
toc: true
categories: 
- "[관심분야 공부] Causal inference"
breadcrumbs: false
---

### 1. Regression Discontinuity(RD)
* 언제 활용할 수 있는가? 
 * control 그룹과 treatment 그룹에 대한 데이터는 있지만, treatment 전후의 데이터는 없으며, treatment 가 임의의 cutoff 혹은 threshold에 의해 정해진 경우에 활용할 수 있음
* 특정 discontinuity를 기점으로 discontinuous jump가 발생하면, 이를 통해 causal inference. Discontinuity가 발생하는 그 변수가 바로 running variable(assignment variable, forcing variable). 
* 만약 discontinuity가 없었다면 있었을 counterfactual에 대해서 discontinuity이전의 데이터를 바탕으로 discontinuity 이후를 extrapolation해서 구하는 방법  
<p><img src="/assets/images/rd.png" title="Regression discontinuity"/></p>
그림 출처: 인과추론의 데이터과학. (2022, June 18) 

### 2. Example of discontinuity 
* 21세(음주 허가 나이)를 기준으로 그 이후 교통 사고 사망률이 증가함  
<p><img src="/assets/images/rd_example.png" title="Regression discontinuity example"/></p>
그림 출처: 인과추론의 데이터과학. (2022, June 18) 

### 3. RD Estimation Strategies
* Regression discontinuity에서 중요한 요인
    1. Discontinuity 주변으로 어느정도 범위까지 고려하는지. 그 범위를 Bandwidth라고 부름. 예컨대, 전체 구간을 다 활용할 수도 있고(global), 특정 구간을 잘라서 분석할 수도(local)
    2. running variable에 대해서 어떤 방식으로 modeling을 해서 extrapolation을 할 지. Functional form 가정 하에 regression을 통해 modeling(parametric), 아무런 functional form 없이 평균값 비교를 통해 modeling (nonparametric)
* 그래서, cutoff 주변의 범위를 줄여서 분석하면 유리함. 범위를 줄이면 줄일수록, discontinuity주변의 특성이 비슷해지기 때문. 그래서, Bandwidth를 줄이는 게 좋으나, 샘플 사이즈가 작아진다는 단점  
<p><img src="/assets/images/Bandwidth.png" title="Bandwidth"/></p>
그림 출처: 인과추론의 데이터과학. (2022, June 18) 

* 아래 그림과 같이, Bandwidth, modeling of running variable을 어떻게 하는지에 따라서, 결과가 크게 달라질 수 있음  
<p><img src="/assets/images/discontinuity.png" title="discontinuity"/></p>
그림 출처: 인과추론의 데이터과학. (2022, June 18) 

### 4. Identification Assumption for RD

* Discontinuity가 있다고 해서 무조건 regression discontinuity로 분석하는 것은 아님. Discontinuity를 기준으로 treatment, control 집단이 있을 때, treatment 전후 데이터가 관찰 가능하다면 DID 방법을 적용하는 것이 더 적절함
* RD, DID 방법 모두 가능한 상황에서는 DID가 더 적절함. Causal inference에서는 늘 가정이 요구되고, 검증하기 쉬운 가정에 기반한 방법이 더 우월함. DID의 가정인 parallel trend assumption은 causal inference에서 가장 검증하기 쉬운 방법임
* 반면, RD의 가정은 검증하기 쉽지 않음. Discontinuity 전후를 분석, Discontinuity 이전을 바탕으로 counterfactual approximation을 하는 것이기 때문에, 아래 그림에서 동그라미 친 부분의 특성이 비슷해야 함  
<p><img src="/assets/images/discontinuity2.png" title="discontinuity2"/></p>
그림 출처: 인과추론의 데이터과학. (2022, June 18)  

* Discontinuity 전후의 데이터가 완전히 ceteris paribus를 충족하기를 기대하는 것은 아님. 차이가 있을 수 있다는 것을 인정함. 하지만, Discontinuity 전후의 차이가 running variable에 대한 어떠한 function으로 전부 다 설명할 수 있다고 봄. 그래서 만약, running variable의 function으로 설명 가능하다면, 차이가 있다 하더라도 control하는 것이므로, 그 차이를 제외하고는 비교 가능하다는 것
* 그런데, 사실상 어떤 모델이 true model인지 알 수는 없음. 그래서 RD, DID 중에 선택할 수 있다면, DID가 더 적합하다는 것 
* 그럼 true model이 뭔지 모르면 어떻게 하는가? 
> sensitivity test

* 아래 그림을 보면, , running variable인 x축과 outcome인 y축을 몇 차식으로 modeling을 하느냐에 따라서, 그래프가 다름(일반적으로 2차, 3차항까지 sensitivity test 진행)  
<p><img src="/assets/images/sensitivity_test.png" title="sensitivity test"/></p>
그림 출처: 인과추론의 데이터과학. (2022, June 18)  

### 5. Example of Regression Discontinuity 
* 아래 예시에서 running variable = 노동 조합 결성 투표 결과(binary). 찬성 50%를 넘으면 노동 조합이 결성되고, 그렇지 않으면 결성되지 않으므로 discontinuity 발생.  그래서 어떻게 보면 non-parametric 즉, experiment에 가까움 
* 그래도, control variable(Industry dummies, Year dummies)을 고려하기 위해 parametric한 방법을 사용함
* 그래서, 아무리 experiment 라고 하더라도, 일반적으로는 parametric을 더 많이 사용함
* 아래 그림 우측 하단의 표를 보면, 가장 왼쪽의 모든 기업을 다 분석에 포함하는 게 global,  50%를 기점으로 bandwidth를 좁혀서 설정하는 게 local. 그런데, 어떤 bandwidth가 최선인지 알 수 없기 때문에, 테스트를 하는 것이 일반적  
<p><img src="/assets/images/discontinuity_ex.png" title="Example of Regression Discontinuity"/></p>
그림 출처: 인과추론의 데이터과학. (2022, June 18) 

* 그런데, 만약 어떤 기업에서는 10:90의 비율로 찬반 결과가 나옴(어떤 기업에서는 40:60). 그러면 running variable을 단순히 binary 로 설정하는 것이 적합하지 않을 수 있음. 그래서, 단순히 binary로 평균을 비교하는 게 아니라, functional form으로 modeling하는 분석을 할 수 있음. 즉, running variable과 outcome의 관계를 parametric하게 추가 분석  
<p><img src="/assets/images/discontinuity_ex2.png" title="Example of Regression Discontinuity"/></p>
그림 출처: 인과추론의 데이터과학. (2022, June 18) 

* 위처럼 다양하게 sensitivity test를 진행하는 것이 일반적임

### 6. Imperfect Compliance: Fuzzy RD
* 현실에서는 running variable에 따라 discontinuity가 깔끔한 경우가 거의 없음. 위에서 살펴본 예시는 보기 드물게 깔끔한 case 
* 현실에서는 discontinuity에 의해서 treatment집단과 control집단이 깔끔하게 나뉘는 것이 아니고, treatment집단이 될 확률이 변함. 즉, Fuzzy RD가 더 많음  
<p><img src="/assets/images/fuzzy rd.png" title="Fuzzy RD"/></p>
그림 출처: 인과추론의 데이터과학. (2022, June 18) 

* Fuzzy RD는 결국 LATE 분석. Running variable에 대한 cutoff를 instrumental variable로 분석하는 게 결국은 regression discontinuity. 그래서, DID와는 접근이 다름. 
* 그래서 discontinuity에 의해 treatment를 받거나, 받지 않는 compliers에 대한 효과임 -> LATE  
<p><img src="/assets/images/sharp RD.png" title="sharp RD & FUZZY RD"/></p>
그림 출처: 인과추론의 데이터과학. (2022, June 18) 

* 참고로, Sharp RD는 always-taker, never-takers가 없는 모두가 compliers인 특별한 상황의 LATE임
* 아래 연구 사례에서는 연구 예산이 높아짐에 따라 공무원의 감시 대상이 되어서, 연구가 딜레이 되는지를 분석하고자 함. 연구 예산이 높아지면 공무원의 감시 대상이 될 확률이 높긴 하지만, 무조건 그런 것은 아님. 그래서 Sharp RD가 아니라, Fuzzy RD. 그래서, Two-stage least square로 분석을 진행함  
<p><img src="/assets/images/FUZZY_2SLS.png" title="FUZZY RD"/></p>
그림 출처: 인과추론의 데이터과학. (2022, June 18) 

### References 
인과추론의 데이터과학. (2022, June 18). [Bootcamp 4-3] 회귀 불연속 [Video]. YouTube. [https://www.youtube.com/watch?v=0V6Oq_5DDhg&list=PLKKkeayRo4PV_6-nbBgmUNOSpG1OO49M3&index=14][1]

[1]: https://www.youtube.com/watch?v=0V6Oq_5DDhg&list=PLKKkeayRo4PV_6-nbBgmUNOSpG1OO49M3&index=14