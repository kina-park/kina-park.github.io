---
title: "[3-4] 가상의 통제집단"
layout: single
toc: true
categories: 
- "[관심분야 공부] Causal inference"
breadcrumbs: false
---

### 1. Donor pool

* Impact of California Anti-Tobacco Legislation를 알고 싶은 상황이라고 가정하자. Treatment를 받기 전 캘리포니아 주와 pararell trend를 충족하는 다른 주가 존재하지 않음. synthetic california를 만들기 위해 어떻게 해야 할까? 

* Impact of Reunification on West Germay를 알고 싶은 상황이라고 가정하자. 위와 마찬가지로, 서독과 유사한 상황의 국가를 찾을 수 없다는 것이 문제. 

* 이때, Synthetic control을 활용해서 몇몇 국가에 weight을 줘서 통일 이전의 서독의 경제성장 흐름을 그대로 모방하는 가상의 통제 집단을 구성할 수 있음. 이걸 donor pool이라 하고, weight을 어떻게 부여하는지가 핵심  
<p><img src="/assets/images/donor_pool.png" title="donor pool"/></p>
그림 출처: 인과추론의 데이터과학. (2022, June 17)

### 2. synthetic control 구성

* 그렇다면 synthetic control을 어떻게 구성하는가? 기본적인 아이디어는 treatment를 받기 이전 기간에 outcome이라든지 추가적으로 predictor가 있어서 treatment를 받기 이전의 outcome과 predictor에 대해서 treatment그룹에서의 값과 control 그룹에서의 조합의 차이를 분석했을 때, 그 차이가 최소가 되도록 weight(원래는 합이 1이 되도록) 구하는 방식. 최근에는 synthetic control을 구하기 위한 방법이 굉장히 다양함. 아래와 같이, Weight이 음수인 경우도 있음  
<p><img src="/assets/images/synthetic_control_weight.png" title="synthetic_control_weight"/></p>  
그림 출처: 인과추론의 데이터과학. (2022, June 17)

* 다음과 같이 최근에 제안된 기법이 있음: Synthetic difference-in-differences, Bayesian synthetic control

    * Synthetic difference-in-differences  
    <p><img src="/assets/images/Synthetic difference-in-differences.png" title="Synthetic difference-in-differences"/></p>
    그림 출처: 인과추론의 데이터과학. (2022, June 17)

    * Bayesian synthetic control  
    <p><img src="/assets/images/Bayesian synthetic control.png" title="Bayesian synthetic control"/></p>
    그림 출처: 인과추론의 데이터과학. (2022, June 17) 

* Synthetic control은 궁극적으로 prediction problem으로 귀결될 수 있다라는 점. 그래서 예측과 관련된 기존의 다양한 기법, 머신러닝도 적용되고 있음. Basically, the synthetic control approach is the prediction problem. Like for the lasso, the goal of synthetic controls is out-of-sample prediction(Abadie 2021, p.408). 계속 주목해 볼만 함

* 그런데, 한 가지 주의할 점. Synthetic control은 우리에게 익숙한 방식으로 통계적으로 asymptotic theory(점근적 이론)에 기반해서 std, p-value를 구할 수 없음. 그래서 이 경우 어떤 식으로 inference를 하는가? -> Placebo tests. Synthetic control과 actual 값의 차이가 실제로 treatment 이전과 이후에 얼마나 차이가 있는지를 계산  
<p><img src="/assets/images/placebo_tests.png" title=" Placebo tests"/></p>
그림 출처: 인과추론의 데이터과학. (2022, June 17) 

* Sensitivity tests for synthetic control
    * Donor pool, variable을 달리 했을 때에도 견고한가? 의 문제. 
    * Train-test split의 방법도 적용 가능함 (기본적으로 prediction의 성격이기 때문). Treatment 이전 기간을 synthetic control 학습하는 기간으로  
<p><img src="/assets/images/train_test_split.png" title="train_test_split"/></p>
그림 출처: 인과추론의 데이터과학. (2022, June 17) 

* Interrupted Time-Series Analysis
    * Control 그룹은 없고, treatment 전후를 기준으로 인과 추론하는 방법. 그러나 인과추론에서 control 그룹이 없으면, 한계가 굉장히 많기 때문에 최후의 보루로 사용하는 방법으로 보는 것이 적절함
    * 참고로, 구글에서 만든 causaleffect r package – industry에서는 자주 활용되지만, 학계에서는 그렇지 않음 

### References 
인과추론의 데이터과학. (2022, June 17). [Bootcamp 3-4] 가상의 통제집단 [Video]. YouTube. [https://www.youtube.com/watch?v=jCNaQocWumo&list=PLKKkeayRo4PV_6-nbBgmUNOSpG1OO49M3&index=11][1]

[1]: https://www.youtube.com/watch?v=jCNaQocWumo&list=PLKKkeayRo4PV_6-nbBgmUNOSpG1OO49M3&index=11 