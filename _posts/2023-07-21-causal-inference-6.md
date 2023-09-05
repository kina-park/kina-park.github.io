---
title: "[2-4] 매칭과 역확률가중치"
layout: single
toc: true
categories: 
- "[관심분야 공부] Causal inference"
breadcrumbs: false
---

### 1. Regression is analogous to matching, but there are differences
* regression은 control variable과 selection bias에 대해서 linear function이라는  functional form을 가정하고, control variable을 conditioning함으로써 특성을 유사하게 만드는 것이라 할 수 있다. 
* 그런데, 매칭은 functional form 없이 단순히 control variable에서의 특성이 유사한 샘플들끼리 서로 직접적으로 매칭함으로써 특성을 유사하게 만드는 방법. Functional form을 특별히 가정하지 않는다는 의미에서 flexible 하다. 


### 2. 매칭의 두 가지 방법 
1. **경향 점수 (propensity score matching / 성향도 점수)**
    * 전통적으로 가장 많이 활용되었음 
    * control variable이 주어진 상태에서, treatment를 받을 확률
    * 경향 점수를 기준으로 처치 집단에서의 propensity score와 통제 집단에서의 propensity score 가 서로 각자 propensity score를 계산하고, propensity score가 비슷한 데이터들끼리 서로 매칭
    * 경향 점수를 계산하는 방법: 로지스틱 회귀, 프로빗 회귀를 활용해서 treatment의 여부가 Y값으로, 나머지 통제 변수들을 독립변수로 넣어서. 그렇다면, 0-1까지의 값을 갖는 경향 점수를 구할 수 있음
    * 매칭도 마찬가지로 selection on observable 전략임 -> propensity score를 모두 설명할 수 있는 즉, treatment를 받을지 말지 여부를 결정하는 모든 변수들을 우리가 알고 있다는 전제 (굉장히 강한 가정 포함)
    * 매칭된 처치 집단과 통제 집단 간의 평균값 단순 비교하면 됨. 더 나아가 매칭된 샘플에 대해서만 regression 모델을 추가적으로 분석할 수도 있음
    * 이와 같은 방법에 대해 다음과 같이 두 가지 비판이 제기됨: 
        * 1)  propensity score가 서로 비슷하다고 해서, 그들의 특성과 그들의 potential outcome, conterfactual이 전부 비슷하다고 상정하는 데 한계가 있다
        * 2) propensity score를 구하는 로지스틱, 프로빗 회귀 방법은 전적으로 편의에 의한 방법이며, 실제로 propensity score가 이 모형을 따른다는 법칙이 없다. 
    <p><img src="/assets/images/propensity_score.png" title="propensity score"/></p>
    그림 출처: 인과추론의 데이터과학. (2022, June 15

2. **Coarsened Extract Matching (CEM)**
    * 경향 점수의 한계로 등장한 방법론임
    * control variable이 직접적으로 비슷한 데이터끼리 매칭시킴. 정확히 일치하는 것은 어렵기 때문에, 약간은 느슨하게 구간을 설정. 구간 매칭. 다양한 통계적 가정에서 자유롭다는 이점이 있음
    * 그러나, 변수가 많아질수록 구간 매칭도 어려워진다는 한계가 있는 관계로, 특정 연구방법론이 우세하다고 볼 수는 없음
    <p><img src="/assets/images/cem.png" title="CEM"/></p>
    그림 출처: 인과추론의 데이터과학. (2022, June 15)

3. **Weighting**
    * weighting보다 매칭이 더 직관적이라는 이유로 더 많이 쓰이긴 함. 그러나, 경우에 따라서 매칭은 아예 사용 불가한 경우가 있음. 하지만, 그럴 때 weighting은 가능함
    * weighting은 기본적으로 propensity score를 활용함 
    * 그런데, 매칭은 propensity score가 비슷한 데이터끼리 매칭해서 통제 집단과 처치 집단의 propensity score를 유사하게 만드는 방법이지만, weighting은 매칭을 하지 않고, propensity scored의 역수(inverse) 만큼의 가중치를 더 부여함. Treatment를 받을 확률이 작은 그룹에는 더 많은 가중치를 부여함으로써, 확률을 더 키우고, 반대로, treatment를 받을 확률이 높은 그룹에는 가중치를 작게 줘서 확률을 낮추는 기법. 그래서, treatment를 받을 확률을 서로 비슷하게 만들어 주는 것 
    * treatment probability의 역수를 가중치를 주는 것이기 때문에, 이를 inverse probability weighting (IPW) 라고 부름. 여기서의  probability는 propensity score를 의미한다고 생각하면 됨. 
    * 그림에서 보면, 성인이면서 남성인 경우(conditioning) 5명이 있을 때, treatment를 받을 확률은 1/5. 그럼 treatment를 받을 확률 1/5의 역수인 5를 가중치로 부여함 
    * 그럼 원래 10명이 있을 때, 처치를 받을 확률이 (동전 던지기의 경우처럼) 5:5 인데, 그림의 아래 부분 처럼 5:5로 만들 수 있음
    <p><img src="/assets/images/weighting.png" title="Weighting"/></p>
    그림 출처: 인과추론의 데이터과학. (2022, June 15)

### 3. Inverse Probability Weighting
* 아래 그림을 보면, (c를 고려하지 않는 경우) X가 1일 때, X가 0일 때보다 Y가 약 45% 정도 높다.  X가 1일 때 C값 평균 != X가 0일 때 C값 평균 -> C가 selection bias  
<p><img src="/assets/images/ipw.png" title="Inverse Probability Weighting"/></p>
그림 출처: 인과추론의 데이터과학. (2022, June 15)

* selection on observavles assumption에 의하면, 노란색 영역은 c가 1일 때, 파란색 영역은 c가 0일 때, 같은 영역 내에서는 exchangeability가 성립한다. 즉, counterfactual를 대체한다. (아래 그림 참고)  
<p><img src="/assets/images/ipw2.png" title="Inverse Probability Weighting"/></p>
그림 출처: 인과추론의 데이터과학. (2022, June 15)

* c가 1일 때, x=0인 경우가 30명. 이 30명에 대한 counterfactual은 사실 없음. 그렇지만, 위의 x=1인 경우의 결과값 90%로 그대로 대체할 수 있다고 봄. 즉, c=1일 때, 처치를 받지 않은 30명이 만약에 처치를 받았다면 90%의 효과가 있었을 것이다 라고 가정할 수 있음

* 이렇게 해서, counterfactual을 채우면 이를, Pseudo-population이라고 부름. 이를 통해, treatment와 counterfactual을 구해서 causal effect를 구할 수 있음

* counterfactual을 대체한다는 것이 어떤 의미일까? 
    * Control variable이 주어진 상태에서의, treatment를 받을 확률(propensity score)의 역수를 곱하고, 반대로 treatment를 받지 않을 확률의 역수를 곱하는 거랑 정확하게 동일한 접근. 
    * C가 1일 때, X=1일 확률 50% 
    * C가 1일 때, X=0일 확률 50% 
    * C가 0일 때, X=1일 확률 75%
    * C가 0일 때, X=0일 확률 25% 

* 이렇게, Pseudo-population을 채우면, 아래 그림과 같이 결과를 얻을 수 있음.  
<p><img src="/assets/images/pseudo-population.png" title="pseudo-population"/></p>
그림 출처: 인과추론의 데이터과학. (2022, June 15)

* 즉, c의 값에 관계없이 트리트먼트를 받을 확률을 50대 50이 됨. 이는 random-assignment와 동일함. 즉, c에 independent하게 완전히 random assignment랑 동일한 효과를 주고, inverse probability weighting의 causal effect를 구할 수 있는 메커니즘이 됨. 

### 4. Weighting vs Regression/Matching
* Weighting은 근본적으로 Regression/Matching과 접근이 다름. 
* Regression에서 control variable 통제를 하는 것은 control variable의 값을 고정하는 방식. 마찬가지로 Matching에서도 처치 집단과 통제 집단에서 control variable의 값들이 서로 같도록 직접 매칭을 함으로써, 그 값을 고정하는 방식. 
* 즉, Regression/Matching 모두 control variable이 selection bias를 모두 설명한다는 가정 하에 control variable의 값을 통제하고 conditioning 함으로써 selection bias를 없애고자 하는 전략. 
* 반면, Weighting은 control variable의 값을 통제하고 conditioning 하는 것이 아니라, selection bias를 야기하는 control variable에 independent하게 (상관없이) treatment를 받을 확률을 50대 50으로, 즉, random assignment에 가깝게 만드는 방식. “control variable에 관계없이  treatment를 50 대 50의 확률로 만들 수 있도록 Pseudo-population을 구성하자!” 이를 그림으로 직관적으로 그려내자면, control variable이 treatment에 미치는 영향을 제거하는 방식(왼쪽). 
<p><img src="/assets/images/weight_difference.png" title="Weighting vs Regression/Matching"/></p>
그림 출처: 인과추론의 데이터과학. (2022, June 15)

* Weighting 기법이 인과추론의 gold-standard인 RCT를 resemble한다고 해서, 어느 경우에나 최선의 방법인 것은 아님. Weighting은 propensity score를 잘 구해야 한다는 굉장히 강한 전제 조건이 필요함

### 5. Comparison of Regression, Matching, and Weighting  
<p><img src="/assets/comparison.png" title="Comparison of Regression, Matching, and Weighting"/></p>  
그림 출처: 인과추론의 데이터과학. (2022, June 15)

* 기본적으로 매칭이 가장 직관적이고 우선적으로 고려하는 방법임
* 그러나, within-group comparison이 굉장히 복잡한 경우라면, 매칭은 어려움. 이 경우 regression이 좀 더 간결하게 분석할 수도 있고, matched sample이 달라지는 게 크게 문제가 되는 상황이라면 weighting이 더 나을 수도. 혹은 매칭을 통해서 컨디셔닝 하는 게 문제가 될 수 있는 research context라고 한다면, weighting이 나을 수도 있음. 
* 매칭을 기준으로 방법론을 비교하면서 선택하는 것이 일반적인 접근
* 그러나, 위 방법들은 모두 관찰된 변수를 통해 선택 편향을 설명할 수 있다는 전략을 취함으로써 공통적인 한계를 지님. 
* 따라서, 위 세가지 방법들 중 한가지를 선택한다면, 반드시 관찰되지 않은 변수들의 영향은 미비하다는 것을 설득력 있게 주장해야 함. 아니면, 관찰 가능한 변수들을 아주 잘 고려해야 할 필요가 있음 
* 즉, 위 세 가지 방법은 causal inference를 위한 최후의 보루, 최후의 수단이 되어야 한다고 보는 것이 적절함. 현실에서 관찰가능한 요인들만 가지고 설명하는 것은 설득력이 사실 떨어짐. 다른 대안이 없을 경우 고려해 볼 수 있는 것들이 selection on observables 전략
* 참고로, 이 세가지 방법은 보조적인 역할로서의 기능이 강한데, 이를테면 Natural experiment 를 더 rigorous하게 만들 수 있음 


### References 
인과추론의 데이터과학. (2022, June 15). [Bootcamp 2-4] 매칭과 역확률가중치 [Video]. YouTube. [https://www.youtube.com/watch?v=BVBUQz3Ix8w&list=PLKKkeayRo4PV_6-nbBgmUNOSpG1OO49M3&index=8][1]

[1]: https://www.youtube.com/watch?v=BVBUQz3Ix8w&list=PLKKkeayRo4PV_6-nbBgmUNOSpG1OO49M3&index=8