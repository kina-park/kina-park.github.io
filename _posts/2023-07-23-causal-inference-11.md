---
title: "[4-1] 도구변수"
layout: single
toc: true
categories: 
- "[관심분야 공부] Causal inference"
breadcrumbs: false
---

### 1. 도구변수 

* 도구변수를 potential outcome framework로 이해할 수 있어야만, regression discontinuity를 보다 제대로 이해하는 것이라 할 수 있음
* 마땅한 causal design이 없는 상황에서 활용할 수 있는 방법이 바로 도구 변수

### 2. Endogeneity in Regression
* Potential Outcome Framework 관점에서 Regression을 재해석한다면, conditional independence가 인과추론의 조건임. Control variable을 통해서 selection bias를 통제하고, 결과적으로는 control variable이 있을 때, control variable이 conditioning 되어 있는 상태에서 treatment variable인 x와 error 항의 상관관계가 없어야 한다는 조건. 

* Causal effect를 구하기 위해서는 Treatment와 error term이 exogenous해야 함. 그러나, Treatment와 error term이 endogenous하기 때문에 인과추론이 어려움. 그렇다면, Treatment와 error term이 100% correlation이 있을까? 그렇지는 않음. 따라서, Treatment variable과 상관관계를 갖는 endogenous한 부분(selection bias로서 causal inference에 문제가 되는 부분)과 상관관계를 갖지 않는 exogenous한 부분(causal inference에 활용할 수 있는 부분)이 공존함

* 만약 통계적 기법을 통해 이 두 부분을 구분할 수 있다면 endogeneity 문제를 해결할 수 있지 않을까? 

### 2. Taking Endogeneity Out: Instrumental Variable

* 위와 같은 목적에서 나온 통계적 도구가 instrumental variable(도구 변수). 말 그대로 도구적인 역할

* 도구변수를 활용해서 treatment variable에서 exogenous한 부분을 예측하면, 예측되지 않는 부분은 endogenous한 부분만 남음. 이를 잘 활용하면 outcome variable에 미치는 인과적인 효과를 잘 추론할 수 있을 것  
<p><img src="/assets/images/iv.png" title="instrumental variable"/></p>
그림 출처: 인과추론의 데이터과학. (2022, June 18) 

### 3. Identification Assumptions for IV

1. 도구변수는 treatment variable을 설명할 수 있다 (Relevance)
2. 도구변수는 error term과의 상관관계가 없어야 한다 
    * 도구변수는 treatment를 통해 outcome variable에 영향을 미쳐야 한다 (Exclusion restriction)
    * 도구변수가 outcome에 대해 어떠한 교란 요인도 갖지 않아야 한다 (Exogeneity)
    <p><img src="/assets/images/two-stage-least-squares.png" title="two-stage-least-squares"/></p>
 
<p><img src="/assets/images/equation.png" title="equation"/></p>
그림 출처: 인과추론의 데이터과학. (2022, June 18)  


### 4. IV Examples

* Exogenous Event-based IVs (Ideal)
    * 연구의 기본 아이디어: 제국주의 당시의 식민지 지배 전략이 현재 많은 국가들의 사회제도의 근간을 이루고 있다. 그래서 사유재산권의 도구변수로서 침략국의 식민지에서의 사망률, 식민지 인구밀도를 활용함. 어떤 국가의 경우 정착을 해서 각종 풍토병 등으로 사망률이 높았고, 또 어떤 국가의 경우 정착이 용이했을 것임. 그래서 식민지에서의 사망률이 높을수록 그곳에 정착해서 살면서 당시 유럽 국가에서 발전되었던 사회제도를 도입하고, 그 지역을 발전시키기 보다는 식민지에서 현지 주민들을 착취, 부유한 자원을 약탈하는 방향(현지 인구밀도가 높을수록 그랬을 것)으로 식민 전략이 이루어졌기 때문에, 그 만큼 그 지역 혹은 나라에서 사유재산권 제도가 발전할 여지가 부족했다는 것. 

    * 식민지 사망률과 인구 밀도는 1990년대의 경제 성장을 설명하는 데 있어, 아마 사회제도를 거치지 않으면 어려울 것. 도구 변수로서의 역할을 잘 한다고 볼 수 있음  
<p align="center"><img src="/assets/images/Exogenous Event-based IVs.png" width="80%" height="80%" title="Exogenous Event-based IVs"/></p>
그림 출처: 인과추론의 데이터과학. (2022, June 18)   

2. 이외에도 지역적 특성, 상위 집단의 트렌드, 네트워크 환경 등이 도구변수로 잘 활용됨 

### References 
인과추론의 데이터과학. (2022, June 18). [Bootcamp 4-1] 도구변수 [Video]. YouTube. [https://www.youtube.com/watch?v=fL_SBIg-bnY&list=PLKKkeayRo4PV_6-nbBgmUNOSpG1OO49M3&index=12][1]

[1]: https://www.youtube.com/watch?v=fL_SBIg-bnY&list=PLKKkeayRo4PV_6-nbBgmUNOSpG1OO49M3&index=12