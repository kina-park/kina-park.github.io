---
title: "[4-4] 통제함수와 선택모형"
layout: single
toc: true
categories: 
- "[관심분야 공부] Causal inference"
breadcrumbs: false
---

### 1. Control Function
* residual inclusion method, selection bias correction method
* 기본적으로 도구변수를 활용하는 것은 동일함(도구변수를 활용해서 exogenous한 부분을 설명하는 것). 그런데, selection bias를 야기하는 endogenous한 부분(variation not explained by IVs), 실제로 endogenous한 부분과 관련되어 있는 selection bias(error term에서 treatment와 관련되어 있는 Predicted Residual)를 활용해서 error term에서의 selection bias를 예측하고, selection bias를 통계적으로 계산해서, 그 계산된 selection bias를 직접 control함으로써 인과추론을 하고자 하는 방법
* 그래서, potential outcome framework, counterfactual의 개념이 아님. 굉장히 통계적 접근. 즉, 인과추론에 문제되는 endogenous한 부분을 통계적으로 통제하자는 접근  
<p><img src="/assets/images/control function.png" title="control function"/></p>
그림 출처: 인과추론의 데이터과학. (2022, June 18) 

* 쉽게 말해, 아래 식에서 v를 통해서 u를 예측하자는 접근
* 아래 p(주황색 표시)가 의미하는 것이 뭘까? 
    * 수식에서는 coefficient처럼 쓰이지만, 사실상 값 자체보다는 부호가 더 중요함. p라는 것은 treatment를 설명하지만 관찰되지 않는 v와 outcome을 설명하는데 관찰되지 않는 u의 correlation. 즉, treatment를 설명하는 unobsevable factor와 outcome을 설명하는 unobservable factor의 correlation  
    <p><img src="/assets/images/equation2.png" title="control function equation"/></p>
그림 출처: 인과추론의 데이터과학. (2022, June 18) 

### 2. Two-Stage Least Squares vs Control Function

* Two-Stage Least Squares의 장점은 LATE 개념 하에서 Potential Outcome Framework에 통합될 수 있다는 것. 단점은 linear한 경우가 아니면 extension이 어려움 
* Control Function의 단점은 통계적으로는 endogeneity를 통제한다는 접근은 make sense하지만 causal effect를 해석하기가 직관적이지가 않음. 장점은 LATE가 아니라, ATE를 구할 수 있음. Endogeneity, selection bias가 복잡한 상황에서 extension이 유연함  
<p><img src="/assets/images/2SLS-Control-function.png" title="Two-Stage Least Squares vs Control Function"/></p>
그림 출처: 인과추론의 데이터과학. (2022, June 18) 

### 3. Example
* Effect of Advertising on Sales
    * 아래 예시처럼 Errorterm이 outcome에만 영향을 미칠 뿐만 아니라, causal effect(treatment의 coefficeint) 자체에도 selection bias가 작용하는 경우가 있을 수 있음  
    <p><img src="/assets/images/effect_of_advertising_on_sales.png" title="Effect of Advertising on Sales"/></p>
그림 출처: 인과추론의 데이터과학. (2022, June 18) 

### References 
인과추론의 데이터과학. (2022, June 18). [Bootcamp 4-4] 통제함수와 선택모형 [Video]. YouTube. [https://www.youtube.com/watch?v=0HUf8aH1B9Y&list=PLKKkeayRo4PV_6-nbBgmUNOSpG1OO49M3&index=15][1]

[1]: https://www.youtube.com/watch?v=0HUf8aH1B9Y&list=PLKKkeayRo4PV_6-nbBgmUNOSpG1OO49M3&index=15