---
title: "인과추론의 어려움과 인과추론의 전략"
layout: single
classes: wide
toc: true
categories: 
- Causal inference
---

### 1. Everything is Endogenous 

* 내생성은 인과추론을 어렵게 하는 근본적인 문제
* 내생성이란 모든 것들이 복잡다기하게 상호 간 영향을 내생적으로 주고 받는 것 
* 우리가 관심있는 원인 변수 이외에도 결과에 영향을 미치는 교란요인들이 서로 얽혀 있기 때문에 인과관계를 드러 내기가 어려
이러한 내생성 문제를 해결하는 것이 causal inference. 대부분의 인과추론 방법론들이 내생성 문제를 극복하기 위해 발전됨.  
* Endogeneity in various forms: (1) selection bias, (2) backdoor path, (3) Endogeneity in regression (correlation between error term and x's)

### 2. 다양한 인과추론 접근법 
* Research design for causal inference
    * Randomized controlled trial, (Natural) Quasi-experiment, Local Average Treatment Effect(LATE)
* Selection model (statistical modeling)
* Causal graph (graphical modeling)

* 실험적 연구방법론에 기반한 인과추론의 최대 단점은 다른 상황에서의 적용이 어렵다는 점
    * Card & Krueger(1994)의 연구결과(natural experiment 기반)에 따르면, 최저 임금 인상은 고용에 부정적 영향을 미치지 않으며, 직관과는 반대로 긍정적인 영향을 미치는 것으로 나타남. 그러나, 이후에 저자 중 한명은 오바마 정권과 함께 일하며 최저 임금을 인상하지 않을 정책을 제안한 바 있음. 왜 그럴까? 이유는 context가 다르다는 것.  
* 즉, external validity, transportability가 실험적 연구방법론에 기반한 인과추론의 최대 약점 중 하나이며, 이러한 문제 해결을 위한 대안으로서 structural causal model을 활용하는 방법이 있음 

### References
인과추론의 데이터과학. (2022, June 14). [Bootcamp 1-2] 인과추론의 어려움과 인과추론 전략 [Video]. YouTube. [https://www.youtube.com/watch?v=luesQBhBBI4&list=PLKKkeayRo4PV_6-nbBgmUNOSpG1OO49M3&index=3.][1]

[1]: https://www.youtube.com/watch?v=luesQBhBBI4&list=PLKKkeayRo4PV_6-nbBgmUNOSpG1OO49M3&index=3
