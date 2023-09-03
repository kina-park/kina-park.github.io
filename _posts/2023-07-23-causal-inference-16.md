---
title: "[5-2] 디자인 기반의 인과추론에서의 인과 그래프 활용"
layout: single
toc: true
categories: 
- Causal inference
---

### 1. Structure-Based Research Design

* 한 가지 사례; ‘호르몬 치료를 받으면 자궁암에 걸릴 확률이 높아진다.’는 결과를 두고([그림1]), 두 가지 입장으로 나뉨. 
    * 일각에서는 호르몬 치료는 자궁출혈이 야기되고, 자궁출혈이 있으면 더 자주 자궁 관련 검사를 진행하게 되므로 자궁암 발병 확률이 높아지는 것이므로, 자궁출혈이 통제되었을 때만(자궁 출혈이 있는 사람들에 한해서 분석을 하는 등), 비로소 인과추론을 할 수 있다고 주장함([그림2], [그림3]). 
    * 반면, 이러한 주장에 대해 물론 가장 이상적으로는 호르몬 치료에 대한 무작위 실험을 하는 것이지만, 이는 윤리적 문제의 소지가 크며, 자궁암 또한 자궁출혈을 야기할 가능성이 있으므로 ([그림4], [그림5]) collider 차단하면 인과추론이 어려움
* 결국, 무작위 실험 없이 인과관계를 추론하기 위해서는 적절한 연구 디자인을 고안해야 함  
<p><img src="/assets/images/Structure-Based Research Design.png" title="Structure-Based Research Design"/></p>
그림 출처: 인과추론의 데이터과학. (2022, June 19) 

* 그런데, 여기서 만약, 자궁출혈을 conditioning 하면, ‘호르몬치료 – 자궁출혈 -자궁암 진단’의 backdoor path는 차단되지만, 자궁출혈은 collider 이므로, 새로운 path를 생성하는 것이 되므로, 또 다른 noncausal association을 만들게 되는 것 ([그림 5]에서 노란색으로 표시) 
* 사실상 우리가 알고 싶은 것은 아래 [그림5]에서 주황색으로 표현된 causal effect  
<p><img src="/assets/images/Structure-Based Research Design2.png" title="Structure-Based Research Design2"/></p>
그림 출처: 인과추론의 데이터과학. (2022, June 19) 

* Weighting, IPA 방법이 Conditioning이 불가한 상황에서 유용하게 활용될 수 있음 

### 2. Design of Control Variables / Conditioning Strategies 
* Selection on Observables Strategies에서 causal graph가 유용할 수 있음  
* 무조건 통제 변수를 많이 투입해서 분석하는 것이 좋을까? 예컨대, 염분섭취량(SOD)가 혈압(SBP)에 미치는 영향의 실제 크기는 2임. 아래 그림처럼 차례로 어떠한 변수도 통제하지 않았을 때, Confounder인 나이(AGE)를 통제했을 때, 나이와 함께 Collider인 단백질 섭취량(PRO)을 통제했을 때의 결과를 살펴보자  
<p><img src="/assets/images/design_of_control_variables.png" title="Design of Control Variables"/></p>
그림 출처: 인과추론의 데이터과학. (2022, June 19) 

* 위 그림의 그래프를 보면 알 수 있듯이, Collider까지 통제한 결과 실제 인과효과와는 매우 다른 값이 도출됨. 즉, 좋은 control 변수와 나쁜 control 변수가 있기 때문에, 모든 변수를 통제하는 것이 최선은 아니라는 것
* 따라서, 변수 통제 여부를 결정하기 위해 causal graph를 그릴 것을 많은 연구자들이 강조함. 우선 backdoor path를 야기하는 confounder 변수는 모두 통제하는 것이 맞음. 반면, mediator는 특수한 경우를 제외하고는 통제하면 안됨. Collider를 통제하면 spurious backdoor path가 생기기 때문에 통제하면 안 됨  
<p><img src="/assets/images/causal_diagram.png" title="causal diagram"/></p>
그림 출처: 인과추론의 데이터과학. (2022, June 19) 

### 3. Communicating Identification Assumptions
* 모든 인과추론에는 적절한 가정이 요구됨. 이러한 가정이 Identification Assumptions. 실제로 Identification Assumptions은 통계적인 검정의 영역이라기 보다는 theoretical justification의 영역임. 
* 가장 대표적인 것이 도구 변수. 도구변수의 Identification Assumptions은 크게 두 가지;    조건 1) 도구변수는 treatment variable을 설명할 수 있어야 함(relevance condition), 조건 2) 도구변수가 그 결과변수에 영향을 주는 unobservable factor가 담겨있는 error term과의 상관관계가 없어야 함 
* 위에서 조건 1) 은 비교적 쉽게 통계적으로 검증이 가능함. 조건 2)는 통계적으로는 make sense 하지만, 완벽하게 검증할 방법이 (부분적으로는 가능하지만)마땅치 않음. 그래서, theoretical justification을 통해 support하는 방법 밖에 없음  
<p><img src="/assets/images/identification_assumptions_of_iv.png" title="identification_assumptions_of_iv"/></p>
그림 출처: 인과추론의 데이터과학. (2022, June 19)  

* 예컨대, 도구변수와 결과변수에 모두 영향을 미치는 unobservable confounder가 있다고 가정해보자. 이러한 경우 만약,  unobservable confounder와 도구변수를 매개하는 변수(위 그림 파란색 부분)를 안다면, 이를 통제함으로써 path를 차단, exogeneity of IV를 충족할 수 있음

### 4. Transportability: From RCTs to Observational Studies
* 기본적으로 design-based approach는 실험적인 접근이라고 볼 수 있음. 실험적 연구의 가장 큰 한계점 중 하나는 Transportability. 예를 들어, 특정 집단에서 시행한 RCT 혹은 특정 상황에서 design을 활용한 causal experiment는 다른 집단 혹은 다른 상황으로의 적용이 어려움
* 그래서, 그 적용이 가능한지를 다루는 것이 transportability  
<p><img src="/assets/images/transportability.png" title="transportability"/></p>
그림 출처: 인과추론의 데이터과학. (2022, June 19)  

### References 
인과추론의 데이터과학. (2022, June 19). [Bootcamp 5-2] 디자인 기반의 인과추론에서의 인과 그래프 활용 [Video]. YouTube. [https://www.youtube.com/watch?v=ZAdr7TB1bF4&list=PLKKkeayRo4PV_6-nbBgmUNOSpG1OO49M3&index=17][1]

[1]: https://www.youtube.com/watch?v=ZAdr7TB1bF4&list=PLKKkeayRo4PV_6-nbBgmUNOSpG1OO49M3&index=17 