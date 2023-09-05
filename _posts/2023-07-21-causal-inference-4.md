---
title: "[2-2] 무작위 통제 실험"
layout: single
toc: true
categories: 
- "[관심분야 공부] Causal inference"
breadcrumbs: false
---

### 1. Random Assignment는 왜 중요한가? 
* 아래 그림은 인과추론을 위한 연구 디자인의 위계 관계를 나타낸 것이다. 
<img src="/assets/images/스크린샷 2023-06-27 170838.png" width="80%" height="80%" title="Causal hierarchy"/>  

* 그렇다면, Random Assignment는 왜 중요한가?  
> 큰 수의 법칙에 근거한다면, random assignment에 의한 처치/통제 집단 구분은 처치 여부만 제외하면 두 집단의 특성이 비슷할 것이다. 따라서, 두 집단은 비교 가능할 것임. 즉, 단연 Ceteris Paribus 조건을 만족하므로, Random Assignment는 이상적인 counterfactual과 가장 가까운 control group을 만들 수 있는 가장 효과적인 방법이다. 

* 아래 그림은 RCT의 좋은 사례를 보여준다. 살펴보면, Confounder에서 각 요인의 집단 별 분포가 유사하게 잘 나타나는 것을 확인할 수 있다. 
<p><img src="/assets/images/rct_example.png" title="RCT distribution"/></p>

* 그러나, 현실에서는 대부분의 경우에 이러한 비용, 윤리 등의 문제로 RCT가 불가능함. 따라서, 앞으로의 강의에서는 RCT가 불가능한 상황에서 어떻게 해야 할지 다룰 것이다. 


### References
인과추론의 데이터과학. (2022, June 15). [Bootcamp 2-2] 무작위 통제실험 [Video]. YouTube. [https://www.youtube.com/watch?v=fTYU0yXHxvU&list=PLKKkeayRo4PV_6-nbBgmUNOSpG1OO49M3&index=5][1]

[1]: https://www.youtube.com/watch?v=fTYU0yXHxvU&list=PLKKkeayRo4PV_6-nbBgmUNOSpG1OO49M3&index=5


