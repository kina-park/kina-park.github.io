---
title: "[3-1] 디자인 기반의 인과추론"
layout: single
toc: true
categories: 
- Causal inference
---

### 1. Research design matters 
<p><img src="/assets/images/causal hierarchy.png" title="causal hierarchy"/></p>
그림 출처: 인과추론의 데이터과학. (2022, June 17)


* Random assignment 없이도 적절한 research design을 잘 사용할 수 있다면, 충분히 비교 가능한 control group을 구성함으로써 causal inference를 할 수 있음. 이러한 디자인을 Quasi-Experimental Design(준실험)이라고 함

* 관찰되지 않은 요인들까지 설명하기 위해서는 당연히 우리가 관찰할 수 있는 변수 그 이상이 필요함. RCT에서는 그것이 random assignment이고, Quasi-Experiment, Instrumental Variable에서는 research design 

* 준실험은 RCT와 동일한 선상에 있다고 볼 수 있음
> Quasi-experiment = RCT without random assignment 
> The only difference between RCT and quasi-experiment lies at the treatment assignment mechanism. 
> 만약, 사회제도가 경제성장에 미치는 영향을 추론하고 싶다고 가정해보자. 비교 국가를 남한과 북한으로 설정하면 어떨까? 남한과 북한은 사회제도라는 요인을 제외하면 다른 모든 요인이 유사한가? 비교 가능한가? 그렇지 않다. 그러므로 사회제도가 경제성장에 미치는 영향을 추론하고 싶다면, 사회제도를 제외한 다른 모든 요인이 유사한 두 국가를 찾아야 하는데, 비교 가능한 control group을 구성할 수 있는 context와 데이터 등을 일컫는 것이 바로 준실험 


* Potential Outcome Framework에 기반한 Quasi-Experiment 방법을 흔히 design-based approach라고 부름. 그리고, design-based approach에서 인과추론이 얼마나 잘 되었는 지의 여부는 얼마나 많은 데이터를 활용했는지, 얼마나 복잡한 통계 모형을 활용했는지에 따라 결정되는 것이 아니라, Ceteris Paribus를 만족하는 비교 가능한 control group을 얼마나 잘 구성할 수 있는 research design을 고안했는지에 따라 인과추론의 질이 결정됨

* Selection on Unobservables -> Research design을 통해 관찰되지 않은 요인에 의한 selection bias까지도 고려할 수 있음




### References 
인과추론의 데이터과학. (2022, June 17). [Bootcamp 3-1] 디자인 기반의 인과추론 [Video]. YouTube. [https://www.youtube.com/watch?v=K0huXsARLc4&list=PLKKkeayRo4PV_6-nbBgmUNOSpG1OO49M3&index=9][1]

[1]: https://www.youtube.com/watch?v=K0huXsARLc4&list=PLKKkeayRo4PV_6-nbBgmUNOSpG1OO49M3&index=9