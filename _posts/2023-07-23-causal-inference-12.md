---
title: "[4-2] 인과추론 관점에서의 도구변수"
layout: single
toc: true
categories: 
- Causal inference
---

### 1. IV from the Perspective of Potential Outcome
* Potential Outcome Framework 관점에서 도구 변수를 이해하는 것이 중요함
* 이를 가능하게 하는 것이 Local Average Treatment Effect(LATE)

### 2. IV as a Treatment Assignment Mechanism

* 기존의 Potential Outcome Framework, Quasi-experiment에서 중요한 것은 Research design. Research design은 결국 Treatment Assignment Mechanism (어떻게 Treatment를 주고, 어떻게 두 집단을 구분할 지) 
* 그래서 도구변수를 하나의 Treatment Assignment Mechanism으로 해석하는 것 
* 아래 그림을 보면, x 축은 도구변수 값.  y축은 treatment 여부 (2 by 2). 도구변수로 treatment를 유도(induce)하는 Treatment Assignment Mechanism이라고 보는 것. 그렇지만, 도구변수로 treatment를 유도하더라도 모두가 도구변수에 의해 treatment를 받을 이유는 사실상 없음. 그렇게 본다면, 도구변수가 어떻든 간에 늘 treatment를 받는 사람들도 있을 것이고(always takers), 항상 treatment를 받지 않는 사람들도 있을 것(never takers). 또한, 우리의 의도대로 도구변수가 1이면 treatment를 받고, 도구변수가 0이면 treatment를 받지 않는(control group이 되는) 도구변수에 순응하는 compliers 집단이 있을 것. 그 반대의 defiers 집단도 있을 것  
<p><img src="/assets/images/defier_complier.png" title="average treatment effects on the compliers"/></p>
그림 출처: 인과추론의 데이터과학. (2022, June 18) 

* 따라서, 도구변수에 의해서 Treatment가 움직이는 compliers, defiers에서의 효과를 도구변수에 의해 야기되는 treatment의 인과적인 효과라고 볼 수 있음

* 그런데, 우리는 도구변수가 0을 통해서 control을 주고 싶고, 도구변수 1을 통해서 treatment를 주고 싶은데, 그 반대로 작용하는 defiers(control group과 treatment group이 섞임)는 사실상 우리가 추론하고자 하는 인과관계를 방해하는 요소임
* 그래서, 아래 그림과 같이, defiers는 없다고 가정함
> : Monotonicity assumption  
<p><img src="/assets/images/Monotonicity assumption.png" title="Monotonicity assumption"/></p>
그림 출처: 인과추론의 데이터과학. (2022, June 18)

* 이러한 가정 하에, 우리가 도구변수를 통해 추론할 수 있는 것
> : compliers라는 sub-population에서의 causal effect. 그래서 전체 Average Treatment Effect가 아닌, Local Average Treatment Effect

### 3. Illustrative Example of LATE

* 베트남전에 참전한 경험이 평생의 소득에 어떠한 영향을 미치는지 분석한 연구 사례 (도구변수는 징병우선순위)

    * 단순 ols 분석 결과; 2% 정도 차이를 보임
    * 도구변수를 활용한 2SLS 분석 결과; 23% 정도 차이를 보임
즉, 전쟁에 참전하면 평생 소득에 있어 23%정도가 낮아짐. 따라서, 정부에서는 보상정책을 시행할 필요가 있음 

* 그런데, 해석을 어떻게 해야 할까? 이게 미국에서 군대를 갈 수 있는 모든 성인 남성에게 다 적용되는 효과인가? 아니면 어떤 특정 사람들한테 적용되나? 23%라는 인과적 효과가 정확히 어디로부터 나온 수치일까? 
* 징병우선순위가 낮을지라도 그 순위와 관계없이 무조건 참전하지 않았을 never-taker가 포함되어 있고(아래 그림의 x 표시된 식의 좌변의 두번째 항), 징병우선순위가 높을지라도 그 순위와 관계없이 무조건 참전했을 always-taker(아래 그림의 x 표시된 식의 좌변의 첫번째 항)가 포함되어 있기 때문에, 아래 수식은 적합하지 않음  
<p><img src="/assets/images/2sls_estimate.png" title="2SLS estimate"/></p>
그림 출처: 인과추론의 데이터과학. (2022, June 18)

* 도구변수 0,1이 random assignment이므로, z=0이든 1이든, compliers, always-takers, never-takers의 비율은 동일함 (애초에 defiers는 없다고 가정). 그래서 도구변수가 1일 때, never-takers의 비율을 구할 수 있는데, 1915 / (1915+865). 마찬가지로 도구변수가 0일 때, always-takers의 비율을 구할 수 있는데 1372 / (5948 + 1372). 그렇다면 1에서 두 비율을 빼면 compliers의 비율을 구할 수 있게 됨  
<p><img src="/assets/images/compliers.png" title="compliers"/></p>
그림 출처: 인과추론의 데이터과학. (2022, June 18)

* 즉, 약 12%의 사람이 징병우선순위가 높아서 이에 순응하여 참전한 사람들로 추정됨

* 그 후 위 사진과 같이 weight를 줘서 계산ㄴ하기 
* compliers에서 도구변수가 1이어서 참전한 사람의 평균 소득과 도구변수가 0이어서 참전하지 않은 사람의 평균 소득을 구할 수 있고, 두 값의 차이가 약 -0.2336. 
* 여기서 도구 변수의 추정이 어떠한 의미를 갖는지 정리하자면, 전쟁 참전으로 인해 약 23% 정도 소득 손해를 보는 사람들은 모든 미국인 성인 남성이 아니라, 징병우선순위에 순응한 사람들. 즉, compliers에 대한 인과효과가 23%라는 것  

> What we can learn from the IV estimates is the causal effect in the subpopulation of compliers. 

* 그러나, 한계점도 존재함. 도구변수에 대한 순응 여부가 compliers이기 때문에, 도구변수가 달라지면 또 다른 도구변수에 순응하는 compliers는 달라질 수 밖에 없음

* LATE의 가장 큰 한계; 도구변수에 specific하기 때문에, 다른 context로 일반화하기가 쉽지 않음. 그래서 특정 context에 국한해서 설명해야 한다는 한계가 있음 

* When LATE becomes ATET and ATE
    * 특정 가정 하에서는 LATE(처치 그룹에서의 효과) 혹은 ATE(전체 집단에서의 효과)가 될 수도 있음
    * 만약 Always-taker가 없다면? Treatment 그룹에 complier만 존재하게 됨
    > LATE = ATET
    * 만약 Never-taker가 없다면? Control 그룹에 complier만 존재하게 됨. 예를 들어, 의무교육이 도입되면 모두에게 적용되므로 Never-taker 없음
    > LATE = ATEU
    * 만약 Always-taker, Never-taker가 없다면? 
    > LATE = ATET



### References 
인과추론의 데이터과학. (2022, June 18). [Bootcamp 4-2] 인과추론 관점에서의 도구변수 [Video]. YouTube. [https://www.youtube.com/watch?v=nRMZ7a4Ah8E&list=PLKKkeayRo4PV_6-nbBgmUNOSpG1OO49M3&index=13][1]

[1]: https://www.youtube.com/watch?v=nRMZ7a4Ah8E&list=PLKKkeayRo4PV_6-nbBgmUNOSpG1OO49M3&index=13