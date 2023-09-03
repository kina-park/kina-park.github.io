---
title: "[3-3] 이중차분법"
layout: single
toc: true
categories: 
- Causal inference
---

### 1. Difference in Differences(DID)

 
<p><img src="/assets/images/DID.png" title="Difference in Differences(DID)"/></p>
그림 출처: 인과추론의 데이터과학. (2022, June 17)

### 2. Identification Assumption for DID
* DID에서 요구되는 가정: 통제 집단과 처치 집단에서 처치가 없을 때의 시간에 따른 변화가 얼마나 비교(parallel) 가능한지. 그러나, 엄밀히 말하면 parallel trend assumption은 불가함. 애초에 parallel trend assumption이라는 것은 counterfactual에 대한 가정. 즉, treatment가 있는 상황에서, treatment가 없었다면 있었을 counterfactual이 control group과 parallel할 것이라는 가정이 있었던 것이기 때문(즉, 이것은 counterfactual에 관한 가정을 전제로 하고 있음)  
<p><img src="/assets/images/did_pararell.png" title="parallel trend assumption"/></p>
그림 출처: 인과추론의 데이터과학. (2022, June 17)

* 그런데 만약, 처치를 받은 시점이 샘플별로 상이하다면? 누군가는 2020년도에 treatment를 받고, 누군가는 2021년도에 treantment를 받는다면? 위와 같이 어떤 공통된 시점으로 구분해서 시각화 하는 것이 쉽지 않을 것임. 그럴 땐 아래의 방법이 유용함: Relative Time Model (Leads-and-Lags Model)

### 3. Relative Time Model (Leads-and-Lags Model)
* Treatment가 발생한 시점을 기준으로 relative time dummies를 만들어서 모델에 투입. 
아래 논문에서 보면, 우버 도입이 treatment이며, 우버 도입 이전 기간에 통제 집단과 처치 집단의 relative time dummy를 보면, 통계적으로 유의하지 않음(parallel trend assumption)  
<p><img src="/assets/images/dummy.png" title="relative time dummies"/></p>
그림 출처: 인과추론의 데이터과학. (2022, June 17)

### 4. Various Cases of Difference-in-Differences

1. 패널데이터에서 각각의 집단에 대해서 시간에 따라 변하지 않는 변화를 설명하기 위해서 unit fixed effects, 집단에 관계없이 특정 시점에 공통적으로 영향이 있는 요인들을 고려하기 위해 time fixed effect. 아래 식에 표시한 두 항  
<p><img src="/assets/images/fixed_effects.png" title="fixed effects"/></p>
그림 출처: 인과추론의 데이터과학. (2022, June 17)

두 항은 자동으로 아래 화살표의 항을 흡수함. 그래서 맨 밑의 식과 같이 사실상 상호작용항만 남게 됨. 패널데이터에서는 fixed effects를 활용하는 것이 거의 필수

<p><img src="/assets/images/interaction_term.png" title="interaction term"/></p>
그림 출처: 인과추론의 데이터과학. (2022, June 17)

2. Treatment의 시점이 다 다른 경우
    * 이 경우 time fixed effects는 포함될 수 없음
    * 만약, 여기에 두 그룹의 구분이 treatment의 여부라면? 
    > time fixed effects는 포함될 수 없음
    
3. Treatment 시점 동일한데, 두 그룹의 그분은 treatment에 영향 받는 정도 차이에 따르는 경우

4. Treatment 시점 다른데, 두 그룹의 구분은 treatment에 영향 받는 정도 차이에 따르는 경우
    * 삼충 차분으로도 볼 수 있음

### References 
인과추론의 데이터과학. (2022, June 17). [Bootcamp 3-3] 이중차분법 [Video]. YouTube. [https://www.youtube.com/watch?v=yCeaZ9Ktk7g&list=PLKKkeayRo4PV_6-nbBgmUNOSpG1OO49M3&index=10][1]

[1]: https://www.youtube.com/watch?v=yCeaZ9Ktk7g&list=PLKKkeayRo4PV_6-nbBgmUNOSpG1OO49M3&index=10