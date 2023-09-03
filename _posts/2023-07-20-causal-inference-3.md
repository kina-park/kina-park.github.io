---
title: "[2-1] 잠재적 결과 프레임워크"
layout: single
toc: true
categories: 
- Causal inference
---

### 1. 잠재적 결과 프레임워크(Potential Outcome Framework)

* 어떤 특정 원인(treatment)의 인과적인 효과를 잠재적 결과의 차이로서 정의하는 방식 (일상적 예시. 그때 내가 다른 결정을 했다면 잠재적으로 지금의 결과가 다를까?) 
* Causal effect = (Actual Outcome for treated if treated) – (Potential Outcome for treated if not treated) 
* 그러나, Potential Outcome for treated if not treated 즉, counterfactual은 현실에서 관찰할 수 없으며, 이에 따라 ITE(Individual treatment effect)는 구할 수 없음. 
* 다만, ATE(Average Treatment Effect) 즉, 처치를 받은 집단과 그렇지 않은 집단 간의 평균 차이를 통해 평균적인 인과적 효과는 추정해 볼 수 있음. 잠재적 결과 프레임워크의 주된 관심사는 ATE 
* 그러나, 이상적인 counterfactual과 control group 간의 차이로부터 인과추론의 근본적 문제가 발생함

<p><img src="/assets/images/counterfactual.png" width="80%" height="80%" title="Potential Outcome Framework"/></p>
사진 출처: 인과추론의 데이터과학. (2022, June 14). [Bootcamp 2-1]점재적결과 프레임워크 [Video]. YouTube. https://www.youtube.com/watch?v=M7e9_i0VNAI&list=PLKKkeayRo4PV_6-nbBgmUNOSpG1OO49M3&index=4

* 비록 우리가 counterfactual은 알 수 없지만, 만약 control group을 적절하게 구성할 수 있다면 control group에서의 값을 counterfactual 값으로 대입해서 직접 비교할 수 있을 것이며, 반대로 treatment의 결과값을 control 그룹의 counterfactual 값에 대입해서 접근해 볼 수 있음.
* 따라서, 어떤 관점에서 보면 잠재적 결과 프레임워크는 인과추론 문제를 결측치(missing value) 관점에서 본다고 할 수 있음. 

* 결측치 관점에서 causal inferenc의 조건
    * Ignorability: 처치 집단과 통제 집단 간 차이를 무시해도 됨
    * Exchangeability: 처치 집단과 통제 집단 간 값을 교환할 수 있음
* 선택 편향(selection bias), causal graph 관점에서 causal inference의 조건 
    * Unconfoundedness: confounder(treatment에도 영향을 미치고, 결과에도 영향을 미치는 외부 요인 / 교란 요인)가 없어야 함
* 통계적 관점에서 causal inference의 조건 
    * Exogeneity: 외생성 
    * 통계적 관점에서는 근본적으로 counterfactual에 가장 가까운 control group을 구성해서, counterfactual을 대신하는 접근 

### 2. Selection Bias, Causal Effect

* 반려동물을 키우는 사람과 그렇지 않은 사람의 우울증을 비교함으로써 인과관계를 추론하는 것은 적절치 않음. 반려동물을 키운다는 사실을 제외하고 나머지 요인들이 그나마 최대한 비슷한 사람들끼리 묶어서 그들 간 비교를 한다면, 반려동물 키우는 것의 인과효과를 추정할 수 있다고 보는 것. 애초에 반려동물을 키우고자 하는 성향이 있을 수가 있음. 예를 들어, 1인가구, 노인가구 등 

* 즉, 반려동물 키우는 여부를 제외하고는 통제 집단과 처치 집단이 전부 다 비슷해야 비교가 가능한 지 여부를 확인해야 함 

* 처치 그룹에서 처치가 없었다면 있었을 잠재적 결과인 counterfactual과 현실에서 실제 처치를 받지 않은 통제 집단 간의 차이를 ‘선택 편향(selection bias)’이라고 부름. 이는 처치를 받을지 말지 사람들이 실제로 선택을 하는 데서 오는 편향이며, 선택 편향을 야기하는 교란 요인을 confounding factor라고 부름


* Observed effect of the treatment = Causal Effect + Selection Bias 
* Causal Effect = (1) - (2) = (Outcome for treated if treated; Real) - (Outcome for treated if not treated; Counterfactual) 
* Selection Bias = (2) - (4) = (Outcome for treated if not treated; Unreal) - (Outcome for untreated if not treated; Real) 

  ||TREATMENT O|TREATMENT X|
  |:------:|:---:|:---:|
  |TREATMENT 집단|(1) Real|(2) Counterfactual|
  |CONTROL 집단|(3) Unreal|(4) Real|


### 3. Ceteris Paribus

* 그래서, potential outcome framework하에서 selection bias를 없애기 위한 인과추론의 가장 중요한 원칙은 단연 Ceteris Paribus(Comparable Control Group) 조건. ‘처치를 받았다는 사실을 제외하고 모든 요인이 모두 동일해서 비교 가능하다‘는 조건. 만일, Ceteris Paribus를 충족하는 control group이 있다면, 이상적인 counterfactual 집단을 대신할 수 있음. 

### References
인과추론의 데이터과학. (2022, June 14). [Bootcamp 2-1]점재적결과 프레임워크 [Video]. YouTube. [https://www.youtube.com/watch?v=M7e9_i0VNAI&list=PLKKkeayRo4PV_6-nbBgmUNOSpG1OO49M3&index=4][1]

[1]: https://www.youtube.com/watch?v=M7e9_i0VNAI&list=PLKKkeayRo4PV_6-nbBgmUNOSpG1OO49M3&index=4

