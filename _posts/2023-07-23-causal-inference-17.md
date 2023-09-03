---
title: "[5-3] 구조적 인과모형"
layout: single
toc: true
categories: 
- Causal inference
---

### 1. Structural Causal Model 
* Probabilistic Causal Mechanisms
* Causal Graph/Diagram이 일반적인 design-based approach에서 가능하지만, 직접적으로 인과추론에 활용되는 것은 단연 Structural Causal Model 

* 좁은 의미에서의 Structural Causal Model
    * Data Generation Process를 표현하는 causal mechanism에 대한 수학적 모델 
* 현실에서 관찰 가능한 것은 Structural Causal Model로 표현되는 data generation process에 의해서 실제로 발생한 데이터. 이에 대한 여러가지 확률 분포를 관찰할 수 있음(Judea pearl)
    1. Associational distribution: 관찰된 데이터에서 볼 수 있는 확률 분포
    2. Interventional distribution: 어떤 변수에 대한 행동을 취했을 때 나타나는 확률 분포. 인과관계에 대한 확률 분포 
    3. Counterfactual distribution: ‘만약 다른 행동을 취했다면 어떻게 됐을까’를 나타내는 확률 분포
* 하위 단계 distribution만 갖고서는 상위의 distribution을 구할 수 없음. 데이터 그 이상이 필요

* Judea Pearl’s Causal Hierarchy 
    1. Level 1: Associational or Observational – 시스템을 그대로 두고 관측 (관찰 데이터) 
    2. Level 2: Interventional or Experimental – 적극적으로 시스템을 변형 (실험 데이터) 
    3. Level 3: Counterfactual – 반사실  
    <p><img src="/assets/images/judea pearl.png" title="Judea Pearl’s Causal Hierarchy"/></p>
그림 출처: 인과추론의 데이터과학. (2022, June 19) 

### 2. Causal Inference with SCM
* Causal inference는 데이터에서 관찰되는 단순 확률분포가 아니라, 어떤 행동을 취했을 때 나타나는 결과이므로, Interventional distribution 혹은 Counterfactual distribution을 구해야 함. 이를 구하기 위해 이용하는 것이 바로 causal graph
* Data generation process에 대한 완벽한 causal model은 현실에서 우리가 알 수 없지만, 적어도 structural constrain하에서 변수들을 어떠한 관계의 graph로 표현할 수 있는지는 알 수 있음. 제한된 정보를 나타내는 Graphical model을 활용해서 하위단계의 정보로 상위단계의 정보를 추정하고자 하는 것 -> causal graph에서의 인과추론 
* associational distribution + 제한적으로 구한 causal graph -> interventional distribution  
<p><img src="/assets/images/structural constraints.png" title="Graphical model"/></p>
그림 출처: 인과추론의 데이터과학. (2022, June 19) 

### 3. Distribution of Causal Effect Using do-operator
* 주어진 causal graph를 그대로 분석하면 backdoor path로 인해 causal effect를 구할 수 없으므로, ‘우선 이론적으로 backdoor path도 없는 상태에서 계산한 인과적인 효과를 인과효과라고 가정하자’ -> do-operator (treatment의 부모노드 즉, treatment에 영향을 줄 수 있는 모든 요인의 효과를 무시하자)
* 아래에서 왼쪽의 그림을 보면, confounder C로 인한 backdoor path 때문에 P(Y|X)는 인과적 효과라고 볼 수 없음. 만약에, confounder 없이 이론적으로 X가 실제 원인이어서 나타나는 Causal effect가 있다면 어떨까? X에 do-operator를 적용하면 아래 오른쪽의 그림처럼 non-causal effect를 차단하는 방식으로 정의할 수 있음
* do-operator는 실제로 action이 있는 개입이라기 보다는 treatment 노드의 부모 노드(treatment 변수에 영향을 주는 다른 요인들의 효과)를 배제하자는 이론적인 개입 
* 따라서, do-operator를 적용한non causal effect의 backdoor path들이 모두 차단되므로, P(Y|do(X)) = causal effect
* 그리고 이러한 distribution을 interventional distribution이라고 부름 
* 그런데, P(Y|do(X))는 이론적 개념이기 때문에 실제로 계산할 수는 없음. 그래서 P(Y|do(X))를 추정하기 위해서는 수학적으로 계산할 수 있는 Conditional probability, conditional distribution 형태로 변환해야 함. 이 과정이 바로 Identification  
<p><img src="/assets/images/do operator.png" title="do-operator"/></p>
그림 출처: 인과추론의 데이터과학. (2022, June 19)  

* 다시 정리하자면, P(Y|do(X))는 이론적으로 구할 수 있는 causal effect. 실제로 직접 구할 수 있는 distribution은 아님 
* 결국 데이터에서 구할 수 있는 associational distribution 혹은 conditional distribution을 통해서 interventional distribution을 어떻게 구할 수 있는지가 관건즉, interventional distribution을 conditional distribution으로 변환할 수 있어야 하고, 이러한 변환의 과정을 identification이라고 함 
* 만약, do-operator를 통해서 구한 interventional distribution을 conditional distribution으로 변환할 수 없는 경우 -> non-identifiable. 구할 수 있는 경우 -> identifiable
* 변환하는 과정에서 필요한 rule의 집합을 do-calculus 라고 정의함 (Judea Pearl)  
<p><img src="/assets/images/do-calculus.png" title="do-calculus"/></p>
그림 출처: 인과추론의 데이터과학. (2022, June 19)  

* 위 그림에서 왼쪽 그래프는 do-operator 적용 전, 오른쪽 그래프는 do-operator 적용 후
* 한편, 하위 정보로 상위의 정보를 구할 수 없다는 것(causal hierarchy theorem)에 한 가지 예외가 있는데, 이것이 바로 random assignment
* Treatment를 받을지 말지 여부가 어떤 요인에도 영향 받지 않고, 오로지 예를 들면 동전던지기를 통해서 결정되므로, do-operator를 실제 데이터에 적용하는 유일한 방법이 random assignment
* 즉, random assignment는 potential outcome framework에서 뿐만 아니라, structural model에서도 여전히 gold standard (물론 관점은 다름)  
<p><img src="/assets/images/random assignment.png" title="random assignment"/></p>
그림 출처: 인과추론의 데이터과학. (2022, June 19)  

### 4. Identification of Causal Effect
* random assignment가 불가능한 상황에서, 이론적으로 만약 random assignment가 있었다면 있었을 do-operator를 통해서 계산한 이론적인 interventional distribution이 있고, 이를 실제로 데이터를 통해서 계산할 수 있는 associational, conditional distribution으로 변환하는 과정이 identification
* 어떤 변수를 conditioning해야만 backdoor paths를 전부 차단할 수 있는지를 정의한 것이 -> Backdoor criterion 
* X와 Y 간의 causal relationship/causal association을 제외한 모든 backdoor paths를 막을 수 있는 변수들의 집합 -> Backdoor criterion 
* 따라서, 아래 그림에서 W2, C는 Backdoor criterion을 만족시킴 
* 그리고 위와 같은 변수들을 통제 변수로 투입하거나, 매칭하거나, IPW등을 통해 conditioning 하는 것을 Backdoor adjustment라고 함  
<p><img src="/assets/images/backdoor adjustment.png" title="Backdoor adjustment"/></p>
그림 출처: 인과추론의 데이터과학. (2022, June 19)  

### 5. Identification of Causal Effect Using do-calculus (with Graph)

* Backdoor criterion을 넘어서 모든 graph 상황에서 적용 가능한 identification을 위한 graph 법칙 
> do-calculus  
<p><img src="/assets/images/identification_do-calculus.png" title="Identification of Causal Effect Using do-calculus"/></p>
그림 출처: 인과추론의 데이터과학. (2022, June 19)  

### 6. Identification of Causal Effect  
<p><img src="/assets/images/query.png" title="query"/></p>
그림 출처: 인과추론의 데이터과학. (2022, June 19)  

* 우리가 이론적으로 구해야 하는 do-operator에 기반한 interventional distribution이 있고, 이것이 우리가 구해야 하는 query 
* 그리고 이를 do-calculus와 우리에게 주어진 causal graph를 활용해서 실제 계산할 수 있는 형태의 distribution으로 변환하는 것이 identification 
* Identification이 가능한지 여부를 do-calculus를 통해 판단할 수 있게 됨. do-calculus가 모든 graph에서 identifiability를 판단할 수 있는 complete rule이라고 연구에서 증명된 바 있음 
* 그렇다면 identification(실제 데이터를 통해 계산할 수 있는 distribution의 형태로 만드는 것)의 다음 단계는? -> 실제 데이터를 통해 추정하는 것 estimation!  
<p><img src="/assets/images/identification_estimation.png" title="Estimation engine"/></p>
그림 출처: 인과추론의 데이터과학. (2022, June 19)  

* Identification 가정을 통해 도출된 probability 식을 실제 데이터를 통해 추정함. Estimation을 위한 새로운 방법이 많이 개발되고 있음(더블 머신러닝 등) 

### 7. Potential Outcome Framework vs Structural Causal Model
* (Random assignment가 불가능할 때) potential outcome framework는 identification, causal effect를 추정할 수 있는지를 어떻게 접근하는가? -> selection process를 설명할 수 있는 research design을 고안함으로써 identification을 하는 접근 
* Structural causal model은 Causal graph하에서 do-calculus 등을 활용해서 interventional distribution을 우리가 실제로 계산 가능한 associational distribution으로 변환하는 과정을 통해서 identification을 하는 접근
* potential outcome framework는 주로 social science 분야에서, Structural causal model은 computer science 분야에서 많이 활용되고 있음. 

    1. Difference - Manipulability
    * Treatment variable에 대한 manipulability에 대한 차이 
    * Potential outcome framework는 기본적으로 manipulability를 가정하고, experimental mindset을 갖고 research design을 수행함. “No Causation without Manipulation”
    * 반면, Causal Graph, Structural causal model에서 manipulability은 아무런 역할을 하지 않음 

    2. Difference - Causal Structure / Knowledge
    * Causal Graph, Structural causal model에서는 아래와 같이 causal knowledge가 필요함 
    * Potential outcome framework에서는 causal knowledge를 무조건 요구하는 것은 아님

### 8. Policy-Based vs Knowledge-Based Causation 

* Manipulability, Causal Structure/Knowledge에 대한 활용 여부가 굉장히 중요한 차이
* Causation의 주된 목적이 Policy-Based Causation인지, 혹은 Knowledge-Based Causation인지에 따라 주로 구분됨 
* Social science 분야에서는 주로 policy-based causation을 지향함
* Computer science 분야에서는 주로 Knowledge-Based Causation을 지향함. 알고리즘, 프로그램을 개발할 때 시스템화 할 수 있는 knowledge 기반의 causation이 필요함. AI, ML 분야에서 Structural causal model 연구가 시작됨 

### References 
1. 인과추론의 데이터과학. (2022, June 19). [Bootcamp 5-3] 구조적 인과모형 [Video]. YouTube. [https://www.youtube.com/watch?v=BKJAinqXXjQ&list=PLKKkeayRo4PV_6-nbBgmUNOSpG1OO49M3&index=18][1]

2. KAIA 한국인공지능학회. (2023. February 7). Causal Inference / Efficient Reinforcement Learning. YouTube. [https://www.youtube.com/watch?v=IGWyYOLvAc8][2]


[1]: https://www.youtube.com/watch?v=BKJAinqXXjQ&list=PLKKkeayRo4PV_6-nbBgmUNOSpG1OO49M3&index=18

[2]: https://www.youtube.com/watch?v=IGWyYOLvAc8