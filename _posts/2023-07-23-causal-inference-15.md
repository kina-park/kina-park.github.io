---
title: "[5-1] 인과 그래프"
layout: single
toc: true
categories: 
- "[관심분야 공부] Causal inference"
breadcrumbs: false
---

### 1. Causal Graph (Diagram) 

* 베이지안 네트워크(Directed Acyclic Graph를 조건부 확률로 도식화한 것)가 causal graph를 표현하는 도구이며, structural causal model을 분석하는 기본 도구임  
<p><img src="/assets/images/causal_graph.png" title="Causal Graph (Diagram)"/></p>
그림 출처: 인과추론의 데이터과학. (2022, June 19) 

### 2. Relationship Types in Causal Graph
* 변수들 간의 관계는 아래와 같이 크게 네 가지로 구분 가능함 
* Confounder 변수는 treatment에 영향을 미침, 따라서 pre-treatment variable
* 반대로, collider는 treatment의 영향을 받아서 그 결과로 나타남, 따라서 post-treatment variable
* 결국, 이러한 관계 속에서 causal effect가 어떻게 발현되며, causal effect를 방해하는 다른 요인들 간의 상관관계는 어떻게 나타나는지 분석하는 것이 핵심  
<p><img src="/assets/images/causal_graph_type.png" title="Relationship Types in Causal Graph"/></p>
그림 출처: 인과추론의 데이터과학. (2022, June 19)  

### 3. Association in Causal Graph
* Graph 상에서 정보의 흐름을 통해 변수들 간의 상관관계를 이해할 수 있음. 즉, Graph 상에서 같은 정보를 공유하고 있다면, association 혹은 correlation이 있다고 간주할 수 있음 
* 예컨대, X와 Y는 공통적으로 B로부터 연결됨. X와 Y는 상관관계가 있다고 볼 수 있음. ‘X-A-B-Z-Y’는 하나의 path
* X가 W를 통해서 Y로 가는 것이 causal effect. ‘X-W-Y’가 바로 causal path. 이를 제외한 나머지 모든 path를 backdoor paths라고 부름 
* 즉, backdoor paths를 모두 차단할 수 있다면, causal effect를 구할 수 있음  
<p><img src="/assets/images/backdoorpath.png" title="Association in Causal Graph"/></p>
그림 출처: 인과추론의 데이터과학. (2022, June 19)  

* 두 변수 간의 정보의 흐름이 막혀 있는 경우: d-separated
* 반대로, path가 연결되어 있다면 d-connected
* 아래 그림에서 보면, 예를 들어 A를 차단하면 X – Y -> d-separated 되는 반면, C의 경우 이를 차단할지라도 여전히 X – Y 연결되므로, d-connected  
<p><img src="/assets/images/d_separ_conn.png" title="d-separated & d-connected"/></p>
그림 출처: 인과추론의 데이터과학. (2022, June 19)  

* 그런데, 만약 A와 D를 차단하면 d-connected. 왜 그럴까? 

### 4. Association in Causal Graph by Structure
<p><img src="/assets/images/media_confo_collid.png" title=" Association in Causal Graph by Structure"/></p>
그림 출처: 인과추론의 데이터과학. (2022, June 19)  

* 만약, Mediator, Confounder, Collider를 conditioning해서 path를 차단하면 어떻게 될까? 
    * conditioning 한다는 것을 직관적으로 이해하자면, 회귀분석에서의 control variable, 매칭 전략  
    <p><img src="/assets/images/conditioning.png" title="conditioning"/></p>
그림 출처: 인과추론의 데이터과학. (2022, June 19) 

* Mediator 
    * Causal effect를 찾기 위해서 일반적으로 mediator는 conditioning하면 안됨. 경우에 따라서, indirect causal effect가 아닌 direct causal effect만 구하고 싶은 특수한 경우에는 mediator를 차단하는 것이 맞음
* Confounder 
    * confounder를 차단하면, 이는 대표적인 backdoor path이기 때문에, causal effect를 구하기 위해서는 꼭 conditioning해야 함 

* Collider 
     * X와 Y는 원래 아무런 관련이 없었지만, collider Z를 conditioning함으로써 즉, 동일한 Z라는 결과를 내기 위해 X와 Y를 조정함으로써 어떠한 일련의 관계가 생기게 됨. 따라서, 기존에 없던 backdoor path가 생기면서 문제가 될 수 있으므로, collider는 차단하면 안 됨


### References 
인과추론의 데이터과학. (2022, June 19). [Bootcamp 5-1] 인과 그래프 [Video]. YouTube. [https://www.youtube.com/watch?v=nMweRDcooXI&list=PLKKkeayRo4PV_6-nbBgmUNOSpG1OO49M3&index=16][1]

[1]: https://www.youtube.com/watch?v=nMweRDcooXI&list=PLKKkeayRo4PV_6-nbBgmUNOSpG1OO49M3&index=16