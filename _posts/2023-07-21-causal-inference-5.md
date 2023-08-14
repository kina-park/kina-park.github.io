---
title: "인과추론 관점에서의 회귀분석"
layout: single
classes: wide
toc: true
categories: 
- Causal inference
---

### 1. 어떤 종류의 Selection bias를 다룰 수 있을까? 

* Selection on Unobservables Strategies
    * Randomized Controlled Trial, Quasi-Experiment, Instrumetal Variable
    * Random assignment, 적절한 research design을 통해 관찰 가능하지 않은 교란 요인들에 의한 selection biaS 문제까지 해결하고자 하는 전략. 아래 전략보다 좀 더 powerful 
* Selection on Observables Strategies
    * Designed Regression / Matching 
    * 관찰 가능한 변수들에 의해서만 처치 집단과 통제 집단이 선택된다는 가정 하에, selection bias를 모두 설명하고자 하는 전략 

### 2. 어떻게 관찰 가능한 변수들에 의해서만 통제, 처치 집단의 균형을 맞출 수 있을까? 
* Regression adjustment  
: 통제 변수의 활용을 통해서 selection bias를 설명하고자 함
(2)	Matching 
: 두 집단이 서로 비교 가능할 수 있도록, 관찰 가능한 변수들의 값이 서로 유사한 데이터들끼리 매칭
(3)	Weighting 
: 처치를 받을 확률의 역수 만큼을 각 데이터에 가중치를 부여함으로써 결과적으로 random assignment와 비슷하게 처치를 받을 확률이 같아지도록 만드는 방법
