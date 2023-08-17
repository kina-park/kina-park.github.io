---
title: "인과추론 - 컨조인트 분석"
toc: true
toc_sticky: true
toc_label: "Table of Contents"
toc_icon: "bookmark"
categories: 
  - Conjoint experiment & Causal inference
sidebar_main: true
breadcrumbs: false
---

Conjoint analysis makes it possible to determine the causal impact of multiple attributes for making multi-dimensional choices (Hainmueller et al., 2014) and has been used for assessing the impact of candidate attributes (Breitenstein, 2019; Carnes and Lupu, 2016; Franchino and Zucchini, 2015; Kirkland and Coppock, 2018; Marx and Schumacher, 2018). We focus on seven key candidate characteristics: gender, ideology, issue focus, political experience, focus of representation, position on EU integration, and citizenship. These attributes cover both commonly employed candidate attributes and attributes unique to the EP elections, thereby making it possible to compare their impacts.

이주민, 난민에 대한 내국인의 태도를 결정하는 요인을 탐색한 기존의 선행연구(Bansak et al., 2016; Landmann et al., 2019; Stephan et al., 1999)에서 중요하게 다뤄진 변수들을 고려하고, 동시에 분단국가라는 한국의 특수한 상황과 아프가니스탄 사태와 같은 최근의 범세계적 상황을 고려하여 가상 인물 프로파일(profile)의 7개 속성과 각 속성의 수준을 선정하였다. 구체적인 내용은 아래 <표 1>에 정리하였다. 한 명의 가상 난민 프로파일을 생성하는데 가능한 경우의 수는 2,304[^scala1]개 였으나, 현실적으로 존재하기 어려운 일부 경우의 수 조합[^scala2]을 제외하여 총 경우의 수는 1,656개으로 진행하였다. 

[^scala1]: 각 속성의 수준을 곱하면 2x3x3x4x4x2x4 = 2,304이다.
[^scala2]: 출신 국가가 ‘예멘’이며 종교가 ‘불교’인 경우, 출신 국가가 ‘아프가니스탄’이며 종교가 ‘불교’인 경우, 출신 국가가 ‘북한’이며 종교가 ‘이슬람교’인 경우, 출신 국가가 ‘북한’이며 한국 이주 이유가 ‘자국 내 전쟁 발생 으로 인한 피난’인 경우, 출신 국가가 ‘북한’이며 한국 이주 이유가 ‘한국 유학’인 경우의 수 조합을 제외하였다.

* Attributes and levels of the conjoint analysis

  |Treatment attribute|Values|Number of cases|
  |:------:|:---:|:---:|
  |성별|남자, 여자|2|
  |나이|25, 40, 55|3|
  |혼인 및 자녀 여부|자녀 있음, 자녀 없음(기혼), 자녀 없음(미혼)|3|
  |출신 지역 및 국가|예멘, 아프가니스탄, 북한, 미얀마|4|
  |종교|이슬람, 기독교, 불교, 종교 없음|4|
  |교육 수준|대학교 학위 취득, 대학교 학위 없음|2|
  |한국 이주 이유|자국 내 전쟁 발생으로 인한 피난,<br>정치적 박해로 인한 망명,<br>한국 유학,<br>한국에서의 취업 기회|4|

