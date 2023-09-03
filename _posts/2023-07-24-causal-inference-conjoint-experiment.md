---
title: "컨조인트 분석 & 인과추론"
toc: true
toc_sticky: true
toc_label: "Table of Contents"
toc_icon: "bookmark"
categories: 
  - Conjoint experiment & Causal inference
sidebar_main: true
breadcrumbs: false
---

## What attributes causally increase or decrease korean public's support for asylum seekers?

### 1. Introduction & Literature Review 

* In 2018 when it was known that more than 500 asylum seekers from Yemen in Jeju applied for refugee status, it caused a heated controversy over the acceptance of asylum seekers and refugees in Korean society. With the elevated concerns about refugees in Korean society, yet, few studies have systematically investigated Koreans’ attitude formation towards refugees or asylum seekers. 

* In Europe, however, a significant body of research that examines the factors influencing public attitudes toward refugees or asylum seekers has been accumulated. Specifically, recent studies have shown that public attitudes toward refugees or asylum seekers differ based on the characteristics of these individuals.(Hager & Veit ,2019; Von Hermanni & Neumann ,2018). 

* What factors may explain variation in preference and support for asylum seekers? In line with recent experimental
studies, we focus on seven key asylum seeker characteristics: gender, age, marital status & children in family, country of origin, religion, education level, reason for flight. These attributes cover integrated threat theory, deservingness attributions, gender considerations, humanitarian concerns that have been prominently addressed in previous studies exploring the factors determining public's attitudes towards refugees and asylum seekers(Bansak et al., 2016; Landmann et al., 2019; Stephan et al., 1999). 

### 2. Conjoint analysis & Causal Inference

* Survey experiments are a gold standard for causal inference, specifically, conjoint analysis enables the estimation of the causal effect of multiple attributes when making multi-dimensional choices or preferences(Hainmueller et al., 2014). Therefore, it has been used for assessing the impact of candidate, immigrant attributes(Hainmueller & Hiscox, 2010; Hainmueler & Hopkins, 2012) in social scientific research. 

* Hainmueller et al. (2014) proposed "a causal estimand that can be nonparametrically identified and easily estimated from conjoint data using a fully randomized design". They use the potential outcome framework of causal inference(Neyman, 1923; Rubin, 1974) for conjoint analysis and define "a causal quntity of interest, the average marginal component effect(AMCE)"(Hainmueller et al, 2014, p.3).  
<p align="center"><img src="/assets/images/amce.png" width="60%" height="60%" title="AMCE"/></p>

### 3. Research Design

* In this study, respondents were repeatedly shown paired conjoint 10 times, randomizing profile's gender, age, marital status & children in family, country of origin, religion, education level, reason for flight(full factorial design). Subsequently, they were forced to choose one of the two applicants(choice-based conjoint analysis) and to give a numerical rating to each profile that represents their degree of support for the profile(rating-based conjoint analysis). The figure below shows an example of online survey.  
<p align="center"><img src="/assets/images/conjoint_example.png" width="60%" height="60%" title="example"/></p>

* The table below shows the list of possible attribute values(attributes and levels) in the conjoint experiment. In taking into account the level of values, the unique context of a divided nation like South Korea and recent global events such as the Afghan crisis were also considered. While there were 2,304 possible combinations to create a single virtual asylum seeker profile[^scala1], the final total number of possible combinations were reduced to 1,656. Profiles that do not exist in the real world were excluded[^scala2], as randomizing attribute combinations could lead to impossible or illogical profiles.  
  
  |Treatment attribute|Values|Number of cases|
  |:------:|:---:|:---:|
  |성별|남자, 여자|2|
  |나이|25, 40, 55|3|
  |혼인 및 자녀 여부|자녀 있음, 자녀 없음(기혼), 자녀 없음(미혼)|3|
  |출신 지역 및 국가|예멘, 아프가니스탄, 북한, 미얀마|4|
  |종교|이슬람, 기독교, 불교, 종교 없음|4|
  |교육 수준|대학교 학위 취득, 대학교 학위 없음|2|
  |한국 이주 이유|자국 내 전쟁 발생으로 인한 피난,<br>정치적 박해로 인한 망명,<br>한국 유학,<br>한국에서의 취업 기회|4|

### 4. Results 






### Code

```r
# 데이터 불러오기
library(readr)
asylum <- read_csv("asylum_choice_rating_survey.csv")
```

```r
# 데이터 요인, 수준 설정
asylum$gender      <- factor(asylum$gender, levels=c("여자","남자"))

asylum$age         <- factor(asylum$age, levels=c(25, 40, 55))

asylum$child_marry <- factor(asylum$child_marry, levels=c('자녀 있음','자녀 없음(미혼)','자녀 없음(기혼)'))

asylum$origin      <- factor(asylum$origin, levels=c('예멘','아프가니스탄','미얀마','북한'))

asylum$religion    <- factor(asylum$religion, levels=c('이슬람','불교','종교 없음','기독교'))

asylum$edulevel    <- factor(asylum$edulevel, levels=c('대학교 학위 취득','대학교 학위 없음'))

asylum$reason      <- factor(asylum$reason, 
                             levels=c('한국에서의 취업 기회','정치적 박해로 인한 망명','자국 내 전쟁 발생으로 인한 피난', '한국 유학'))
``` 

```r
# attribute_list 생성
attribute_list <- list()

attribute_list[["gender"]] <- c("여자","남자")
attribute_list[["age"]] <- c(25, 40, 55)
attribute_list[["child_marry"]] <- c('자녀 있음','자녀 없음(미혼)','자녀 없음(기혼)')
attribute_list[["origin"]] <- c('예멘','아프가니스탄','미얀마','북한')
attribute_list[["religion"]] <- c('이슬람','불교','종교 없음','기독교')
attribute_list[["edulevel"]] <- c('대학교 학위 취득','대학교 학위 없음')
attribute_list[["reason"]] <- c('한국에서의 취업 기회','정치적 박해로 인한 망명','자국 내 전쟁 발생으로 인한 피난', '한국 유학')
```

```r
# constraint_list 생성
constraint_list <- list()

constraint_list[[1]] <- list()
constraint_list[[1]][["origin"]] <- c('예멘','아프가니스탄')
constraint_list[[1]][["religion"]] <- c('불교')

constraint_list[[2]] <- list()
constraint_list[[2]][["origin"]] <- c('북한')
constraint_list[[2]][["religion"]] <- c('이슬람')

constraint_list[[3]] <- list()
constraint_list[[3]][["origin"]] <- c('북한')
constraint_list[[3]][["reason"]] <- c('자국 내 전쟁 발생으로 인한 피난', '한국 유학')
```

```r
# design 
asylumdesign <- makeDesign(type='constraints', attribute.levels=attribute_list, constraints=constraint_list)
```

```r
# baseline 설정
baselines <- list()

baselines$gender<- "여자"
baselines$age <- "25"
baselines$child_marry <- "자녀 없음(기혼)"
baselines$origin<- "미얀마"
baselines$religion<- "종교 없음"
baselines$edulevel<- "대학교 학위 없음"
baselines$reason <- "한국에서의 취업 기회"
```




[^scala1]: Multiplying the levels of each attribute results in 2x3x3x4x4x2x4 = 2,304.
[^scala2]: The following combinations were excluded: 
출신 국가가 ‘예멘’이며 종교가 ‘불교’인 경우, 출신 국가가 ‘아프가니스탄’이며 종교가 ‘불교’인 경우, 출신 국가가 ‘북한’이며 종교가 ‘이슬람교’인 경우, 출신 국가가 ‘북한’이며 한국 이주 이유가 ‘자국 내 전쟁 발생 으로 인한 피난’인 경우, 출신 국가가 ‘북한’이며 한국 이주 이유가 ‘한국 유학’인 경우의 수 조합을 제외하였다.




