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

* In addition, it is also analyzed that variations between subgroups in the all respondents by including interaction terms to inspect whether the causal effect of an attribute depends on the characteristics of the respondents(Hainmueller et al., 2014), which is called an average component interaction effect (ACIE). It can show effect sizes for the different groups such as gender, ideology, islamophobia etc.

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

* First, we present the AMCEs for the seven attributes included in the conjoint experiment in the figure below (choice-based conjoint analysis).  
<p align="center"><img src="/assets/images/f1_amce.png" title="AMCEs"/></p>

* For gender, there is a negative impact for male asylum seekers(&beta;= -.0670, &rho;<.001) over female. With respect to age, there is a negative impact for 40 yeals old(&beta;= -.0430, &rho;<.001), 55 years old asylum seekers(&beta;= -.0737, &rho;<.001) over 25 years old.For children in family and marital status, there is a negative impact for asylum seekers who are married and have no kid(&beta;= -.0405, &rho;<.001), who never married and have no kid (&beta;= -.0429, &rho;<.001) over asylum seekers who have kids. For education level, there is a negative impact for asylum seekers without a university degree(&beta;= -.0383, &rho;<.001) as opposed to whom with a university degree. With respect to origin, there is a negative impact for asylum seekers from Yemen(&beta;= -.0684, &rho;<.001) and Afghanistan(&beta;= -.0670, &rho;<.001) over Myanmar, however, there is a positive impact for asylum seekers from North Korea(&beta;= .0889, &rho;<.001). For reason for flight, there is a negative impact for asylum seekers fleeing economic hardship(&beta;= -.1067, &rho;<.001), political persecution(&beta;= -.1066, &rho;<.001), study abroad(&beta;= -.0747, &rho;<.001) over fleeing from war. With respect to religion, there is a negative impact for muslim asylum seekers(&beta;= -.2039, &rho;<.001) over asylum seekers who don't have religion. 

* Second, we present whether these effects differ across respondents characteristic, especially islamophobia. With respect to religion of asylum seekers, there is a negative impact for muslim asylum seekers(&beta;= -.1217, &rho;<.001) over asylum seekers who don't have religion among respondents whose level of islamophbia is relatively low, however,  a negative impact for muslim asylum seekers(&beta;= -.2727, &rho;<.001) among respondents whose level of islamophbia is relatively high.  
<p align="center"><img src="/assets/images/acie.png" title="ACIE"/></p>


### Code (R)

```r
# 데이터 불러오기
library(readr)
asylum <- read_csv("asylum_choice_rating_survey.csv")

# 패키지 설치
install.packages('cjoint')
```

```r
# 데이터 요인, 수준 설정
asylum$gender      <- factor(asylum$gender, levels=c("여자","남자"))

asylum$age         <- factor(asylum$age, levels=c(25, 40, 55))

asylum$child_marry <- factor(asylum$child_marry, levels=c('자녀 있음','자녀 없음(미혼)','자녀 없음(기혼)'))

asylum$origin      <- factor(asylum$origin, levels=c('예멘','아프가니스탄','미얀마','북한'))

asylum$religion    <- factor(asylum$religion, levels=c('이슬람','불교','종교 없음','기독교'))

asylum$edulevel    <- factor(asylum$edulevel, levels=c('대학교 학위 취득','대학교 학위 없음'))

asylum$reason      <- factor(asylum$reason, levels=c('한국에서의 취업 기회','정치적 박해로 인한 망명',
                                                      '자국 내 전쟁 발생으로 인한 피난', '한국 유학'))
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

```R
# design 
asylumdesign <- makeDesign(type='constraints', attribute.levels=attribute_list, constraints=constraint_list)
```

```R
# baseline 설정
baselines <- list()

baselines$gender<- "여자"
baselines$age <- "25"
baselines$child_marry <- "자녀 있음"
baselines$origin<- "미얀마"
baselines$religion<- "종교 없음"
baselines$edulevel<- "대학교 학위 없음"
baselines$reason <- "자국 내 전쟁 발생으로 인한 피난"

# formula
# f1; choice-based conjoint analysis
f1 <- chosen_asylum ~  gender + age + child_marry+ origin + religion + edulevel+ reason

# f2; rating-based conjoint analysis
f2 <- rating ~  gender + age + child_marry+ origin + religion + edulevel+ reason
```

```R
# Run AMCE estimator using all attributes in the design
f1_amce <- amce(formula = f1, 
                data=asylum, 
                respondent.id="CaseID", 
                design=asylumdesign,
                baselines=baselines)
                                 
f2_amce <- amce(formula = f2, 
                data=asylum, 
                respondent.id="CaseID", 
                design=asylumdesign,
                baselines=baselines) 
```

```R
# Run AMCE estimator using all attributes in the design with interactions

# 1. 준비 작업 
# 범주형(factor) 변수(성별, 정치적 성향) level 설정
asylum$RES_gender_f[asylum$RES_gender == 1] <- "male"
asylum$RES_gender_f[asylum$RES_gender == 2] <- "female"
asylum$RES_gender_f <- factor(asylum$RES_gender_f, levels=c("male","female"))

asylum$RES_political_f[asylum$RES_political %in% c(1, 2)] <- "liberal"
asylum$RES_political_f[asylum$RES_political == 3] <- "moderate"
asylum$RES_political_f[asylum$RES_political %in% c(4, 5)] <- "conservative"
asylum$RES_political_f <- factor(asylum$RES_political_f, levels=c("liberal","moderate", "conservative"))

# 상호작용변수 이름 목록(성별, 나이, 정치적 성향, 인도주의적관심 성향, 이슬람공포증, 소득수준, 계층의식)
variable_names <- c("RES_gender_f", "RES_age", "RES_political_f", "RES_human_egal_1", "RES_islam_phobia", "RES_income", "RES_class")

# 2. ACIE
acie_list <- list()

for (variable in variable_names) {
  formula <- paste("chosen_asylum ~ gender:", variable, " + age:", variable, " + child_marry:", variable,
                   " + origin:", variable, " + religion:", variable, " + edulevel:", variable, " + reason:", variable, sep = "")
  amce_model <- amce(formula = as.formula(formula), 
                     data = asylum, 
                     respondent.id = "CaseID", 
                     design = asylumdesign,
                     baselines = baselines,
                     respondent.varying = variable)
  acie_list[[variable]] <- amce_model
}
```

```R
# Print summary 
summary(acie_list[[1]])

# plot
plot(acie_list[[1]], 
     xlab = "Change in Pr(Asylum seeker preferred for admission to South Korea)",
     xlim = c(-0.3, 0.3),
     plot.display="interaction")
```




[^scala1]: Multiplying the levels of each attribute results in 2x3x3x4x4x2x4 = 2,304.
[^scala2]: The following combinations were excluded: 출신 국가가 ‘예멘’이며 종교가 ‘불교’인 경우, 출신 국가가‘아프가니스탄’이며 종교가 ‘불교’인 경우, 출신 국가가 ‘북한’이며 종교가 ‘이슬람교’인 경우, 출신 국가가 ‘북한’이며 한국 이주 이유가 ‘자국 내 전쟁 발생 으로 인한 피난’인 경우, 출신 국가가 ‘북한’이며 한국 이주 이유가 ‘한국 유학’인 경우의 수 조합을 제외하였다.




