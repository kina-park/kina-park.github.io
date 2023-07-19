---
title: "한국인의 난민에 대한 태도 예측요인 탐색을 위한 다중회귀분석 및 상호작용효과 분석"
layout: single
classes: wide
toc: true
categories: 
  - 난민 연구
---


```python
from google.colab import drive
drive.mount('/content/drive')
```


```python
!pip install seaborn
!pip install statsmodels
!pip install pyprocessmacro
!pip install scikit-learn
!pip install pyreadstat
!sudo apt-get install -y fonts-nanum
!sudo fc-cache -fv
!rm ~/.cache/matplotlib -rf

import warnings
warnings.filterwarnings('ignore')
```


```python
import pandas
import pyreadstat
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
import statsmodels.api as sm
from pyprocessmacro import Process

dtafile = '/content/drive/MyDrive/Analysis_0602/refugee_0426.2023.dta'
df, meta = pyreadstat.read_dta(dtafile)
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 800 entries, 0 to 799
    Columns: 287 entries, no to _Ireligion2_2
    dtypes: float64(279), int64(6), object(2)
    memory usage: 1.8+ MB



```python
# 데이터 값 변경
df.replace({"q92":{1:0, 2:1, 3:2, 4:3, 5:4},
            "q96":{1:0, 2:1, 3:2, 4:3, 5:4},
            "q97":{1:0, 2:1, 3:2, 4:3, 5:4},
            "q175":{2:0}}, inplace=True)
```


```python
# 데이터 컬럼명 변경
df.rename(columns = {'q1r_gender' : 'gender',
                       'q2_1':'age',
                       'q175':'child',
                       '_Iq5_region_1':'region_1',
                       '_Iq5_region_2':'region_2',
                       'q180edu2':'edulevel',
                       'q181':'income',
                       '_Ireligion2_1':'religion_1',
                       '_Ireligion2_2':'religion_2',
                       'q92':'dir_contact',
                       'q96':'massmedia',
                       'q97':'sns',
                     'q99':'dir_cont_qual',
                     'q103':'mass_media_qual',
                     'q104':'sns_qual',
                       'q43':'poli_pers',
                       'q44':'relig_pers',
                       'q45':'econo',
                       'q46':'fake',
                       'q153':'pol_orien'}, inplace = True)
```


```python
# 다양한 데이터셋 구성

# N = 800
columns = ["gender", "age", "child", "region_1", "region_2", "edulevel", "income", "religion_1", "religion_2",
           "T_realf", "T_symbf", "T_safecoh2", "T_health", "dir_contact", "massmedia", "sns",
           "poli_pers", "relig_pers", "econo", "fake",
           "humani", "nat_eth2", "nat_cit2", "SDO_D", "SDO_E", "rightw", "pol_orien", "islamo", "socd_p", "socd_n", "beh_p", "beh_n"]
df_800 = df[columns]

# N = 410 (직접 접촉 경험 o)
df_410 = df[df["dir_cont_qual"] != 6.0]
columns = ["gender", "age", "child", "region_1", "region_2", "edulevel", "income", "religion_1", "religion_2",
           "T_realf", "T_symbf", "T_safecoh2", "T_health", "dir_contact", "massmedia", "sns", 'dir_cont_qual',
           "poli_pers", "relig_pers", "econo", "fake",
           "humani", "nat_eth2", "nat_cit2", "SDO_D", "SDO_E", "rightw", "pol_orien", "islamo", "socd_p", "socd_n", "beh_p", "beh_n"]
df_410 = df_410[columns]

# N = 715 (대중매체 접촉 경험 o)
df_715 = df[df["mass_media_qual"] != 6.0]
columns = ["gender", "age", "child", "region_1", "region_2", "edulevel", "income", "religion_1", "religion_2",
           "T_realf", "T_symbf", "T_safecoh2", "T_health", "dir_contact", "massmedia", "sns", 'mass_media_qual',
           "poli_pers", "relig_pers", "econo", "fake",
           "humani", "nat_eth2", "nat_cit2", "SDO_D", "SDO_E", "rightw", "pol_orien", "islamo", "socd_p", "socd_n", "beh_p", "beh_n"]
df_715 = df_715[columns]

# N = 591 (SNS 접촉 경험 o)
df_591 = df[df["sns_qual"] != 6.0]
columns = ["gender", "age", "child", "region_1", "region_2", "edulevel", "income", "religion_1", "religion_2",
           "T_realf", "T_symbf", "T_safecoh2", "T_health", "dir_contact", "massmedia", "sns", 'sns_qual',
           "poli_pers", "relig_pers", "econo", "fake",
           "humani", "nat_eth2", "nat_cit2", "SDO_D", "SDO_E", "rightw", "pol_orien", "islamo", "socd_p", "socd_n", "beh_p", "beh_n"]
df_591 = df_591[columns]
```


```python
# 다중회귀분석
def regression_model(df, y_data):

    x_data = df.drop(["beh_n", "beh_p"], axis=1)
    x_data_add = sm.add_constant(x_data, has_constant = "add")
    y_data = y_data

    reg_model = sm.OLS(y_data, x_data_add)
    fitted_reg_model = reg_model.fit()
    return fitted_reg_model.summary()


# 조절효과분석
def moderation_model(df, x_var, m_var, y_var):

    control_var = df.drop(["beh_n", "beh_p"], axis=1).columns.tolist()
    control_var.remove(x_var)
    control_var.remove(m_var)

    p = Process(data=df,
            model=1,
            x=x_var,
            m=m_var,
            y=y_var,
            controls=control_var)
    print(f"******************** 종속변수 {y_var}에 대한 {x_var} x {m_var}의 조절효과 분석 결과********************")
    return p.summary()
```

    /usr/local/lib/python3.10/dist-packages/ipykernel/ipkernel.py:283: DeprecationWarning: `should_run_async` will not call `transform_cell` automatically in the future. Please pass the result to `transformed_cell` argument and any exception that happen during thetransform in `preprocessing_exc_tuple` in IPython 7.17 and above.
      and should_run_async(code)



```python
# 회귀분석 예시
y_list = ["beh_p", "beh_n"]
df_list = [df_800, df_410, df_715, df_591]
n_list = [800, 410, 715, 591]
for (df, n) in zip(df_list, n_list):
  for y in y_list:
    print(f"               n = {n}일때, 종속변수 {y}에 대한 회귀분석 결과")
    print(regression_model(df, df[y]))
    print()
    print()
```

                   n = 800일때, 종속변수 beh_p에 대한 회귀분석 결과
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                  beh_p   R-squared:                       0.628
    Model:                            OLS   Adj. R-squared:                  0.613
    Method:                 Least Squares   F-statistic:                     43.24
    Date:                Sun, 04 Jun 2023   Prob (F-statistic):          3.11e-143
    Time:                        07:33:41   Log-Likelihood:                -615.98
    No. Observations:                 800   AIC:                             1294.
    Df Residuals:                     769   BIC:                             1439.
    Df Model:                          30                                         
    Covariance Type:            nonrobust                                         
    ===============================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------
    const           2.2106      0.343      6.450      0.000       1.538       2.883
    gender         -0.0426      0.041     -1.033      0.302      -0.123       0.038
    age             0.0018      0.002      0.849      0.396      -0.002       0.006
    child           0.1341      0.056      2.409      0.016       0.025       0.243
    region_1        0.0792      0.040      1.977      0.048       0.001       0.158
    region_2       -0.1348      0.174     -0.775      0.439      -0.476       0.207
    edulevel        0.0206      0.013      1.569      0.117      -0.005       0.046
    income         -0.0086      0.010     -0.835      0.404      -0.029       0.012
    religion_1     -0.0466      0.058     -0.808      0.419      -0.160       0.067
    religion_2     -0.0715      0.053     -1.358      0.175      -0.175       0.032
    T_realf        -0.2353      0.033     -7.047      0.000      -0.301      -0.170
    T_symbf        -0.1059      0.028     -3.818      0.000      -0.160      -0.051
    T_safecoh2     -0.0992      0.037     -2.689      0.007      -0.172      -0.027
    T_health        0.0167      0.025      0.668      0.505      -0.033       0.066
    dir_contact     0.1036      0.029      3.587      0.000       0.047       0.160
    massmedia      -0.0312      0.022     -1.419      0.156      -0.074       0.012
    sns             0.0597      0.020      3.009      0.003       0.021       0.099
    poli_pers      -0.0041      0.027     -0.150      0.881      -0.057       0.049
    relig_pers      0.1573      0.025      6.187      0.000       0.107       0.207
    econo           0.1363      0.023      5.879      0.000       0.091       0.182
    fake            0.0321      0.021      1.559      0.119      -0.008       0.073
    humani          0.1994      0.030      6.645      0.000       0.140       0.258
    nat_eth2        0.0257      0.029      0.872      0.383      -0.032       0.084
    nat_cit2       -0.0521      0.035     -1.475      0.141      -0.121       0.017
    SDO_D           0.0348      0.033      1.069      0.285      -0.029       0.099
    SDO_E          -0.0638      0.027     -2.387      0.017      -0.116      -0.011
    rightw         -0.0372      0.062     -0.596      0.551      -0.160       0.085
    pol_orien      -0.0546      0.025     -2.220      0.027      -0.103      -0.006
    islamo         -0.0045      0.022     -0.205      0.837      -0.047       0.038
    socd_p          0.0291      0.036      0.801      0.423      -0.042       0.100
    socd_n          0.0593      0.030      1.975      0.049       0.000       0.118
    ==============================================================================
    Omnibus:                       10.391   Durbin-Watson:                   1.989
    Prob(Omnibus):                  0.006   Jarque-Bera (JB):               13.156
    Skew:                           0.156   Prob(JB):                      0.00139
    Kurtosis:                       3.545   Cond. No.                         897.
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    
    
                   n = 800일때, 종속변수 beh_n에 대한 회귀분석 결과
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                  beh_n   R-squared:                       0.287
    Model:                            OLS   Adj. R-squared:                  0.260
    Method:                 Least Squares   F-statistic:                     10.34
    Date:                Sun, 04 Jun 2023   Prob (F-statistic):           1.72e-39
    Time:                        07:33:41   Log-Likelihood:                -989.51
    No. Observations:                 800   AIC:                             2041.
    Df Residuals:                     769   BIC:                             2186.
    Df Model:                          30                                         
    Covariance Type:            nonrobust                                         
    ===============================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------
    const           0.6277      0.547      1.148      0.251      -0.445       1.701
    gender         -0.0556      0.066     -0.846      0.398      -0.185       0.073
    age            -0.0045      0.003     -1.307      0.192      -0.011       0.002
    child           0.1500      0.089      1.690      0.091      -0.024       0.324
    region_1        0.0377      0.064      0.590      0.555      -0.088       0.163
    region_2        0.2390      0.277      0.861      0.389      -0.306       0.784
    edulevel        0.0350      0.021      1.674      0.094      -0.006       0.076
    income         -0.0142      0.016     -0.866      0.386      -0.046       0.018
    religion_1      0.1169      0.092      1.270      0.205      -0.064       0.298
    religion_2     -0.1270      0.084     -1.513      0.131      -0.292       0.038
    T_realf        -0.0269      0.053     -0.504      0.614      -0.131       0.078
    T_symbf        -0.0058      0.044     -0.131      0.896      -0.093       0.081
    T_safecoh2      0.2529      0.059      4.299      0.000       0.137       0.368
    T_health        0.0254      0.040      0.635      0.525      -0.053       0.104
    dir_contact     0.1578      0.046      3.426      0.001       0.067       0.248
    massmedia      -0.0008      0.035     -0.022      0.983      -0.070       0.068
    sns             0.0122      0.032      0.385      0.700      -0.050       0.074
    poli_pers      -0.0821      0.043     -1.894      0.059      -0.167       0.003
    relig_pers      0.0511      0.041      1.261      0.208      -0.028       0.131
    econo           0.0537      0.037      1.451      0.147      -0.019       0.126
    fake            0.1633      0.033      4.969      0.000       0.099       0.228
    humani         -0.1283      0.048     -2.681      0.008      -0.222      -0.034
    nat_eth2        0.0731      0.047      1.553      0.121      -0.019       0.165
    nat_cit2       -0.0822      0.056     -1.458      0.145      -0.193       0.028
    SDO_D           0.1119      0.052      2.158      0.031       0.010       0.214
    SDO_E          -0.0323      0.043     -0.758      0.449      -0.116       0.051
    rightw          0.0907      0.100      0.911      0.363      -0.105       0.286
    pol_orien      -0.0295      0.039     -0.752      0.452      -0.107       0.048
    islamo          0.0544      0.035      1.565      0.118      -0.014       0.123
    socd_p          0.0612      0.058      1.057      0.291      -0.052       0.175
    socd_n          0.0036      0.048      0.075      0.940      -0.090       0.098
    ==============================================================================
    Omnibus:                        2.935   Durbin-Watson:                   2.199
    Prob(Omnibus):                  0.230   Jarque-Bera (JB):                2.986
    Skew:                           0.128   Prob(JB):                        0.225
    Kurtosis:                       2.847   Cond. No.                         897.
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    
    
                   n = 410일때, 종속변수 beh_p에 대한 회귀분석 결과
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                  beh_p   R-squared:                       0.700
    Model:                            OLS   Adj. R-squared:                  0.675
    Method:                 Least Squares   F-statistic:                     28.44
    Date:                Sun, 04 Jun 2023   Prob (F-statistic):           4.98e-80
    Time:                        07:33:41   Log-Likelihood:                -284.06
    No. Observations:                 410   AIC:                             632.1
    Df Residuals:                     378   BIC:                             760.6
    Df Model:                          31                                         
    Covariance Type:            nonrobust                                         
    =================================================================================
                        coef    std err          t      P>|t|      [0.025      0.975]
    ---------------------------------------------------------------------------------
    const             1.8867      0.472      3.993      0.000       0.958       2.816
    gender           -0.0094      0.056     -0.170      0.865      -0.119       0.100
    age               0.0062      0.003      2.029      0.043       0.000       0.012
    child             0.0332      0.078      0.426      0.670      -0.120       0.187
    region_1          0.0912      0.055      1.663      0.097      -0.017       0.199
    region_2         -0.1818      0.263     -0.691      0.490      -0.699       0.336
    edulevel          0.0014      0.018      0.078      0.938      -0.034       0.037
    income           -0.0034      0.014     -0.243      0.808      -0.030       0.024
    religion_1        0.0050      0.076      0.066      0.948      -0.145       0.155
    religion_2       -0.0063      0.071     -0.090      0.929      -0.145       0.132
    T_realf          -0.3206      0.045     -7.057      0.000      -0.410      -0.231
    T_symbf          -0.1099      0.038     -2.872      0.004      -0.185      -0.035
    T_safecoh2        0.0037      0.050      0.076      0.940      -0.094       0.101
    T_health         -0.0199      0.035     -0.574      0.566      -0.088       0.048
    dir_contact       0.0277      0.032      0.855      0.393      -0.036       0.091
    massmedia        -0.0950      0.029     -3.263      0.001      -0.152      -0.038
    sns               0.0927      0.028      3.299      0.001       0.037       0.148
    dir_cont_qual     0.1016      0.031      3.277      0.001       0.041       0.163
    poli_pers         0.0112      0.038      0.292      0.771      -0.064       0.086
    relig_pers        0.2035      0.036      5.710      0.000       0.133       0.274
    econo             0.1195      0.033      3.567      0.000       0.054       0.185
    fake              0.0301      0.027      1.120      0.264      -0.023       0.083
    humani            0.1254      0.042      3.020      0.003       0.044       0.207
    nat_eth2         -0.0446      0.043     -1.029      0.304      -0.130       0.041
    nat_cit2         -0.0148      0.047     -0.312      0.755      -0.108       0.078
    SDO_D             0.0697      0.045      1.545      0.123      -0.019       0.158
    SDO_E            -0.0664      0.033     -2.040      0.042      -0.130      -0.002
    rightw            0.0286      0.096      0.300      0.765      -0.159       0.216
    pol_orien        -0.0191      0.033     -0.585      0.559      -0.083       0.045
    islamo            0.0104      0.029      0.356      0.722      -0.047       0.068
    socd_p           -0.0018      0.051     -0.036      0.971      -0.101       0.098
    socd_n            0.0791      0.041      1.928      0.055      -0.002       0.160
    ==============================================================================
    Omnibus:                        2.074   Durbin-Watson:                   1.936
    Prob(Omnibus):                  0.355   Jarque-Bera (JB):                2.026
    Skew:                           0.037   Prob(JB):                        0.363
    Kurtosis:                       3.336   Cond. No.                         941.
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    
    
                   n = 410일때, 종속변수 beh_n에 대한 회귀분석 결과
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                  beh_n   R-squared:                       0.300
    Model:                            OLS   Adj. R-squared:                  0.242
    Method:                 Least Squares   F-statistic:                     5.216
    Date:                Sun, 04 Jun 2023   Prob (F-statistic):           9.96e-16
    Time:                        07:33:41   Log-Likelihood:                -504.78
    No. Observations:                 410   AIC:                             1074.
    Df Residuals:                     378   BIC:                             1202.
    Df Model:                          31                                         
    Covariance Type:            nonrobust                                         
    =================================================================================
                        coef    std err          t      P>|t|      [0.025      0.975]
    ---------------------------------------------------------------------------------
    const             1.3572      0.809      1.677      0.094      -0.234       2.949
    gender           -0.1189      0.095     -1.249      0.212      -0.306       0.068
    age              -0.0055      0.005     -1.046      0.296      -0.016       0.005
    child             0.1937      0.134      1.448      0.148      -0.069       0.457
    region_1          0.1630      0.094      1.735      0.084      -0.022       0.348
    region_2          0.1942      0.451      0.431      0.667      -0.692       1.081
    edulevel          0.0370      0.031      1.194      0.233      -0.024       0.098
    income           -0.0055      0.024     -0.231      0.817      -0.052       0.041
    religion_1       -0.0077      0.130     -0.059      0.953      -0.264       0.249
    religion_2       -0.2058      0.121     -1.702      0.090      -0.444       0.032
    T_realf          -0.1317      0.078     -1.692      0.091      -0.285       0.021
    T_symbf           0.0167      0.066      0.254      0.799      -0.112       0.146
    T_safecoh2        0.2559      0.085      3.013      0.003       0.089       0.423
    T_health          0.0797      0.059      1.345      0.179      -0.037       0.196
    dir_contact       0.1305      0.056      2.350      0.019       0.021       0.240
    massmedia         0.0051      0.050      0.102      0.919      -0.093       0.103
    sns               0.0012      0.048      0.024      0.981      -0.094       0.096
    dir_cont_qual    -0.0210      0.053     -0.395      0.693      -0.125       0.083
    poli_pers        -0.0660      0.066     -1.007      0.315      -0.195       0.063
    relig_pers        0.0966      0.061      1.583      0.114      -0.023       0.217
    econo             0.0301      0.057      0.525      0.600      -0.083       0.143
    fake              0.1779      0.046      3.865      0.000       0.087       0.268
    humani           -0.1210      0.071     -1.700      0.090      -0.261       0.019
    nat_eth2          0.0055      0.074      0.074      0.941      -0.141       0.152
    nat_cit2         -0.0291      0.081     -0.360      0.719      -0.188       0.130
    SDO_D             0.1657      0.077      2.143      0.033       0.014       0.318
    SDO_E            -0.0748      0.056     -1.343      0.180      -0.184       0.035
    rightw           -0.1557      0.164     -0.952      0.342      -0.477       0.166
    pol_orien        -0.0305      0.056     -0.546      0.586      -0.140       0.079
    islamo            0.0678      0.050      1.358      0.175      -0.030       0.166
    socd_p            0.0840      0.087      0.967      0.334      -0.087       0.255
    socd_n           -0.0026      0.070     -0.037      0.970      -0.141       0.136
    ==============================================================================
    Omnibus:                        0.660   Durbin-Watson:                   1.989
    Prob(Omnibus):                  0.719   Jarque-Bera (JB):                0.767
    Skew:                           0.048   Prob(JB):                        0.681
    Kurtosis:                       2.811   Cond. No.                         941.
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    
    
                   n = 715일때, 종속변수 beh_p에 대한 회귀분석 결과
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                  beh_p   R-squared:                       0.640
    Model:                            OLS   Adj. R-squared:                  0.624
    Method:                 Least Squares   F-statistic:                     39.15
    Date:                Sun, 04 Jun 2023   Prob (F-statistic):          1.20e-129
    Time:                        07:33:41   Log-Likelihood:                -546.93
    No. Observations:                 715   AIC:                             1158.
    Df Residuals:                     683   BIC:                             1304.
    Df Model:                          31                                         
    Covariance Type:            nonrobust                                         
    ===================================================================================
                          coef    std err          t      P>|t|      [0.025      0.975]
    -----------------------------------------------------------------------------------
    const               1.9990      0.364      5.495      0.000       1.285       2.713
    gender             -0.0214      0.043     -0.493      0.622      -0.106       0.064
    age                 0.0026      0.002      1.102      0.271      -0.002       0.007
    child               0.0912      0.060      1.530      0.126      -0.026       0.208
    region_1            0.0917      0.042      2.169      0.030       0.009       0.175
    region_2           -0.1127      0.184     -0.614      0.540      -0.474       0.248
    edulevel            0.0204      0.014      1.472      0.141      -0.007       0.048
    income             -0.0075      0.011     -0.681      0.496      -0.029       0.014
    religion_1         -0.0668      0.061     -1.093      0.275      -0.187       0.053
    religion_2         -0.0726      0.055     -1.309      0.191      -0.182       0.036
    T_realf            -0.2497      0.036     -6.968      0.000      -0.320      -0.179
    T_symbf            -0.0991      0.029     -3.388      0.001      -0.157      -0.042
    T_safecoh2         -0.0582      0.040     -1.452      0.147      -0.137       0.020
    T_health            0.0108      0.026      0.408      0.683      -0.041       0.063
    dir_contact         0.0958      0.030      3.203      0.001       0.037       0.154
    massmedia          -0.0349      0.025     -1.411      0.159      -0.084       0.014
    sns                 0.0568      0.020      2.774      0.006       0.017       0.097
    mass_media_qual     0.0718      0.025      2.898      0.004       0.023       0.120
    poli_pers          -0.0137      0.029     -0.481      0.631      -0.070       0.042
    relig_pers          0.1643      0.027      6.096      0.000       0.111       0.217
    econo               0.1428      0.024      5.879      0.000       0.095       0.191
    fake                0.0462      0.022      2.109      0.035       0.003       0.089
    humani              0.1845      0.033      5.579      0.000       0.120       0.249
    nat_eth2            0.0220      0.032      0.696      0.487      -0.040       0.084
    nat_cit2           -0.0643      0.037     -1.722      0.086      -0.138       0.009
    SDO_D               0.0198      0.035      0.571      0.568      -0.048       0.088
    SDO_E              -0.0622      0.028     -2.193      0.029      -0.118      -0.007
    rightw             -0.0374      0.066     -0.570      0.569      -0.166       0.091
    pol_orien          -0.0373      0.026     -1.409      0.159      -0.089       0.015
    islamo             -0.0091      0.023     -0.393      0.695      -0.055       0.036
    socd_p              0.0243      0.038      0.632      0.528      -0.051       0.100
    socd_n              0.0523      0.032      1.647      0.100      -0.010       0.115
    ==============================================================================
    Omnibus:                       12.012   Durbin-Watson:                   2.059
    Prob(Omnibus):                  0.002   Jarque-Bera (JB):               15.077
    Skew:                           0.200   Prob(JB):                     0.000532
    Kurtosis:                       3.588   Cond. No.                         904.
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    
    
                   n = 715일때, 종속변수 beh_n에 대한 회귀분석 결과
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                  beh_n   R-squared:                       0.307
    Model:                            OLS   Adj. R-squared:                  0.275
    Method:                 Least Squares   F-statistic:                     9.751
    Date:                Sun, 04 Jun 2023   Prob (F-statistic):           4.06e-37
    Time:                        07:33:41   Log-Likelihood:                -876.59
    No. Observations:                 715   AIC:                             1817.
    Df Residuals:                     683   BIC:                             1963.
    Df Model:                          31                                         
    Covariance Type:            nonrobust                                         
    ===================================================================================
                          coef    std err          t      P>|t|      [0.025      0.975]
    -----------------------------------------------------------------------------------
    const               0.6718      0.577      1.165      0.245      -0.461       1.805
    gender             -0.0553      0.069     -0.805      0.421      -0.190       0.080
    age                -0.0026      0.004     -0.715      0.475      -0.010       0.005
    child               0.1659      0.094      1.755      0.080      -0.020       0.351
    region_1            0.0467      0.067      0.696      0.487      -0.085       0.178
    region_2           -0.0140      0.291     -0.048      0.962      -0.586       0.558
    edulevel            0.0428      0.022      1.953      0.051      -0.000       0.086
    income             -0.0118      0.017     -0.679      0.497      -0.046       0.022
    religion_1          0.0729      0.097      0.752      0.453      -0.118       0.263
    religion_2         -0.1377      0.088     -1.565      0.118      -0.311       0.035
    T_realf            -0.0595      0.057     -1.047      0.296      -0.171       0.052
    T_symbf            -0.0112      0.046     -0.241      0.810      -0.102       0.080
    T_safecoh2          0.2592      0.063      4.083      0.000       0.135       0.384
    T_health            0.0146      0.042      0.347      0.729      -0.068       0.097
    dir_contact         0.1683      0.047      3.550      0.000       0.075       0.261
    massmedia           0.0101      0.039      0.257      0.798      -0.067       0.087
    sns                 0.0184      0.032      0.567      0.571      -0.045       0.082
    mass_media_qual    -0.0913      0.039     -2.325      0.020      -0.168      -0.014
    poli_pers          -0.0990      0.045     -2.191      0.029      -0.188      -0.010
    relig_pers          0.0674      0.043      1.577      0.115      -0.017       0.151
    econo               0.0599      0.039      1.554      0.121      -0.016       0.136
    fake                0.1512      0.035      4.349      0.000       0.083       0.220
    humani             -0.1642      0.052     -3.130      0.002      -0.267      -0.061
    nat_eth2            0.0509      0.050      1.016      0.310      -0.047       0.149
    nat_cit2           -0.0540      0.059     -0.912      0.362      -0.170       0.062
    SDO_D               0.1324      0.055      2.413      0.016       0.025       0.240
    SDO_E              -0.0211      0.045     -0.470      0.639      -0.110       0.067
    rightw              0.1263      0.104      1.216      0.225      -0.078       0.330
    pol_orien          -0.0292      0.042     -0.696      0.486      -0.112       0.053
    islamo              0.0326      0.037      0.887      0.375      -0.040       0.105
    socd_p              0.0917      0.061      1.504      0.133      -0.028       0.211
    socd_n              0.0244      0.050      0.485      0.628      -0.074       0.123
    ==============================================================================
    Omnibus:                        3.601   Durbin-Watson:                   2.164
    Prob(Omnibus):                  0.165   Jarque-Bera (JB):                3.536
    Skew:                           0.136   Prob(JB):                        0.171
    Kurtosis:                       2.789   Cond. No.                         904.
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    
    
                   n = 591일때, 종속변수 beh_p에 대한 회귀분석 결과
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                  beh_p   R-squared:                       0.659
    Model:                            OLS   Adj. R-squared:                  0.640
    Method:                 Least Squares   F-statistic:                     34.88
    Date:                Sun, 04 Jun 2023   Prob (F-statistic):          6.75e-110
    Time:                        07:33:41   Log-Likelihood:                -444.90
    No. Observations:                 591   AIC:                             953.8
    Df Residuals:                     559   BIC:                             1094.
    Df Model:                          31                                         
    Covariance Type:            nonrobust                                         
    ===============================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------
    const           1.9320      0.397      4.866      0.000       1.152       2.712
    gender         -0.0004      0.048     -0.009      0.993      -0.094       0.093
    age             0.0029      0.003      1.128      0.260      -0.002       0.008
    child           0.1257      0.067      1.880      0.061      -0.006       0.257
    region_1        0.0854      0.047      1.833      0.067      -0.006       0.177
    region_2       -0.1450      0.206     -0.703      0.482      -0.550       0.260
    edulevel        0.0227      0.015      1.502      0.134      -0.007       0.052
    income         -0.0068      0.012     -0.566      0.571      -0.030       0.017
    religion_1     -0.0187      0.067     -0.277      0.782      -0.151       0.114
    religion_2     -0.0289      0.060     -0.478      0.633      -0.148       0.090
    T_realf        -0.2671      0.039     -6.839      0.000      -0.344      -0.190
    T_symbf        -0.0930      0.033     -2.853      0.004      -0.157      -0.029
    T_safecoh2     -0.0352      0.044     -0.805      0.421      -0.121       0.051
    T_health       -0.0116      0.030     -0.391      0.696      -0.070       0.047
    dir_contact     0.0883      0.030      2.898      0.004       0.028       0.148
    massmedia      -0.0471      0.028     -1.682      0.093      -0.102       0.008
    sns             0.0514      0.025      2.063      0.040       0.002       0.100
    sns_qual        0.0595      0.027      2.204      0.028       0.006       0.112
    poli_pers      -0.0208      0.031     -0.667      0.505      -0.082       0.040
    relig_pers      0.1711      0.029      5.827      0.000       0.113       0.229
    econo           0.1532      0.027      5.627      0.000       0.100       0.207
    fake            0.0421      0.024      1.749      0.081      -0.005       0.089
    humani          0.1834      0.036      5.143      0.000       0.113       0.253
    nat_eth2        0.0085      0.036      0.237      0.813      -0.062       0.079
    nat_cit2       -0.0591      0.041     -1.433      0.152      -0.140       0.022
    SDO_D           0.0522      0.039      1.351      0.177      -0.024       0.128
    SDO_E          -0.0506      0.030     -1.663      0.097      -0.110       0.009
    rightw         -0.0355      0.074     -0.479      0.632      -0.181       0.110
    pol_orien      -0.0402      0.029     -1.374      0.170      -0.098       0.017
    islamo         -0.0076      0.025     -0.309      0.757      -0.056       0.041
    socd_p          0.0197      0.042      0.463      0.643      -0.064       0.103
    socd_n          0.0487      0.035      1.379      0.168      -0.021       0.118
    ==============================================================================
    Omnibus:                       10.618   Durbin-Watson:                   2.026
    Prob(Omnibus):                  0.005   Jarque-Bera (JB):               13.991
    Skew:                           0.183   Prob(JB):                     0.000916
    Kurtosis:                       3.658   Cond. No.                         896.
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    
    
                   n = 591일때, 종속변수 beh_n에 대한 회귀분석 결과
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                  beh_n   R-squared:                       0.315
    Model:                            OLS   Adj. R-squared:                  0.277
    Method:                 Least Squares   F-statistic:                     8.309
    Date:                Sun, 04 Jun 2023   Prob (F-statistic):           7.97e-30
    Time:                        07:33:41   Log-Likelihood:                -728.22
    No. Observations:                 591   AIC:                             1520.
    Df Residuals:                     559   BIC:                             1661.
    Df Model:                          31                                         
    Covariance Type:            nonrobust                                         
    ===============================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------
    const           0.4566      0.641      0.712      0.477      -0.803       1.716
    gender         -0.0784      0.077     -1.021      0.308      -0.229       0.072
    age            -0.0065      0.004     -1.538      0.125      -0.015       0.002
    child           0.2586      0.108      2.394      0.017       0.046       0.471
    region_1        0.1337      0.075      1.776      0.076      -0.014       0.282
    region_2        0.3559      0.333      1.069      0.286      -0.298       1.010
    edulevel        0.0326      0.024      1.334      0.183      -0.015       0.080
    income          0.0040      0.019      0.210      0.834      -0.034       0.042
    religion_1      0.0429      0.109      0.395      0.693      -0.171       0.256
    religion_2     -0.1176      0.098     -1.204      0.229      -0.309       0.074
    T_realf        -0.0777      0.063     -1.231      0.219      -0.202       0.046
    T_symbf        -0.0262      0.053     -0.498      0.619      -0.130       0.077
    T_safecoh2      0.2912      0.071      4.121      0.000       0.152       0.430
    T_health        0.0528      0.048      1.099      0.272      -0.042       0.147
    dir_contact     0.1698      0.049      3.451      0.001       0.073       0.266
    massmedia       0.0187      0.045      0.412      0.680      -0.070       0.108
    sns            -0.0059      0.040     -0.146      0.884      -0.085       0.073
    sns_qual       -0.0443      0.044     -1.015      0.310      -0.130       0.041
    poli_pers      -0.0928      0.050     -1.844      0.066      -0.192       0.006
    relig_pers      0.0746      0.047      1.573      0.116      -0.019       0.168
    econo           0.0631      0.044      1.435      0.152      -0.023       0.149
    fake            0.1592      0.039      4.092      0.000       0.083       0.236
    humani         -0.1364      0.058     -2.368      0.018      -0.250      -0.023
    nat_eth2        0.1163      0.058      2.006      0.045       0.002       0.230
    nat_cit2       -0.1263      0.067     -1.896      0.059      -0.257       0.005
    SDO_D           0.1272      0.062      2.037      0.042       0.005       0.250
    SDO_E          -0.0257      0.049     -0.523      0.601      -0.122       0.071
    rightw          0.1307      0.120      1.091      0.276      -0.105       0.366
    pol_orien      -0.0356      0.047     -0.754      0.451      -0.128       0.057
    islamo          0.0415      0.040      1.040      0.299      -0.037       0.120
    socd_p          0.0848      0.069      1.236      0.217      -0.050       0.219
    socd_n          0.0074      0.057      0.129      0.897      -0.105       0.119
    ==============================================================================
    Omnibus:                        1.088   Durbin-Watson:                   2.114
    Prob(Omnibus):                  0.580   Jarque-Bera (JB):                1.143
    Skew:                           0.050   Prob(JB):                        0.565
    Kurtosis:                       2.810   Cond. No.                         896.
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    
    



```python
# 조절효과

y_list = ["beh_p", "beh_n"]

for y in y_list:
  moderation_model(df = df_410, x_var="dir_contact", m_var="dir_cont_qual", y_var=y)

for y in y_list:
  moderation_model(df = df_715, x_var="massmedia", m_var="mass_media_qual", y_var=y)

for y in y_list:
  moderation_model(df = df_591, x_var="sns", m_var="sns_qual", y_var=y)
```

    Process successfully initialized.
    Based on the Process Macro by Andrew F. Hayes, Ph.D. (www.afhayes.com)
    
    
    ****************************** SPECIFICATION ****************************
    
    Model = 1
    
    Variables:
        Cons = Cons
        x = dir_contact
        m = dir_cont_qual
        y = beh_p
    Statistical Controls:
     gender, age, child, region_1, region_2, edulevel, income, religion_1, religion_2, T_realf, T_symbf, T_safecoh2, T_health, massmedia, sns, poli_pers, relig_pers, econo, fake, humani, nat_eth2, nat_cit2, SDO_D, SDO_E, rightw, pol_orien, islamo, socd_p, socd_n
    
    
    
    Sample size:
    410
    ******************** 종속변수 beh_p에 대한 dir_contact x dir_cont_qual의 조절효과 분석 결과********************
    
    ***************************** OUTCOME MODELS ****************************
    
    Outcome = beh_p 
    OLS Regression Summary
    
         R²  Adj. R²    MSE       F  df1  df2  p-value
     0.7011   0.6748 0.2536 27.6276   32  377   0.0000
    
    Coefficients
    
                                coeff     se       t      p    LLCI    ULCI
    Cons                       1.9986 0.4815  4.1505 0.0000  1.0548  2.9424
    dir_contact               -0.0596 0.0804 -0.7415 0.4588 -0.2172  0.0980
    dir_cont_qual              0.0826 0.0349  2.3671 0.0184  0.0142  0.1509
    dir_contact*dir_cont_qual  0.0277 0.0233  1.1872 0.2359 -0.0180  0.0734
    gender                    -0.0080 0.0556 -0.1436 0.8859 -0.1169  0.1009
    age                        0.0063 0.0031  2.0659 0.0395  0.0003  0.0123
    child                      0.0259 0.0783  0.3314 0.7405 -0.1274  0.1793
    region_1                   0.0941 0.0549  1.7162 0.0870 -0.0134  0.2017
    region_2                  -0.1890 0.2631 -0.7183 0.4730 -0.7047  0.3267
    edulevel                  -0.0020 0.0183 -0.1108 0.9118 -0.0379  0.0338
    income                    -0.0014 0.0139 -0.0980 0.9220 -0.0286  0.0259
    religion_1                 0.0056 0.0761  0.0736 0.9413 -0.1435  0.1547
    religion_2                -0.0124 0.0707 -0.1754 0.8609 -0.1510  0.1262
    T_realf                   -0.3205 0.0454 -7.0593 0.0000 -0.4095 -0.2315
    T_symbf                   -0.1133 0.0384 -2.9546 0.0033 -0.1885 -0.0381
    T_safecoh2                 0.0057 0.0496  0.1159 0.9078 -0.0914  0.1029
    T_health                  -0.0233 0.0347 -0.6708 0.5028 -0.0913  0.0447
    massmedia                 -0.0938 0.0291 -3.2199 0.0014 -0.1509 -0.0367
    sns                        0.0933 0.0281  3.3225 0.0010  0.0383  0.1484
    poli_pers                  0.0108 0.0382  0.2829 0.7774 -0.0641  0.0858
    relig_pers                 0.2044 0.0356  5.7371 0.0000  0.1346  0.2742
    econo                      0.1161 0.0336  3.4568 0.0006  0.0503  0.1820
    fake                       0.0330 0.0270  1.2230 0.2221 -0.0199  0.0858
    humani                     0.1264 0.0415  3.0447 0.0025  0.0450  0.2078
    nat_eth2                  -0.0463 0.0434 -1.0668 0.2867 -0.1313  0.0388
    nat_cit2                  -0.0105 0.0474 -0.2206 0.8255 -0.1034  0.0825
    SDO_D                      0.0665 0.0452  1.4716 0.1420 -0.0221  0.1550
    SDO_E                     -0.0682 0.0325 -2.0949 0.0368 -0.1320 -0.0044
    rightw                     0.0294 0.0955  0.3083 0.7580 -0.1577  0.2166
    pol_orien                 -0.0173 0.0326 -0.5315 0.5954 -0.0813  0.0466
    islamo                     0.0077 0.0292  0.2642 0.7918 -0.0496  0.0650
    socd_p                    -0.0099 0.0511 -0.1930 0.8470 -0.1100  0.0903
    socd_n                     0.0735 0.0413  1.7813 0.0757 -0.0074  0.1544
    
    -------------------------------------------------------------------------
    
    
    ********************** CONDITIONAL EFFECTS **********************
    
    Conditional effect(s) of dir_contact on beh_p at values of the moderator(s):
    
      dir_cont_qual  Effect     SE       t      p    LLCI   ULCI
             1.0000 -0.0319 0.0598 -0.5340 0.5936 -0.1492 0.0853
             2.0000 -0.0043 0.0421 -0.1010 0.9196 -0.0869 0.0783
             3.0000  0.0234 0.0326  0.7184 0.4730 -0.0405 0.0873
             4.0000  0.0511 0.0379  1.3478 0.1785 -0.0232 0.1254
             5.0000  0.0788 0.0539  1.4631 0.1443 -0.0268 0.1844
    
    Process successfully initialized.
    Based on the Process Macro by Andrew F. Hayes, Ph.D. (www.afhayes.com)
    
    
    ****************************** SPECIFICATION ****************************
    
    Model = 1
    
    Variables:
        Cons = Cons
        x = dir_contact
        m = dir_cont_qual
        y = beh_n
    Statistical Controls:
     gender, age, child, region_1, region_2, edulevel, income, religion_1, religion_2, T_realf, T_symbf, T_safecoh2, T_health, massmedia, sns, poli_pers, relig_pers, econo, fake, humani, nat_eth2, nat_cit2, SDO_D, SDO_E, rightw, pol_orien, islamo, socd_p, socd_n
    
    
    
    Sample size:
    410
    ******************** 종속변수 beh_n에 대한 dir_contact x dir_cont_qual의 조절효과 분석 결과********************
    
    ***************************** OUTCOME MODELS ****************************
    
    Outcome = beh_n 
    OLS Regression Summary
    
         R²  Adj. R²    MSE      F  df1  df2  p-value
     0.3020   0.2407 0.7445 5.0970   32  377   0.0000
    
    Coefficients
    
                                coeff     se       t      p    LLCI   ULCI
    Cons                       1.5400 0.8251  1.8665 0.0628 -0.0771 3.1571
    dir_contact               -0.0121 0.1378 -0.0880 0.9299 -0.2821 0.2579
    dir_cont_qual             -0.0520 0.0598 -0.8707 0.3845 -0.1692 0.0651
    dir_contact*dir_cont_qual  0.0452 0.0400  1.1314 0.2586 -0.0331 0.1235
    gender                    -0.1165 0.0952 -1.2244 0.2216 -0.3031 0.0700
    age                       -0.0053 0.0052 -1.0106 0.3129 -0.0155 0.0050
    child                      0.1818 0.1341  1.3555 0.1761 -0.0810 0.4446
    region_1                   0.1678 0.0940  1.7850 0.0751 -0.0164 0.3520
    region_2                   0.1825 0.4508  0.4047 0.6859 -0.7012 1.0661
    edulevel                   0.0314 0.0314  1.0008 0.3176 -0.0301 0.0928
    income                    -0.0022 0.0238 -0.0929 0.9260 -0.0488 0.0444
    religion_1                -0.0067 0.1303 -0.0513 0.9591 -0.2621 0.2488
    religion_2                -0.2157 0.1212 -1.7799 0.0759 -0.4533 0.0218
    T_realf                   -0.1316 0.0778 -1.6918 0.0915 -0.2841 0.0209
    T_symbf                    0.0111 0.0657  0.1692 0.8658 -0.1177 0.1399
    T_safecoh2                 0.2591 0.0850  3.0504 0.0024  0.0926 0.4256
    T_health                   0.0741 0.0595  1.2470 0.2132 -0.0424 0.1907
    massmedia                  0.0071 0.0499  0.1422 0.8870 -0.0907 0.1049
    sns                        0.0022 0.0481  0.0457 0.9636 -0.0922 0.0966
    poli_pers                 -0.0665 0.0655 -1.0152 0.3107 -0.1950 0.0619
    relig_pers                 0.0981 0.0610  1.6073 0.1088 -0.0215 0.2178
    econo                      0.0246 0.0576  0.4282 0.6688 -0.0882 0.1375
    fake                       0.1826 0.0462  3.9530 0.0001  0.0921 0.2731
    humani                    -0.1194 0.0711 -1.6781 0.0942 -0.2588 0.0201
    nat_eth2                   0.0028 0.0743  0.0380 0.9697 -0.1429 0.1485
    nat_cit2                  -0.0221 0.0812 -0.2719 0.7858 -0.1813 0.1371
    SDO_D                      0.1604 0.0774  2.0723 0.0389  0.0087 0.3121
    SDO_E                     -0.0778 0.0558 -1.3953 0.1637 -0.1871 0.0315
    rightw                    -0.1544 0.1636 -0.9440 0.3458 -0.4750 0.1662
    pol_orien                 -0.0276 0.0559 -0.4942 0.6215 -0.1372 0.0819
    islamo                     0.0635 0.0501  1.2684 0.2054 -0.0346 0.1617
    socd_p                     0.0709 0.0875  0.8096 0.4187 -0.1007 0.2424
    socd_n                    -0.0117 0.0707 -0.1655 0.8686 -0.1503 0.1269
    
    -------------------------------------------------------------------------
    
    
    ********************** CONDITIONAL EFFECTS **********************
    
    Conditional effect(s) of dir_contact on beh_n at values of the moderator(s):
    
      dir_cont_qual  Effect     SE      t      p    LLCI   ULCI
             1.0000  0.0331 0.1025 0.3228 0.7470 -0.1678 0.2339
             2.0000  0.0783 0.0722 1.0841 0.2790 -0.0633 0.2198
             3.0000  0.1235 0.0559 2.2101 0.0277  0.0140 0.2330
             4.0000  0.1687 0.0650 2.5962 0.0098  0.0413 0.2961
             5.0000  0.2139 0.0923 2.3181 0.0210  0.0330 0.3948
    
    Process successfully initialized.
    Based on the Process Macro by Andrew F. Hayes, Ph.D. (www.afhayes.com)
    
    
    ****************************** SPECIFICATION ****************************
    
    Model = 1
    
    Variables:
        Cons = Cons
        x = massmedia
        m = mass_media_qual
        y = beh_p
    Statistical Controls:
     gender, age, child, region_1, region_2, edulevel, income, religion_1, religion_2, T_realf, T_symbf, T_safecoh2, T_health, dir_contact, sns, poli_pers, relig_pers, econo, fake, humani, nat_eth2, nat_cit2, SDO_D, SDO_E, rightw, pol_orien, islamo, socd_p, socd_n
    
    
    
    Sample size:
    715
    ******************** 종속변수 beh_p에 대한 massmedia x mass_media_qual의 조절효과 분석 결과********************
    
    ***************************** OUTCOME MODELS ****************************
    
    Outcome = beh_p 
    OLS Regression Summary
    
         R²  Adj. R²    MSE       F  df1  df2  p-value
     0.6428   0.6255 0.2811 38.3611   32  682   0.0000
    
    Coefficients
    
                                coeff     se       t      p    LLCI    ULCI
    Cons                       2.2558 0.3782  5.9640 0.0000  1.5144  2.9971
    massmedia                 -0.1550 0.0561 -2.7632 0.0059 -0.2650 -0.0451
    mass_media_qual           -0.0288 0.0489 -0.5892 0.5559 -0.1247  0.0670
    massmedia*mass_media_qual  0.0435 0.0182  2.3834 0.0174  0.0077  0.0792
    gender                    -0.0221 0.0432 -0.5124 0.6085 -0.1068  0.0625
    age                        0.0026 0.0023  1.1437 0.2532 -0.0019  0.0072
    child                      0.0876 0.0594  1.4754 0.1406 -0.0288  0.2041
    region_1                   0.0849 0.0423  2.0079 0.0450  0.0020  0.1677
    region_2                  -0.1258 0.1832 -0.6867 0.4925 -0.4849  0.2333
    edulevel                   0.0200 0.0138  1.4481 0.1480 -0.0071  0.0470
    income                    -0.0081 0.0109 -0.7434 0.4575 -0.0296  0.0133
    religion_1                -0.0684 0.0610 -1.1223 0.2621 -0.1879  0.0511
    religion_2                -0.0771 0.0553 -1.3935 0.1639 -0.1856  0.0313
    T_realf                   -0.2452 0.0358 -6.8557 0.0000 -0.3153 -0.1751
    T_symbf                   -0.0959 0.0292 -3.2874 0.0011 -0.1531 -0.0387
    T_safecoh2                -0.0601 0.0399 -1.5051 0.1328 -0.1383  0.0182
    T_health                   0.0142 0.0264  0.5375 0.5911 -0.0376  0.0660
    dir_contact                0.0915 0.0299  3.0649 0.0023  0.0330  0.1500
    sns                        0.0589 0.0204  2.8866 0.0040  0.0189  0.0990
    poli_pers                 -0.0097 0.0285 -0.3402 0.7338 -0.0655  0.0461
    relig_pers                 0.1643 0.0269  6.1164 0.0000  0.1117  0.2170
    econo                      0.1445 0.0242  5.9637 0.0000  0.0970  0.1919
    fake                       0.0515 0.0220  2.3428 0.0194  0.0084  0.0945
    humani                     0.1785 0.0331  5.3991 0.0000  0.1137  0.2433
    nat_eth2                   0.0152 0.0316  0.4798 0.6315 -0.0468  0.0771
    nat_cit2                  -0.0598 0.0373 -1.6052 0.1089 -0.1328  0.0132
    SDO_D                      0.0184 0.0345  0.5335 0.5938 -0.0492  0.0860
    SDO_E                     -0.0656 0.0283 -2.3156 0.0209 -0.1211 -0.0101
    rightw                    -0.0386 0.0653 -0.5914 0.5545 -0.1666  0.0894
    pol_orien                 -0.0335 0.0264 -1.2667 0.2057 -0.0853  0.0183
    islamo                    -0.0122 0.0231 -0.5277 0.5979 -0.0576  0.0331
    socd_p                     0.0249 0.0383  0.6497 0.5161 -0.0502  0.1000
    socd_n                     0.0545 0.0317  1.7222 0.0855 -0.0075  0.1166
    
    -------------------------------------------------------------------------
    
    
    ********************** CONDITIONAL EFFECTS **********************
    
    Conditional effect(s) of massmedia on beh_p at values of the moderator(s):
    
      mass_media_qual  Effect     SE       t      p    LLCI    ULCI
               1.0000 -0.1115 0.0405 -2.7526 0.0061 -0.1910 -0.0321
               2.0000 -0.0681 0.0283 -2.4030 0.0165 -0.1236 -0.0126
               3.0000 -0.0246 0.0251 -0.9818 0.3266 -0.0737  0.0245
               4.0000  0.0189 0.0335  0.5638 0.5731 -0.0467  0.0844
               5.0000  0.0623 0.0477  1.3067 0.1918 -0.0312  0.1558
    
    Process successfully initialized.
    Based on the Process Macro by Andrew F. Hayes, Ph.D. (www.afhayes.com)
    
    
    ****************************** SPECIFICATION ****************************
    
    Model = 1
    
    Variables:
        Cons = Cons
        x = massmedia
        m = mass_media_qual
        y = beh_n
    Statistical Controls:
     gender, age, child, region_1, region_2, edulevel, income, religion_1, religion_2, T_realf, T_symbf, T_safecoh2, T_health, dir_contact, sns, poli_pers, relig_pers, econo, fake, humani, nat_eth2, nat_cit2, SDO_D, SDO_E, rightw, pol_orien, islamo, socd_p, socd_n
    
    
    
    Sample size:
    715
    ******************** 종속변수 beh_n에 대한 massmedia x mass_media_qual의 조절효과 분석 결과********************
    
    ***************************** OUTCOME MODELS ****************************
    
    Outcome = beh_n 
    OLS Regression Summary
    
         R²  Adj. R²    MSE      F  df1  df2  p-value
     0.3074   0.2738 0.7121 9.4590   32  682   0.0000
    
    Coefficients
    
                                coeff     se       t      p    LLCI    ULCI
    Cons                       0.5392 0.6020  0.8956 0.3708 -0.6407  1.7191
    massmedia                  0.0721 0.0893  0.8075 0.4197 -0.1029  0.2471
    mass_media_qual           -0.0394 0.0778 -0.5055 0.6134 -0.1919  0.1132
    massmedia*mass_media_qual -0.0225 0.0290 -0.7735 0.4395 -0.0793  0.0344
    gender                    -0.0549 0.0687 -0.7990 0.4246 -0.1896  0.0798
    age                       -0.0027 0.0037 -0.7274 0.4672 -0.0099  0.0045
    child                      0.1677 0.0946  1.7736 0.0766 -0.0176  0.3530
    region_1                   0.0502 0.0673  0.7467 0.4555 -0.0816  0.1820
    region_2                  -0.0072 0.2916 -0.0248 0.9802 -0.5788  0.5643
    edulevel                   0.0430 0.0219  1.9620 0.0502  0.0000  0.0860
    income                    -0.0115 0.0174 -0.6589 0.5102 -0.0456  0.0226
    religion_1                 0.0737 0.0970  0.7597 0.4477 -0.1165  0.2639
    religion_2                -0.1354 0.0881 -1.5374 0.1247 -0.3081  0.0372
    T_realf                   -0.0618 0.0569 -1.0862 0.2778 -0.1734  0.0497
    T_symbf                   -0.0128 0.0464 -0.2754 0.7831 -0.1038  0.0782
    T_safecoh2                 0.2602 0.0635  4.0968 0.0000  0.1357  0.3847
    T_health                   0.0128 0.0421  0.3044 0.7609 -0.0696  0.0952
    dir_contact                0.1706 0.0475  3.5893 0.0004  0.0774  0.2637
    sns                        0.0173 0.0325  0.5316 0.5952 -0.0464  0.0810
    poli_pers                 -0.1011 0.0453 -2.2324 0.0259 -0.1899 -0.0123
    relig_pers                 0.0674 0.0428  1.5766 0.1154 -0.0164  0.1512
    econo                      0.0590 0.0386  1.5315 0.1261 -0.0165  0.1346
    fake                       0.1485 0.0350  4.2491 0.0000  0.0800  0.2171
    humani                    -0.1610 0.0526 -3.0601 0.0023 -0.2642 -0.0579
    nat_eth2                   0.0544 0.0503  1.0813 0.2800 -0.0442  0.1530
    nat_cit2                  -0.0563 0.0593 -0.9499 0.3425 -0.1725  0.0599
    SDO_D                      0.1331 0.0549  2.4251 0.0156  0.0255  0.2406
    SDO_E                     -0.0194 0.0451 -0.4309 0.6667 -0.1078  0.0689
    rightw                     0.1269 0.1039  1.2215 0.2223 -0.0767  0.3306
    pol_orien                 -0.0312 0.0421 -0.7420 0.4584 -0.1137  0.0513
    islamo                     0.0342 0.0368  0.9288 0.3533 -0.0380  0.1064
    socd_p                     0.0913 0.0610  1.4981 0.1346 -0.0282  0.2109
    socd_n                     0.0233 0.0504  0.4616 0.6445 -0.0755  0.1221
    
    -------------------------------------------------------------------------
    
    
    ********************** CONDITIONAL EFFECTS **********************
    
    Conditional effect(s) of massmedia on beh_n at values of the moderator(s):
    
      mass_media_qual  Effect     SE       t      p    LLCI   ULCI
               1.0000  0.0496 0.0645  0.7697 0.4418 -0.0768 0.1761
               2.0000  0.0272 0.0451  0.6030 0.5467 -0.0612 0.1156
               3.0000  0.0047 0.0399  0.1187 0.9055 -0.0735 0.0829
               4.0000 -0.0177 0.0532 -0.3328 0.7394 -0.1221 0.0866
               5.0000 -0.0402 0.0759 -0.5291 0.5969 -0.1890 0.1086
    
    Process successfully initialized.
    Based on the Process Macro by Andrew F. Hayes, Ph.D. (www.afhayes.com)
    
    
    ****************************** SPECIFICATION ****************************
    
    Model = 1
    
    Variables:
        Cons = Cons
        x = sns
        m = sns_qual
        y = beh_p
    Statistical Controls:
     gender, age, child, region_1, region_2, edulevel, income, religion_1, religion_2, T_realf, T_symbf, T_safecoh2, T_health, dir_contact, massmedia, poli_pers, relig_pers, econo, fake, humani, nat_eth2, nat_cit2, SDO_D, SDO_E, rightw, pol_orien, islamo, socd_p, socd_n
    
    
    
    Sample size:
    591
    ******************** 종속변수 beh_p에 대한 sns x sns_qual의 조절효과 분석 결과********************
    
    ***************************** OUTCOME MODELS ****************************
    
    Outcome = beh_p 
    OLS Regression Summary
    
         R²  Adj. R²    MSE       F  df1  df2  p-value
     0.6592   0.6391 0.2795 33.7347   32  558   0.0000
    
    Coefficients
    
                   coeff     se       t      p    LLCI    ULCI
    Cons          1.9209 0.4096  4.6900 0.0000  1.1181  2.7236
    sns           0.0568 0.0548  1.0366 0.3004 -0.0506  0.1643
    sns_qual      0.0628 0.0403  1.5581 0.1198 -0.0162  0.1418
    sns*sns_qual -0.0019 0.0171 -0.1117 0.9111 -0.0355  0.0316
    gender       -0.0006 0.0476 -0.0124 0.9901 -0.0939  0.0928
    age           0.0030 0.0026  1.1303 0.2588 -0.0022  0.0081
    child         0.1260 0.0670  1.8818 0.0604 -0.0052  0.2573
    region_1      0.0855 0.0467  1.8332 0.0673 -0.0059  0.1770
    region_2     -0.1451 0.2063 -0.7032 0.4822 -0.5495  0.2593
    edulevel      0.0228 0.0151  1.5049 0.1329 -0.0069  0.0525
    income       -0.0068 0.0119 -0.5663 0.5714 -0.0302  0.0166
    religion_1   -0.0187 0.0674 -0.2775 0.7815 -0.1507  0.1133
    religion_2   -0.0286 0.0606 -0.4719 0.6372 -0.1474  0.0902
    T_realf      -0.2671 0.0391 -6.8324 0.0000 -0.3437 -0.1905
    T_symbf      -0.0930 0.0326 -2.8525 0.0045 -0.1570 -0.0291
    T_safecoh2   -0.0349 0.0439 -0.7964 0.4261 -0.1209  0.0510
    T_health     -0.0117 0.0298 -0.3934 0.6941 -0.0702  0.0467
    dir_contact   0.0886 0.0306  2.8943 0.0039  0.0286  0.1485
    massmedia    -0.0475 0.0283 -1.6817 0.0932 -0.1029  0.0079
    poli_pers    -0.0210 0.0313 -0.6719 0.5019 -0.0823  0.0403
    relig_pers    0.1713 0.0295  5.8171 0.0000  0.1136  0.2290
    econo         0.1532 0.0273  5.6191 0.0000  0.0997  0.2066
    fake          0.0420 0.0242  1.7381 0.0827 -0.0054  0.0893
    humani        0.1837 0.0358  5.1302 0.0000  0.1135  0.2539
    nat_eth2      0.0087 0.0359  0.2407 0.8099 -0.0618  0.0791
    nat_cit2     -0.0592 0.0413 -1.4334 0.1523 -0.1401  0.0217
    SDO_D         0.0523 0.0387  1.3516 0.1770 -0.0236  0.1282
    SDO_E        -0.0503 0.0305 -1.6468 0.1002 -0.1101  0.0096
    rightw       -0.0352 0.0743 -0.4743 0.6355 -0.1808  0.1104
    pol_orien    -0.0405 0.0294 -1.3768 0.1691 -0.0982  0.0172
    islamo       -0.0077 0.0247 -0.3128 0.7545 -0.0562  0.0407
    socd_p        0.0195 0.0425  0.4592 0.6462 -0.0638  0.1029
    socd_n        0.0487 0.0353  1.3787 0.1686 -0.0205  0.1179
    
    -------------------------------------------------------------------------
    
    
    ********************** CONDITIONAL EFFECTS **********************
    
    Conditional effect(s) of sns on beh_p at values of the moderator(s):
    
      sns_qual  Effect     SE      t      p    LLCI   ULCI
        1.0000  0.0549 0.0403 1.3616 0.1739 -0.0241 0.1340
        2.0000  0.0530 0.0289 1.8354 0.0670 -0.0036 0.1096
        3.0000  0.0511 0.0251 2.0398 0.0418  0.0020 0.1002
        4.0000  0.0492 0.0317 1.5498 0.1218 -0.0130 0.1114
        5.0000  0.0473 0.0444 1.0642 0.2877 -0.0398 0.1344
    
    Process successfully initialized.
    Based on the Process Macro by Andrew F. Hayes, Ph.D. (www.afhayes.com)
    
    
    ****************************** SPECIFICATION ****************************
    
    Model = 1
    
    Variables:
        Cons = Cons
        x = sns
        m = sns_qual
        y = beh_n
    Statistical Controls:
     gender, age, child, region_1, region_2, edulevel, income, religion_1, religion_2, T_realf, T_symbf, T_safecoh2, T_health, dir_contact, massmedia, poli_pers, relig_pers, econo, fake, humani, nat_eth2, nat_cit2, SDO_D, SDO_E, rightw, pol_orien, islamo, socd_p, socd_n
    
    
    
    Sample size:
    591
    ******************** 종속변수 beh_n에 대한 sns x sns_qual의 조절효과 분석 결과********************
    
    ***************************** OUTCOME MODELS ****************************
    
    Outcome = beh_n 
    OLS Regression Summary
    
         R²  Adj. R²    MSE      F  df1  df2  p-value
     0.3180   0.2776 0.7263 8.1317   32  558   0.0000
    
    Coefficients
    
                   coeff     se       t      p    LLCI    ULCI
    Cons          0.2240 0.6603  0.3392 0.7346 -1.0701  1.5180
    sns           0.1085 0.0884  1.2275 0.2201 -0.0647  0.2818
    sns_qual      0.0259 0.0650  0.3978 0.6909 -0.1015  0.1532
    sns*sns_qual -0.0401 0.0276 -1.4529 0.1468 -0.0942  0.0140
    gender       -0.0821 0.0768 -1.0698 0.2852 -0.2326  0.0683
    age          -0.0063 0.0042 -1.4907 0.1366 -0.0145  0.0020
    child         0.2647 0.1080  2.4519 0.0145  0.0531  0.4763
    region_1      0.1358 0.0752  1.8059 0.0715 -0.0116  0.2833
    region_2      0.3537 0.3326  1.0632 0.2882 -0.2983  1.0056
    edulevel      0.0345 0.0244  1.4135 0.1581 -0.0133  0.0824
    income        0.0039 0.0193  0.2036 0.8387 -0.0338  0.0417
    religion_1    0.0423 0.1086  0.3899 0.6967 -0.1705  0.2552
    religion_2   -0.1105 0.0977 -1.1316 0.2583 -0.3020  0.0809
    T_realf      -0.0772 0.0630 -1.2254 0.2210 -0.2007  0.0463
    T_symbf      -0.0281 0.0526 -0.5345 0.5932 -0.1312  0.0750
    T_safecoh2    0.2971 0.0707  4.2021 0.0000  0.1585  0.4356
    T_health      0.0508 0.0481  1.0563 0.2913 -0.0434  0.1450
    dir_contact   0.1760 0.0493  3.5676 0.0004  0.0793  0.2727
    massmedia     0.0109 0.0455  0.2401 0.8103 -0.0783  0.1002
    poli_pers    -0.0973 0.0504 -1.9316 0.0539 -0.1961  0.0014
    relig_pers    0.0790 0.0475  1.6650 0.0965 -0.0140  0.1721
    econo         0.0618 0.0439  1.4071 0.1600 -0.0243  0.1480
    fake          0.1560 0.0389  4.0057 0.0001  0.0797  0.2323
    humani       -0.1295 0.0577 -2.2431 0.0253 -0.2427 -0.0163
    nat_eth2      0.1192 0.0579  2.0564 0.0402  0.0056  0.2327
    nat_cit2     -0.1279 0.0666 -1.9217 0.0552 -0.2584  0.0025
    SDO_D         0.1294 0.0624  2.0732 0.0386  0.0071  0.2517
    SDO_E        -0.0199 0.0492 -0.4035 0.6867 -0.1163  0.0766
    rightw        0.1369 0.1197  1.1430 0.2535 -0.0978  0.3716
    pol_orien    -0.0434 0.0474 -0.9144 0.3609 -0.1364  0.0496
    islamo        0.0392 0.0399  0.9834 0.3258 -0.0389  0.1173
    socd_p        0.0816 0.0685  1.1900 0.2346 -0.0528  0.2159
    socd_n        0.0081 0.0569  0.1430 0.8864 -0.1035  0.1197
    
    -------------------------------------------------------------------------
    
    
    ********************** CONDITIONAL EFFECTS **********************
    
    Conditional effect(s) of sns on beh_n at values of the moderator(s):
    
      sns_qual  Effect     SE       t      p    LLCI   ULCI
        1.0000  0.0684 0.0650  1.0519 0.2933 -0.0591 0.1959
        2.0000  0.0283 0.0466  0.6077 0.5436 -0.0630 0.1196
        3.0000 -0.0118 0.0404 -0.2924 0.7701 -0.0910 0.0674
        4.0000 -0.0519 0.0512 -1.0146 0.3107 -0.1522 0.0484
        5.0000 -0.0920 0.0716 -1.2849 0.1994 -0.2324 0.0483
    


    /usr/local/lib/python3.10/dist-packages/ipykernel/ipkernel.py:283: DeprecationWarning: `should_run_async` will not call `transform_cell` automatically in the future. Please pass the result to `transformed_cell` argument and any exception that happen during thetransform in `preprocessing_exc_tuple` in IPython 7.17 and above.
      and should_run_async(code)
    <frozen importlib._bootstrap>:914: ImportWarning: APICoreClientInfoImportHook.find_spec() not found; falling back to find_module()
    <frozen importlib._bootstrap>:914: ImportWarning: _PyDriveImportHook.find_spec() not found; falling back to find_module()
    <frozen importlib._bootstrap>:914: ImportWarning: _OpenCVImportHook.find_spec() not found; falling back to find_module()
    <frozen importlib._bootstrap>:914: ImportWarning: _BokehImportHook.find_spec() not found; falling back to find_module()
    <frozen importlib._bootstrap>:914: ImportWarning: _AltairImportHook.find_spec() not found; falling back to find_module()



```python
# DATA LIST FREE/
#    massmedi   mass_qua   beh_p      .
# BEGIN DATA.
#      2.0000     2.0000     2.3185
#      3.0000     2.0000     2.2505
#      4.0000     2.0000     2.1824
#      2.0000     3.0000     2.3332
#      3.0000     3.0000     2.3086
#      4.0000     3.0000     2.2840
#      2.0000     4.0000     2.3478
#      3.0000     4.0000     2.3667
#      4.0000     4.0000     2.3856
# END DATA.
# GRAPH/SCATTERPLOT=
#  massmedi WITH     beh_p    BY       mass_qua .

import matplotlib.pyplot as plt
plt.rc('font', family='NanumBarunGothic')

# 데이터 설정
massmedi = [2.0000, 3.0000, 4.0000, 2.0000, 3.0000, 4.0000, 2.0000, 3.0000, 4.0000]
mass_qua = [2.0000, 2.0000, 2.0000, 3.0000, 3.0000, 3.0000, 4.0000, 4.0000, 4.0000]
beh_p = [2.3185, 2.2505, 2.1824, 2.3332, 2.3086, 2.2840, 2.3478, 2.3667, 2.3856]

# Scatter plot 그리기
scatter = plt.scatter(massmedi, beh_p, c=mass_qua, cmap='viridis')  # massmedi를 x축으로, beh_p를 y축으로 설정하고, mass_qua에 따라 색상을 다르게 표시

# 선 그래프 그리기 # linestyle ='-.'
for i in range(len(massmedi) - 1):
    if mass_qua[i] == 2 and mass_qua[i+1] == 2:  # Connect yellow scatterplot points with yellow line
        plt.plot([massmedi[i], massmedi[i + 1]], [beh_p[i], beh_p[i + 1]], c='#440154FF', linewidth=2)
    elif mass_qua[i] == 3 and mass_qua[i+1] == 3:  # Connect green scatterplot points with green line
        plt.plot([massmedi[i], massmedi[i + 1]], [beh_p[i], beh_p[i + 1]], c='#287C8EFF', linewidth=2)
    elif mass_qua[i] == 4 and mass_qua[i+1] == 4:  # Connect blue scatterplot points with blue line
        plt.plot([massmedi[i], massmedi[i + 1]], [beh_p[i], beh_p[i + 1]], c='#FDE725FF', linewidth=2)


# 축과 제목 설정
plt.xlabel('대중매체 접촉 빈도')
plt.ylabel('긍정적 행동의사')
plt.title('대중매체 접촉 빈도 X 대중매체 접촉의 질')

# 범례 설정
plt.colorbar(scatter, ticks=[2, 3, 4], label='대중매체 접촉의 질')  # 범례에 2, 3, 4만 표시

# 그래프 출력
plt.show()

# 그림으로 저장
# plt.savefig('/content/drive/MyDrive/Analysis_0602/moderating_graph.png')
```

    /usr/local/lib/python3.10/dist-packages/ipykernel/ipkernel.py:283: DeprecationWarning: `should_run_async` will not call `transform_cell` automatically in the future. Please pass the result to `transformed_cell` argument and any exception that happen during thetransform in `preprocessing_exc_tuple` in IPython 7.17 and above.
      and should_run_async(code)
    <frozen importlib._bootstrap>:914: ImportWarning: APICoreClientInfoImportHook.find_spec() not found; falling back to find_module()
    <frozen importlib._bootstrap>:914: ImportWarning: _PyDriveImportHook.find_spec() not found; falling back to find_module()
    <frozen importlib._bootstrap>:914: ImportWarning: _OpenCVImportHook.find_spec() not found; falling back to find_module()
    <frozen importlib._bootstrap>:914: ImportWarning: _BokehImportHook.find_spec() not found; falling back to find_module()
    <frozen importlib._bootstrap>:914: ImportWarning: _AltairImportHook.find_spec() not found; falling back to find_module()



    
![png](diaspora_analysis_0602_files/diaspora_analysis_0602_9_1.png)
    



```python
# Scatter plot 그리기
scatter = plt.scatter(massmedi, beh_p, color='black')  # massmedi를 x축으로, beh_p를 y축으로 설정하고, 점 색상을 검은색으로 표시

# 선 그래프 그리기
lines_dict = {}
for i in range(len(massmedi) - 1):
    if mass_qua[i] == 2 and mass_qua[i + 1] == 2:
        line, = plt.plot([massmedi[i], massmedi[i + 1]], [beh_p[i], beh_p[i + 1]], linewidth=2, color='black')
        lines_dict.setdefault('대중매체 접촉의 질: 2', line)
    elif mass_qua[i] == 3 and mass_qua[i + 1] == 3:
        line, = plt.plot([massmedi[i], massmedi[i + 1]], [beh_p[i], beh_p[i + 1]], linewidth=2, linestyle='--', color='black')
        lines_dict.setdefault('대중매체 접촉의 질: 3', line)
    elif mass_qua[i] == 4 and mass_qua[i + 1] == 4:
        line, = plt.plot([massmedi[i], massmedi[i + 1]], [beh_p[i], beh_p[i + 1]], linewidth=2, linestyle='-.', color='black')
        lines_dict.setdefault('대중매체 접촉의 질: 4', line)

# 축과 제목 설정
plt.xlabel('대중매체 접촉 빈도')
plt.ylabel('긍정적 행동의사')
# plt.title('대중매체 접촉 빈도 X 대중매체 접촉의 질')

# 범례 설정
lines = list(lines_dict.values())
labels = list(lines_dict.keys())
plt.legend(list(reversed(lines)), list(reversed(labels)))

# 그래프 출력
plt.show()
```


    
![png](diaspora_analysis_0602_files/diaspora_analysis_0602_10_0.png)

