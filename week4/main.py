import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF

"""
%matplotlib inline
from IPython.core.pylabtools import figsize
figsize(12, 8)
sns.set()
"""

np.random.seed(42)
    
dataframe = pd.DataFrame({"normal": sct.norm.rvs(20, 4, size=10000),
                          "binomial": sct.binom.rvs(100, 0.2, size=10000)})


# ## Inicie sua análise a partir da parte 1 a partir daqui

dataframe.head()
norm = dataframe["normal"]
binom = dataframe["binomial"]
sns.distplot(norm)


# ## Questão 1
# 
# Qual a diferença entre os quartis (Q1, Q2 e Q3) das variáveis `normal` e `binomial` de `dataframe`? Responda como
# uma tupla de três elementos arredondados para três casas decimais.
# 
# Em outra palavras, sejam `q1_norm`, `q2_norm` e `q3_norm` os quantis da variável `normal` e `q1_binom`, `q2_binom` e
# `q3_binom` os quantis da variável `binom`, qual a diferença
# `(q1_norm - q1 binom, q2_norm - q2_binom, q3_norm - q3_binom)`?
def q1():
    qnorm = norm.quantile([0.25, 0.50, 0.75])
    qbinom = binom.quantile([0.25, 0.50, 0.75])
    return tuple(round(qnorm - qbinom, 3))


# Para refletir:
# 
# * Você esperava valores dessa magnitude?
# 
# * Você é capaz de explicar como distribuições aparentemente tão diferentes (discreta e contínua, por exemplo)
# conseguem dar esses valores?


# ## Questão 2
# 
# Considere o intervalo $[\bar{x} - s, \bar{x} + s]$, onde $\bar{x}$ é a média amostral e $s$ é o desvio padrão.
# Qual a probabilidade nesse intervalo, calculada pela função de distribuição acumulada empírica
# (CDF empírica) da variável `normal`? Responda como uma único escalar arredondado para três casas decimais.
def q2():
    x, s = norm.mean(), norm.std()
    ecdf = ECDF(norm)
    probability = float(round((ecdf(x + s) - ecdf(x - s)), 3))
    return probability


# Para refletir:
# 
# * Esse valor se aproxima do esperado teórico?
# * Experimente também para os intervalos $[\bar{x} - 2s, \bar{x} + 2s]$ e $[\bar{x} - 3s, \bar{x} + 3s]$.

# ## Questão 3
# 
# Qual é a diferença entre as médias e as variâncias das variáveis `binomial` e `normal`?
# Responda como uma tupla de dois elementos arredondados para três casas decimais.
# 
# Em outras palavras, sejam `m_binom` e `v_binom` a média e a variância da variável `binomial`,
# e `m_norm` e `v_norm` a média e a variância da variável `normal`. Quais as diferenças
# `(m_binom - m_norm, v_binom - v_norm)`?
def q3():
    m_norm, v_norm = norm.mean(), norm.var()
    m_binom, v_binom = binom.mean(), binom.var()
    return tuple([round(m_binom - m_norm, 3), round(v_binom - v_norm, 3)])

# Para refletir:
# 
# * Você esperava valore dessa magnitude?
# * Qual o efeito de aumentar ou diminuir $n$ (atualmente 100) na distribuição da variável `binomial`?


# ## Parte 2

# ### _Setup_ da parte 2
stars = pd.read_csv("pulsar_stars.csv")
stars.rename({old_name: new_name
              for (old_name, new_name)
              in zip(stars.columns, ["mean_profile", "sd_profile", "kurt_profile", "skew_profile", "mean_curve",
                                     "sd_curve", "kurt_curve", "skew_curve", "target"])}, axis=1, inplace=True)
stars.loc[:, "target"] = stars.target.astype(bool)


# ## Inicie sua análise da parte 2 a partir daqui

# Sua análise da parte 2 começa aqui.
stars.head()


# ## Questão 4
# 
# Considerando a variável `mean_profile` de `stars`:
# 
# 1. Filtre apenas os valores de `mean_profile` onde `target == 0` (ou seja, onde a estrela não é um pulsar).
# 2. Padronize a variável `mean_profile` filtrada anteriormente para ter média 0 e variância 1.
# 
# Chamaremos a variável resultante de `false_pulsar_mean_profile_standardized`.
# 
# Encontre os quantis teóricos para uma distribuição normal de média 0 e variância 1 para 0.80, 0.90 e 0.95
# através da função `norm.ppf()` disponível em `scipy.stats`.
# 
# Quais as probabilidade associadas a esses quantis utilizando a CDF empírica da variável
# `false_pulsar_mean_profile_standardized`? Responda como uma tupla de três elementos arredondados para
# três casas decimais.
def q4():
    false_pulsar = stars.query("target == False")['mean_profile']
    false_pulsar_mean_profile_standardized = pd.Series(sct.zscore(false_pulsar))
    quantile = sct.norm.ppf([0.8, 0.90, 0.95])
    ecdf = ECDF(false_pulsar_mean_profile_standardized)
    return tuple(ecdf(quantile).round(3))


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?

# ## Questão 5
# 
# Qual a diferença entre os quantis Q1, Q2 e Q3 de `false_pulsar_mean_profile_standardized` e os mesmos quantis
# teóricos de uma distribuição normal de média 0 e variância 1? Responda como uma tupla de três elementos
# arredondados para três casas decimais.
def q5():
    false_pulsar = stars.query("target == False")['mean_profile']
    false_pulsar_mean_profile_standardized = pd.Series(sct.zscore(false_pulsar))
    ppf_quantile = sct.norm.ppf([0.25, 0.50, 0.75])
    false_pmps_quantile = np.quantile(false_pulsar_mean_profile_standardized, [0.25, 0.50, 0.75])
    return tuple(np.round(false_pmps_quantile - ppf_quantile, 3))

# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?
# * Curiosidade: alguns testes de hipóteses sobre normalidade dos dados utilizam essa mesma abordagem.
