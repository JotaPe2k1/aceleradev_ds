#!/usr/bin/env python
# coding: utf-8

# # Desafio 4
# 
# Neste desafio, vamos praticar um pouco sobre testes de hipóteses. Utilizaremos o _data set_ [2016 Olympics in Rio de Janeiro](https://www.kaggle.com/rio2016/olympic-games/), que contém dados sobre os atletas das Olimpíadas de 2016 no Rio de Janeiro.
# 
# Esse _data set_ conta com informações gerais sobre 11538 atletas como nome, nacionalidade, altura, peso e esporte praticado. Estaremos especialmente interessados nas variáveis numéricas altura (`height`) e peso (`weight`). As análises feitas aqui são parte de uma Análise Exploratória de Dados (EDA).
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns


# %matplotlib inline
# 
# from IPython.core.pylabtools import figsize
# 
# 
# figsize(12, 8)
# 
# sns.set()

# In[2]:


athletes = pd.read_csv("athletes.csv")


# In[3]:


def get_sample(df, col_name, n=100, seed=42):
    """Get a sample from a column of a dataframe.
    
    It drops any numpy.nan entries before sampling. The sampling
    is performed without replacement.
    
    Example of numpydoc for those who haven't seen yet.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Source dataframe.
    col_name : str
        Name of the column to be sampled.
    n : int
        Sample size. Default is 100.
    seed : int
        Random seed. Default is 42.
    
    Returns
    -------
    pandas.Series
        Sample of size n from dataframe's column.
    """
    np.random.seed(seed)
    
    random_idx = np.random.choice(df[col_name].dropna().index, size=n, replace=False)
    
    return df.loc[random_idx, col_name]


# ## Inicia sua análise a partir daqui

# In[4]:


# Sua análise começa aqui.
df = athletes.loc[:,:]
df.head()


# In[5]:


df.describe()


# In[6]:


df.info()


# In[7]:


df.isnull().sum()


# ## Questão 1
# 
# Considerando uma amostra de tamanho 3000 da coluna `height` obtida com a função `get_sample()`, execute o teste de normalidade de Shapiro-Wilk com a função `scipy.stats.shapiro()`. Podemos afirmar que as alturas são normalmente distribuídas com base nesse teste (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).

# In[8]:


def q1():
    sample = get_sample(df, 'height', 3000)
    return True if sct.shapiro(sample)[1] > 0.05 else False


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Plote o qq-plot para essa variável e a analise.
# * Existe algum nível de significância razoável que nos dê outro resultado no teste? (Não faça isso na prática. Isso é chamado _p-value hacking_, e não é legal).

# sample = get_sample(df, 'height', 3000)
# print(f"mean {sample.mean()}, std {sample.std()}")
# sns.distplot(sample)
# plt.show()

# plt.hist(get_sample(df, 'height', 3000), bins=25,)
# plt.show() # São condizentes, o gráfico se aproxima de uma distribuição normal mas não o suficiente para ser considerada uma

# import statsmodels.api as sm
# import pylab as py
# sm.qqplot(get_sample(df, 'height', 3000), line="s")
# py.show()

# ## Questão 2
# 
# Repita o mesmo procedimento acima, mas agora utilizando o teste de normalidade de Jarque-Bera através da função `scipy.stats.jarque_bera()`. Agora podemos afirmar que as alturas são normalmente distribuídas (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).

# In[9]:


def q2():
    sample = get_sample(df, 'height', 3000)
    return True if sct.jarque_bera(sample)[1] > 0.05 else False


# __Para refletir__:
# 
# * Esse resultado faz sentido?

# Sim, bem como o resultado anterior a este

# ## Questão 3
# 
# Considerando agora uma amostra de tamanho 3000 da coluna `weight` obtida com a função `get_sample()`. Faça o teste de normalidade de D'Agostino-Pearson utilizando a função `scipy.stats.normaltest()`. Podemos afirmar que os pesos vêm de uma distribuição normal ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# In[10]:


def q3():
    sample = get_sample(df, 'weight', 3000)
    return True if sct.normaltest(sample)[1] > 0.05 else False


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Um _box plot_ também poderia ajudar a entender a resposta.

# sample = get_sample(df, 'weight', 3000)
# plt.hist(sample)
# plt.show()
# sns.distplot(sample)
# plt.show()
# sns.boxplot(sample)
# plt.show()
# Resultado condizente, claramente essa curva não é simétrica.

# ## Questão 4
# 
# Realize uma transformação logarítmica em na amostra de `weight` da questão 3 e repita o mesmo procedimento. Podemos afirmar a normalidade da variável transformada ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# In[11]:


def q4():
    df['weight_log'] = np.log(df.loc[:,'weight'])
    sample = get_sample(df, 'weight_log', 3000)
    return True if sct.normaltest(sample)[1] > 0.05 else False


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Você esperava um resultado diferente agora?

# q4()
# df['weight_log'].hist(bins=25)
# plt.show() # Confesso que fiquei em dúvida no começo, mas realmente não é uma distribuição normal

# > __Para as questão 5 6 e 7 a seguir considere todos testes efetuados ao nível de significância de 5%__.

# ## Questão 5
# 
# Obtenha todos atletas brasileiros, norte-americanos e canadenses em `DataFrame`s chamados `bra`, `usa` e `can`,respectivamente. Realize um teste de hipóteses para comparação das médias das alturas (`height`) para amostras independentes e variâncias diferentes com a função `scipy.stats.ttest_ind()` entre `bra` e `usa`. Podemos afirmar que as médias são estatisticamente iguais? Responda com um boolean (`True` ou `False`).

# In[12]:


bra = df.loc[df['nationality'] == 'BRA']['height']
usa = df.loc[df['nationality'] == 'USA']['height']
can = df.loc[df['nationality'] == 'CAN']['height']


# In[13]:


def q5():
    p_value = sct.ttest_ind(bra, usa, equal_var = False, nan_policy='omit')[1]
    return True if p_value > 0.05 else False


# ## Questão 6
# 
# Repita o procedimento da questão 5, mas agora entre as alturas de `bra` e `can`. Podemos afimar agora que as médias são estatisticamente iguais? Reponda com um boolean (`True` ou `False`).

# In[14]:


def q6():
    p_value = sct.ttest_ind(bra, can, equal_var = False, nan_policy='omit')[1]
    return True if p_value > 0.05 else False


# ## Questão 7
# 
# Repita o procedimento da questão 6, mas agora entre as alturas de `usa` e `can`. Qual o valor do p-valor retornado? Responda como um único escalar arredondado para oito casas decimais.

# In[15]:


def q7():
    p_value = sct.ttest_ind(usa, can, equal_var = False, nan_policy='omit')[1]
    return round(float(p_value), 8)


# __Para refletir__:
# 
# * O resultado faz sentido?
# * Você consegue interpretar esse p-valor?
# * Você consegue chegar a esse valor de p-valor a partir da variável de estatística?

# Sim, faz sentido, o valor de p é a margem de erro assumido ao dizer que H0 seria rejeitado e passaríamos a considerar H1, por mais que as médias se aproximem, seria incorreto dizer que a distribuição que forma a média é igual em caso do teste alegar False

# In[ ]:




