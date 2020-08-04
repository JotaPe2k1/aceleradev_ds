#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[39]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import (KBinsDiscretizer, OneHotEncoder, StandardScaler)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import (CountVectorizer, TfidfTransformer, TfidfVectorizer)


# In[2]:


countries = pd.read_csv("countries.csv")


# In[3]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head(5)


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# In[4]:


df = countries.copy()
df.isnull().sum()


# In[5]:


df.info()


# In[6]:


df.head()


# ### Removendo espaços desnecessários

# In[7]:


df["Country"] = df["Country"].str.strip()
df["Region"] = df["Region"].str.strip()


# ### convertendo os numeros que estão como categóricos

# In[8]:


to_num = ['Pop_density', 'Coastline_ratio', 'Net_migration', 'Infant_mortality',
 'Literacy', 'Phones_per_1000', 'Arable', 'Crops', 'Other', 'Climate', 'Birthrate',
 'Deathrate', 'Agriculture', 'Industry', 'Service']


# In[9]:


for var in to_num:
    df[var] = pd.to_numeric(df[var].str.replace(',', '.'))


# In[10]:


df.info()


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[11]:


def q1():
    return sorted(df['Region'].unique())


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[12]:


def q2():
    disc = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
    x = df[['Pop_density']] ; y = df['Country']
    disc_quantile_90 = disc.fit_transform(x, y) >= 9.
    return int(sum(disc_quantile_90))


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[13]:


def q3():
    enc = OneHotEncoder(sparse=False, dtype=np.int)
    target = countries[['Region', 'Climate']].fillna('NaN')
    target_encoded = enc.fit_transform(target)
    return target_encoded.shape[1]


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[14]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[35]:


def q4():
    number_var = df[list(df.select_dtypes(include='number').columns)]
    # Substituição de nulos e padronização de valores
    number_pip = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                   ('scaler', StandardScaler())])
    # Aplicando pipeline nas variaveis numericas
    number_var_pip = number_pip.fit_transform(number_var)
    pipeline_median = pd.DataFrame(number_var_pip, columns=number_var.columns)
    # Aplicando pipeline no test_country
    test_country_pip = number_pip.transform([test_country[2:]])
    position_pip = pd.DataFrame(test_country_pip, columns=number_var.columns)
    arable = position_pip['Arable']
    return float(arable.round(3))


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[57]:


def q5():
    # Criando uma série com a coluna Net_migration
    net_migration = df['Net_migration'].dropna()
    # calculando o iqr
    q1 = net_migration.quantile(0.25)
    q3 = net_migration.quantile(0.75)
    iqr = q3 - q1
    # Encontrando o intervalo normal
    normal_iqr = [q1 - 1.5 * iqr, q3 + 1.5 * iqr]
    # Encontrando os outliers
    bellow_outliers = net_migration[(net_migration < normal_iqr[0])]
    above_outliers = net_migration[(net_migration > normal_iqr[1])]
    return (len(bellow_outliers), len(above_outliers), False)


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[41]:


categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)


# In[42]:


def q6():
    vec = CountVectorizer()
    vec_sum = vec.fit_transform(newsgroup.data)
    vec_words = vec.vocabulary_
    return int(vec_sum[:, vec_words['phone']].sum())


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[ ]:


def q7():
    tfidf = TfidfVectorizer().fit(newsgroup.data)
    tfidf_sum = tfidf.transform(newsgroup.data)
    tfidf_words = tfidf.vocabulary_
    return float(tfidf_sum[:, tfidf_words['phone']].sum().round(3))

