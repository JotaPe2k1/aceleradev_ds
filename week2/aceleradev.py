import streamlit as st
import pandas as pd

def main():
    st.image('squad.jpeg')
    st.title('AceleraDev Data Science')
    st.header('Pré-processamento de dados em Python')
    st.markdown("<h5>App feito na intensão de melhorar a visualização e elucidar as respostas dos exercícios propostos na segunda semana do aceleradev_ds na codenation!</h5>", unsafe_allow_html=True)
    file = st.file_uploader('Primeiro, selecione a base de dados (black_friday.csv).', type='csv')
    if file is not None:
        black_friday = pd.read_csv(file)
        st.markdown("<h5>A partir de agora, podemos ter de maniera simplificada e visual, como os dados estão dispostos.</h5>", unsafe_allow_html=True)
        st.header('Vizualização do DataFrame')
        st.markdown('<h5>Arraste para selecionar o numero de observações que deseja ver do DataFrame.<h5>', unsafe_allow_html=True)
        numeros = st.slider('Note que limitamos o tamanho da visualização para obtermos maior performance na execução do app', min_value=1, max_value=20)
        st.table(black_friday.head(numeros))
        st.subheader('Questão 1')
        st.markdown('Quantas observações e quantas colunas há no dataset?')
        st.markdown("<h4>Para essa questão o mais simples a se fazer é utilizar o atributo shape. Logo obteremos:</h4>", unsafe_allow_html=True)
        st.write(black_friday.shape)
        tamanho = pd.DataFrame({'observacoes': [black_friday.shape[0]], 'colunas': [black_friday.shape[1]]})
        st.table(tamanho)
        st.subheader('Questão 2')
        st.markdown('Há quantas mulheres com idade entre 26 e 35 anos no dataset?')
        st.markdown("<h4>Essa questão fez confusão na cabeça de muitos, o motivo seria o enunciado que da a entender que\
            a busca seria por ids unicos, quando na verdade a solução exige contagem repetida dos ids.Logo, a melhor escolha seria filtrar por duas condições, sendo elas.<br> Age == 26-35 e Gender == F.<br>Assim encontraremos o seguinte resultado:</h4>", unsafe_allow_html=True)
        st.write(black_friday[(black_friday['Gender'] == 'F') & (black_friday['Age'] == '26-35')].shape[0])
        qtd_mulheres = st.slider('Escolha o numero de observaçoes que deseja ver de mulheres com idade entre 26 e 35 anos', min_value=1, max_value=20)
        st.dataframe(black_friday[(black_friday['Gender'] == 'F') & (black_friday['Age'] == '26-35')].head(qtd_mulheres))
        st.subheader('Questão 3')
        st.markdown('Quantos usuários únicos há no dataset?')
        st.markdown("<h4>Para essa questão, utilizamos o método nunique do pandas.</h4>", unsafe_allow_html=True)
        st.write(black_friday['User_ID'].nunique())
        st.dataframe(black_friday['User_ID'].unique())
        st.subheader('Questão 4')
        st.markdown('Quantos tipos de dados diferentes existem no dataset?')
        st.markdown("<h4>Aqui, utilizaremos o mesmo raciocínio da questão anterior, já que queremos obter quantos valores exclusivos temos, ou seja, nunique novamente. Mas agora queremos saber os tipos, o que fazer?<br>Isso mesmo! dtypes. Tente você mesmo, df.dtypes.nunique()</h4>", unsafe_allow_html=True)
        st.write(black_friday.dtypes.nunique())
        exploracao = pd.DataFrame({'nomes': black_friday.columns, 'tipos': black_friday.dtypes, 'NA #': black_friday.isna().sum(),
                                   'NA %': (black_friday.isna().sum() / black_friday.shape[0]) * 100})
        st.write(exploracao.tipos.value_counts())
        st.markdown('**Nomes das colunas do tipo int64:**')
        st.markdown(list(exploracao[exploracao['tipos'] == 'int64']['nomes']))
        st.markdown('**Nomes das colunas do tipo float64:**')
        st.markdown(list(exploracao[exploracao['tipos'] == 'float64']['nomes']))
        st.markdown('**Nomes das colunas do tipo object:**')
        st.markdown(list(exploracao[exploracao['tipos'] == 'object']['nomes']))

        st.subheader('Questão 5')
        st.markdown('Qual porcentagem dos registros possui ao menos um valor null (`None`, `NaN`, `etc`)?')
        st.markdown("<h4>Para descobrirmos a porcentagem de nulos, realizaremos o seguinte cálculo:<br>\
            (tamanho total do df - valor de nulos do df) / tamanho total do df</h4>", unsafe_allow_html=True)
        st.write((len(black_friday) - len(black_friday.dropna())) / len(black_friday))
        st.subheader('Questão 6')
        st.markdown('Quantos valores null existem na variável (coluna) com o maior número de null?')
        st.markdown("<h4>Nessa questão, utilizaremos 3 funções bastantes simples, porém úteis. caso vc tente usar df.isnull(), você recebera um novo dataframe de booleanos, informando se o valor é ou não nulo.<br>\
            Caso você adicione a função sum(), ou seja, df.isnull().sum(). Você obterá um pd.Series com o valor de elemenos nulos de cada coluna. Mas pense bem, não queremos todos os valores nulos, queremos saber apenas quantos valores nulos existem dentro da coluna com maior numero de elementos nulos.<br>\
                Pense bem... o valor máximo. Agora sacou né? tente retornar -> max(df.isnull().sum())</h4>", unsafe_allow_html=True)
        valores = black_friday['Product_Category_3'].isna().sum()
        st.write(int(valores))
        st.dataframe(black_friday['Product_Category_3'])
        st.subheader('Questão 7')
        st.markdown('Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`?')
        st.markdown("<h4>Aqui poderíamos utilizar diversas maneiras de verificar o valor mais frequente, o método mais simples em minha opnião é utilizar o método mode() do pandas, o mesmo retorna um pd.Series com o valor da moda (valor que mais se repete) dentro desse df.<br>\
            Tente usar -> df['Product_Category_3'].mode()[0]. É necessário o índice 0 para especificar que é o primeiro valor da Serie retornada, mesmo que nesse caso a Serie tenha apenas um valor.</h4>", unsafe_allow_html=True)
        st.write(black_friday['Product_Category_3'].mode()[0])
        st.subheader('Questão 8')
        st.markdown('Qual a nova média da variável (coluna) `Purchase` após sua normalização?')
        st.markdown("<h4>Dentre as diversas formas de normalização, aqui utilizamos a técnica minmax!</h4>", unsafe_allow_html=True)
        normalizacao = ((black_friday['Purchase'] - black_friday['Purchase'].min()) / (
                    black_friday['Purchase'].max() - black_friday['Purchase'].min()))
        resposta = normalizacao.mean()
        st.write(float(resposta))
        st.dataframe(normalizacao)
        st.subheader('Questão 9')
        st.markdown('Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização?')
        st.markdown("<h4>Para a padronização utilizamos Z-Score e filtramos apenas valores entre -1 e 1!</h4>", unsafe_allow_html=True)
        padronizacao = ((black_friday['Purchase'] - black_friday['Purchase'].mean()) / black_friday['Purchase'].std())
        resposta = ((padronizacao >= -1) & (padronizacao <= 1)).sum()
        st.write(int(resposta))
        st.dataframe(padronizacao)
        st.subheader('Questão 10')
        st.markdown('Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`?')
        st.markdown("<h4>Para obter a resposta desta questão são necessárias apenas duas coisas, os valores nulos de Product_Category_2 e os valores nulos de Product_Category_3<br>\
            se compararmos um com o outro saberemos informar se a afirmação pe verdadeira ou falsa.<br>\
                Nesse caso a resposta é:  </h4>", unsafe_allow_html=True)
        categoria_2 = black_friday['Product_Category_2'].isna()
        categoria_3 = black_friday['Product_Category_3'].isna()
        iguais = (categoria_2 & categoria_3)
        resultado = bool((iguais == categoria_2).all())
        if resultado:
            st.write(resultado)
            st.write('Sim, Podemos afirmar que se uma observação é null em Product_Category_2 ela também é em Product_Category_3')


        st.markdown("Note que as respostas apresentadas não são exclusivas e nem as melhores para cada problema, existem n maneiras de se obter o mesmo resultado, aqui apenas mostramos alguns.<br>\
            note também que, contas `matemáticas e estatísticas`, por mais que simples, aparecem de maneira constante. O que quero dizer com isso? Para se tornar um bom cientista de dados você precisará conhecer o suficiente dessas áreas para conseguir resolver problemas do dia a dia.<br>Não sabote seu conhecimento, não pule degraus dessa longa escadaria que você vai subir, confie no seu potencial e no seu esforço que os resultados certamente virão!<br>\
                :D", unsafe_allow_html=True)


if __name__ == '__main__':
    main()
