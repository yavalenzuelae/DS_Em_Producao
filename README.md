# Predição de Vendas de uma Rede Lojas 

Projeto de ciência de dados (projeto_vendas_lojas_v0.ipynb) que implementa um ciclo do *Cross Industry Standard Process* - CRISP de 9 passos para resolver um problema de negócio e entregar valor para uma empresa. São usados dados públicos para a demonstração do método (arquivo data). Os 9 passos considerados são: Questão de negócio, Entendimento do negócio, Coleta de dados, Limpeza dos dados, Exploração dos dados, Modelagem dos dados, Algoritmos de Machine Learning - ML, Avaliação do algoritmo e o Modelo em produção (handler.py).

## Questão de Negócio

O objetivo do projeto é predecir as vendas, nas próximas seis (06) semanas, de uma rede de farmácias. Se assume que o projeto surge do problema que encara uma empresa em preveer as vendas de toda uma rede e, assim, estimar quanto deve investir para reformar todas as suas lojas.

## Premisas do Negócio

Considerando que temos um conhecimento prévio dos atributos do dataset (ver pasta data):

1. Vamos supor que os valores faltantes na coluna competition_distance tem o significado de que temos competidores ainda mais longe do que a máxima distância presente na coluna. Logo, o preenchimento desses campos será feito com um valor muito maior que a distância máxima. 
2. Valores faltantes nas colunas competition_open_since_[month/year] serão substituídos pelo mês (ano) da coluna date. Este valor vai supor que o competidor abriu no mesmo dia em que foram feitas as vendas especificadas na coluna sales.
3. Valores faltantes nas colunas promo2_since_[week/year] significa que a loja não tem data da continuidade de uma promoção porque a loja não está participando. Porém, vamos supor que na mesma data em que foram feitas as vendas a loja estava numa promoção à qual decidiu dar continuidade.
4. Lojas com o maior (menor) sortimento tem assortment extended (basic).
5. As lojas que participam da promo2 começam no primeiro dia da semana (promo2_since_week).

## Planejamento da Solução

Para resolver a questão de negócio, acima:
 
1. Coletar os dados (https://www.kaggle.com/c/rossmann-store-sales).
2. Limpar os dados (*Data Cleaning*). Este passo aborda: descrição dos dados (*Data Description*), engenharia de features (*Feature Engineering*) e filtragem de variáveis (*Variable Filtering*).
3. Realizar um Análise exploratória dos dados (*Exploratory Data Analysis* - EDA). Este passo aborda: análise univariada, bivariada e multivariada da variável resposta.
4. Preparar os dados (*Data Preparation*) prévio à aplicação dos algoritmos de ML. Este passo envolve o rescaling, enconding e demais transformações nas variáveis contidas nos dados coletados. E, uma seleção das variáveis (*Feature Selection*) mais relevântes para o modelo.    
5. Modelar os dados com algoritmos de ML (*Machine Learning Modeling*). Neste passo aplicamos varios algoritmos de ML e determinamos a performance de cada com respeito ao valor de base (*baseline*). 
6. Realizar um ajuste fino dos parâmetros (*hiperparameter fine tuning*). Neste passo selecionamos o algoritmo de ML com a melhor performance encontrada no passo anterior e realizamos uma busca de um conjunto de parâmetros que melhora ainda mais a performance encontrada.
7. Relizar uma interpretação e tradução do erro. Neste passo realizamos uma tradução da performance do algoritmo de ML para a performance do negócio.

## Os 5 Principais Insights do Negócio

1. As lojas deveriam vender menos durante os feriados escolares. 
   - Falsa: As lojas vendem, na média, mais durante os feriados escolares.
2. As lojas deveriam vender mais depois do dia 10 de cada mês.
   - Falsa: As lojas vendem, na média, menos depois do dia 10.
3. As lojas abertas durante o feriado de natal deveriam vender mais. 
   - Falsa: As lojas abertas durante o feriado de natal vendem menos.
4. As lojas com mais promoções consecutivas deveriam vender mais. 
   - Falsa: As lojas com mais promoções consecutivas vendem menos.
5. As lojas com competidores mais próximos deveriam vender menos. 
   - Falsa: As lojas com competidores mais próximos vendem, na média, o mesmo que as mais distantes.

## Resultados Financeiros para o Negócio

O modelo prevee que, nas próximas seis semanas, a rede de lojas vai realizar, na média, 7730880 vendas. No pior cenário, a rede vai realizar 6851054 vendas e, no melhor 8610705 (ver subseção 9.1.1. do notebook). A predição para uma loja específica da rede pode ser obtida usando um bot no Telegram (... por aqui link do contato ...) criado para este projeto (pasta rossmann-telegram-api). Na média, o erro percentual médio absoluto que, teria a predição das vendas de uma das lojas, quaisquer, da rede, é de 12% (ver subseção 9.1. do notebook).

## Performance dos Modelos de Machine Learning

Os dados foram modelados usando os algoritmos mostrados na tabela abaixo. A tabela mostra os erros das predições estimadas: *Mean Absolute Error* - MAE, *Mean Absolute Percentage Error* - MAPE e *Root Mean Square Error* - RMSE, com *Cross-Validation* (CV). Os resultados foram obtidos dividindo os dados de treino em cinco regiões (Kfold = 5).

| Modelo |	MAE_CV |	MAPE_CV |	RMSE_CV |
| --- | --- | --- | --- |
| Random Forest Regressor |	845,16 $\pm$ 220,89 |	11,66 $\pm$ 2,38 |	1262,86 $\pm$ 323,88 |
| Regressão Linear	| 2083,01 $\pm$ 305,27 |	29,85 $\pm$ 1,45	| 2965,89 $\pm$ 473,78 |
| Regressão Linear Regularizada - Lasso	| 2121,8 $\pm$ 345,83	| 29,13 $\pm$ 1,2	| 3064,48 $\pm$ 507,7 |
| XGBoost Regressor	| 7064.74 $\pm$ 594.59 |	95.11 $\pm$ 0,2	| 7727.09 $\pm$ 695.08 |

Na busca de um conjunto de parâmetros para melhorar os resultados do modelo com a melhor performance, o Random Forest Regressor (ver tabela acima), foi realizado um ajuste fino com o método *Random Search*. Se encontrou (primero ciclo) que os parâmetros por padrão já geram o melhor resultado. Porém, quando se salva o modelo treinado o tamanho é de 6,2 GB, inviabilizando a criação de um modelo em produção num servidor externo de baixa capacidade. O problema foi resolvido (segundo ciclo) com um ajuste fino do XGBoost Regressor, sem perda de performance (ver código: predicao_vendas_lojas_v0.ipynb). Neste caso o tamanho salvo do modelo treinado foi de 12 MB. Na tabela abaixo se mostra o resultado de CV usando um total de 05 combinações de parâmetros (MAX_EVAL = 5) com o *Random Search* (ver subseção 8.1 do notebook).

| Modelo | MAE_CV | MAPE_CV | RMSE_CV |
| --- | --- | --- | --- |
XGBoost Regressor | 937.48 $\pm$ 121.16 | 13.12 $\pm$ 1.07 | 1340.12 $\pm$ 170.82 |
 
O modelo final foi avaliado com os dados de teste (ver subseção 8.2 do notebook). Os resultados se mostram a seguir:

| Modelo | MAE | MAPE | RMSE |
| --- | --- | --- | --- |
XGBoost Regressor | 785.97 | 11,93 | 1119.86 |

Com o modelo definido e ajustado foi feita uma interpretação e tradução do erro (módulo 9.0. no notebook). Num gráfico das vendas e das predições ao longo do tempo (ver na subseção 9.2. do notebook) pode-se observar que as predições seguem a tendência das vendas observadas nas próximas seis semanas. 

Se fizermos uma predição das vendas para uma data específica (dentro das próximas seis semanas) teriamos uma taxa de erro (= predictions / sales), na média, de 1,01 (ver subseção 9.2.). Logo, a predição se encontrara por cima da observação num 1 %.

O modelo descreve satisfatóriamente o fenômeno em questão, predição das vendas da rede de lojas nas próximas seis semanas. Como demonstrado pela distribuição do erro (= sales - predictions), a qual é próxima de uma normal (ver subseção 9.2. do notebook), e pelo gráfico do error vs predictions (ver subseção 9.2. do notebook), no qual, a grande maioria dos erros, estão confinados entre dois valores, -10000 e +10000. 

## Conclusão

Se prevee que, nas próximas seis semanas, a rede de lojas vai realizar, no pior cenário, 6851054 vendas e, no melhor 8610705. Para atingir o resultado mencionado, foi implementado o modelo de ML, XGBoost Regressor, entre 5 modelos aplicados (ver seção 7.0. no notebook). O MAPE_CV é de 13.12 $\pm$ 1.07 (ver tabela acima). Quando aplicado o modelo nos dados de teste o MAPE é de 11,93 % (ver subseção 8.2. do notebook). Resultado satisfatório se observamos que está abaixo do esperado pelo CV. A performance do modelo é ~88 %.      

## Perspectivas

Assumindo que o time de negócio quer melhorar ainda mais a performance de 88%. Sugeriria, num terceiro ciclo, implementar um novo ajuste fino dos parâmetros do XGBoost Regressor, sem comprometer o tamanho do modelo final treinado, e/ou re-avaliar as features selecionadas.
