### Módulo de Aplicação de Regressão Linear

## 1 Descrição do projeto
Este módulo contém uma classe chamada `AplicandoRegressao`, que é projetada para realizar análises de regressão linear em conjuntos de dados. A classe utiliza a biblioteca `scikit-learn` para implementar a regressão linear e o modelo Lasso, que é responsável por selecionar as variáveis explicativas que, em conjunto, conseguem melhor estimar o comportamento da variável de resposta.

É necessário que o dataframe esteja totalmente tratado para ser manipulado pelo módulo.

## 2 Ferramentas e técnicas utilizadas
- VS Code
- Python 3.10.9

## 3 Objetivos do autor
- Aplicar conceitos aprendidos na Pós Graduação da Conquer em Business Intelligence e Analytics;
- Explorar o uso de classes e métodos para organização e fluxo do código;
- Explorar o contato com Machinne Leraning, aplicando e exercitando conceitos.

#### Funcionalidades Principais:

1. **Iniciação da Classe:**
   - Método: `__init__(self, df: DataFrame, y: str, tamanho_teste: float = 0.1, random_state: int = 42)`

   - Descrição: Este método inicializa a classe `AplicandoRegressao`. Recebe um DataFrame (`df`) contendo os dados, o nome da coluna alvo (`y`), além de parâmetros opcionais para o tamanho do conjunto de teste (`tamanho_teste`) e a semente para a geração de números aleatórios (`random_state`).

   - Exemplo:
     ```python
     from aplicando_regressao import AplicandoRegressao
     reg = AplicandoRegressao(df, y='price')
     ```

2. **Seleção de Colunas Relevantes:**
   - Método: `_melhores_colunas(self) -> list`

   - Descrição: Este método utiliza o modelo Lasso para identificar as colunas mais relevantes para a regressão linear. As colunas são selecionadas com base nos coeficientes não nulos do modelo Lasso.

   - Exemplo:
     ```python
     melhores_colunas = reg._melhores_colunas()
     ```

3. **Treino e Teste do Modelo:**
   - Método: `_treino_teste(self) -> tuple`

   - Descrição: Este método divide o conjunto de dados em conjuntos de treino e teste, ajusta um modelo de regressão linear aos dados de treino e retorna os rótulos reais (`y_test`) e as previsões (`y_pred`).

   - Exemplo:
     ```python
     y_test, y_pred = reg._treino_teste()
     ```

4. **Resultados da Regressão:**
   - Método: `resultados_regressao(self) -> dict`

   - Descrição: Este método calcula diversas métricas de regressão, incluindo Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), Mean Squared Error (MSE) e R-squared. Retorna um dicionário contendo esses resultados.

   - Exemplo:
     ```python
     resultados = reg.resultados_regressao()
     ```

5. **Exemplo de Uso Completo:**
   - Descrição: Exemplo de uso completo da classe, carregando dados de um arquivo CSV, removendo colunas desnecessárias e aplicando a análise de regressão.

   - Exemplo:
     ```python
     from pandas import read_csv

     arquivo = r'C:\caminho\para\seu\arquivo.csv'
     df = read_csv(arquivo, sep=';', decimal=',')

     # Remova colunas desnecessárias
     df = df.drop(['coluna1', 'coluna2'], axis=1)

     # Crie uma instância da classe e execute a análise de regressão
     reg = AplicandoRegressao(df, y='coluna_alvo')
     resultados = reg.resultados_regressao()

     print(resultados)
     ```

### Dicas de Utilização:

1. **Escolha do Conjunto de Teste:**
   - É importante ajustar o parâmetro `tamanho_teste` na inicialização da classe de acordo com o tamanho desejado do conjunto de teste. Isso afeta a divisão entre os conjuntos de treino e teste.

2. **Interpretação dos Resultados:**
   - Após executar `resultados_regressao()`, examine as métricas retornadas para avaliar o desempenho do modelo. Valores mais baixos para MAE, MAPE e MSE indicam um bom resultado do modelo.

3. **Adaptação para Seu Conjunto de Dados:**
   - Antes de aplicar a classe ao seu conjunto de dados, revise as colunas removidas e ajuste os parâmetros conforme necessário para atender às características específicas do seu conjunto de dados.

### Considerações Finais:
Este módulo fornece uma estrutura reutilizável para análise de regressão linear em conjuntos de dados. A documentação visa ajudar tanto iniciantes quanto usuários experientes a entender e aplicar a classe de maneira eficaz.
