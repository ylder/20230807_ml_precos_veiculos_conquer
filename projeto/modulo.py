# python version 3.10.9

from pandas import DataFrame, read_csv
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score


class AplicandoRegressao:
    def __init__(self, df: DataFrame, y: str, tamanho_teste: float = 0.1, random_state: int = 42):
        """
        Inicializa a classe de aplicação de regressão linear.

        Parâmetros:
        - df (DataFrame): DataFrame contendo os dados.
        - y (str): Nome da coluna alvo.
        - tamanho_teste (float): Proporção do conjunto de teste (padrão é 0.1).
        - random_state (int): Seed para reprodução dos resultados (padrão é 42).
        """
        # Armazena o DataFrame e inicializa o modelo Lasso.
        self._df = df
        self._lasso_model = LassoCV(cv=5)
        # Define a variável de resposta y.
        self.y = self._df[y]
        # Seleciona as melhores colunas usando o método privado _melhores_colunas().
        self.melhores_colunas = self._melhores_colunas()
        # Cria um subconjunto de dados apenas com as melhores colunas.
        self.X = self._df[self.melhores_colunas]
        # Configurações para a divisão do conjunto de treino e teste.
        self._tamanho_teste = tamanho_teste
        self._random_state = random_state
        # Variáveis para armazenar os resultados do modelo.
        self.intercessao = None
        self.coeficientes = None

    def _melhores_colunas(self) -> list:
        """
        Seleciona as melhores colunas usando o modelo Lasso.

        Retorna:
        - list: Lista de nomes das colunas selecionadas.
        """
        # Remove a coluna de resposta temporariamente.
        X_temp = self._df.drop(columns=self.y.name, axis=1)
        # Ajusta o modelo Lasso aos dados.
        self._lasso_model.fit(X_temp, self.y)
        # Obtém as colunas selecionadas pelo modelo (coeficientes não nulos).
        colunas_selecionadas = X_temp.columns[self._lasso_model.coef_ != 0]
        return colunas_selecionadas.tolist()

    def _treino_teste(self) -> tuple:
        """
        Divide os dados em conjuntos de treino e teste, ajusta o modelo e retorna as previsões.

        Retorna:
        - tuple: Tupla contendo arrays y_test e y_pred.
        """
        # Divide os dados em conjuntos de treino e teste.
        X_train, X_test, y_train, y_test = train_test_split(
            self.X,
            self.y,
            test_size=self._tamanho_teste,
            random_state=self._random_state
        )

        # Inicializa e ajusta o modelo de regressão linear.
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Armazena a intercessão e os coeficientes do modelo.
        self.intercessao = model.intercept_
        self.coeficientes = dict(zip(self.X.columns, model.coef_))

        # Realiza previsões no conjunto de teste.
        y_pred = model.predict(X_test)

        return y_test, y_pred

    def resultados_regressao(self) -> dict:
        """
        Calcula métricas de regressão e retorna os resultados.

        Retorna:
        - dict: Dicionário contendo métricas de regressão.
        """
        # Obtém as previsões e os rótulos reais do conjunto de teste.
        y_test, y_pred = self._treino_teste()

        # Calcula diversas métricas de regressão.
        mae = mean_absolute_error(y_test, y_pred)
        mae_perc = mean_absolute_percentage_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Armazena as métricas em um dicionário.
        return {
            "Mean Absolute Error": mae,
            "Mean Absolute Percentage Error": mae_perc,
            "Mean Squared Error": mse,
            "R-squared": r2
        }


if __name__ == '__main__':
    # Lê o arquivo CSV e remove colunas não necessárias.
    df = read_csv('projeto/veiculos.csv', sep=';', decimal=',')
    df = df.drop(['carname', 'idCar'], axis=1)

    # Instancia a classe AplicandoRegressao com o DataFrame e a coluna alvo.
    reg = AplicandoRegressao(df, y='price')

    # Imprime os resultados da regressão.
    reg.resultados_regressao()