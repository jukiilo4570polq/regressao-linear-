from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Dados fictícios
aparelhos = [1, 2, 3, 4, 5]
custos = [50, 100, 150, 200, 250]

# Treinando o modelo
X = [[x] for x in aparelhos]
y = custos
modelo = LinearRegression()
modelo.fit(X, y)

# Previsão e avaliação
y_pred = modelo.predict(X)
mse = mean_squared_error(y, y_pred)
print(f"Erro Quadrático Médio: {mse}")
