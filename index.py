import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor


endPointRoot = 'https://github.com/ageron/data/raw/main/'
lifeStat = pd.read_csv(endPointRoot + 'lifesat/lifesat.csv')
X = lifeStat[['GDP per capita (USD)']].values
Y = lifeStat[['Life satisfaction']].values


lifeStat.plot(kind='scatter', grid=True, x='GDP per capita (USD)', y='Life satisfaction',)

plt.axis([23_500, 62_500, 4, 9])
plt.show()

model = KNeighborsRegressor(n_neighbors=3)

model.fit(X, Y)

X_new = [[37_000]]

print(model.predict(X_new))