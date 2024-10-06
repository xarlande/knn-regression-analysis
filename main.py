import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Крок 1: Згенерувати випадковий набір даних
np.random.seed(0)  # Для відтворюваності результатів
X = np.random.uniform(0, 1000, size=(1000, 1))  # 1000 значень в діапазоні від 0 до 1000
# Створимо залежну змінну з деякою функціональною залежністю і шумом
Y = np.sin(X / 100) + np.random.normal(0, 0.1, size=(1000, 1))

# Крок 2: Нормалізувати значення
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_Y = StandardScaler()
Y_scaled = scaler_Y.fit_transform(Y)

# Крок 3: Розділити на навчальну і тестову вибірки
X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y_scaled, test_size=0.2, random_state=42
)

# Крок 4: Навчити KNN-регресор з різними значеннями K
k_values = range(1, 21)
mse_train = []
mse_test = []

for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, Y_train.ravel())
    Y_train_pred = knn.predict(X_train)
    Y_test_pred = knn.predict(X_test)
    mse_train.append(mean_squared_error(Y_train, Y_train_pred))
    mse_test.append(mean_squared_error(Y_test, Y_test_pred))

# Крок 5: Вибрати найкраще значення K
best_k = k_values[np.argmin(mse_test)]
print(f"Найкраще значення K: {best_k}")

# Крок 6: Візуалізація результатів
# Графік залежності MSE від K
plt.figure(figsize=(10, 5))
plt.plot(k_values, mse_train, label='Навчальна вибірка')
plt.plot(k_values, mse_test, label='Тестова вибірка')
plt.xlabel('K - кількість сусідів')
plt.ylabel('Середньоквадратична помилка (MSE)')
plt.title('Залежність MSE від K')
plt.legend()
plt.show()

# Візуалізація регресії з найкращим K
knn_best = KNeighborsRegressor(n_neighbors=best_k)
knn_best.fit(X_train, Y_train.ravel())
Y_pred = knn_best.predict(X_scaled)

plt.figure(figsize=(10, 5))
plt.scatter(X_scaled, Y_scaled, color='blue', label='Дані')
plt.scatter(X_scaled, Y_pred, color='red', label='KNN Регресія')
plt.xlabel('X (нормалізований)')
plt.ylabel('Y (нормалізований)')
plt.title(f'KNN Регресія з K={best_k}')
plt.legend()
plt.show()
