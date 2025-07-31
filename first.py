
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


data = pd.DataFrame({
    'SquareFootage': [1400, 1600, 1700, 1875, 1100, 1550, 2350, 2450, 1425, 1700],
    'Bedrooms': [3, 3, 3, 4, 2, 3, 4, 4, 3, 3],
    'Bathrooms': [1, 2, 2, 2, 1, 2, 3, 3, 2, 2],
    'Price': [245000, 312000, 279000, 308000, 199000, 219000, 405000, 324000, 319000, 255000]
})

print("\nFirst 5 rows of dataset:")
print(data.head())


sns.pairplot(data, x_vars=['SquareFootage', 'Bedrooms', 'Bathrooms'], y_vars='Price', height=4, aspect=1, kind='scatter')
plt.suptitle("Feature vs Price Relationship", y=1.02)
plt.show()


plt.figure(figsize=(6, 4))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

X = data[['SquareFootage', 'Bedrooms', 'Bathrooms']]
y = data['Price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("\nModel Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)
print("R² Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)  # reference line
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.show()

new_house = np.array([[2000, 3, 2]])  # 2000 sqft, 3 bedrooms, 2 bathrooms
predicted_price = model.predict(new_house)
print(f"\nPredicted Price for 2000 sqft, 3 bed, 2 bath: ${predicted_price[0]:,.2f}")
