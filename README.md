# Predicting Fuel Economy Using Simple Linear Regression
**Author:** Iaroslava Mizai  
>**Project created as a final project for the course:** Simple Linear Regression for the Absolute Beginner  
>**Instructor:** Ryan Ahmed 

## Project Overview
This project aims to develop a predictive model that analyzes the relationship between Vehicle Horse Power (HP) and Fuel Economy, measured in Miles Per Gallon (MPG). By leveraging data from a comprehensive Fuel Economy dataset, we seek to understand how variations in horsepower can impact fuel efficiency, which is crucial for automotive manufacturers aiming to optimize vehicle performance and environmental impact.

### Goals
+ To understand the relationship between vehicle horsepower and fuel economy.
+ To provide insights that can guide automotive manufacturers in optimizing vehicle design for better fuel efficiency.
+ To develop a reliable predictive model that can be used for future analysis and decision-making.
  
### The Questions
1) How does horsepower affect fuel economy?
2) Can we create a reliable predictive model for MPG based on HP?
   
### Tools I Used
+ **Python**: For data analysis and model development.
  + **Pandas**: For data manipulation and analysis.
  + **NumPy**: For numerical calculations.
  + **Matplotlib & Seaborn**: For data visualization.
  + **Scikit-Learn**: For implementing the linear regression model.
+ Jupyter Notebooks: The tool I used to run my Python scripts.
+ Visual Studio Code: My go-to for executing my Python scripts.
+ Git & GitHub: Essential for version control and sharing my Python code and analysis.


### Key Steps
1) **Exploratory Data Analysis**
2) **Create testing and train dataset** 
3) **Model Development**
4) **Test Model**

### Dataset
The dataset used for this analysis is the **FuelEconomy.csv**, which includes the following variables:
+ Independent Variable (X): Vehicle Horse Power (HP)
+ Dependent Variable (Y): Mileage Per Gallon (MPG)

## Ananysis

### 1) Exploratory Data Analysis
a) Data Loading
Import the "FuelEconomy.csv" dataset into the analysis environment.

```python
#!pip install --upgrade seaborn
#!pip install --upgrade pandas

#import libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```
```python
# Load dataset 
df = pd.read_csv(r"C:\Users\ymiza\Documents\my page\Python\Python_Data_Project\Simple_Linear_Regression_Project\FuelEconomy.csv")

df.head()
```
Insights:
The dataset consists of 100 entries and includes two critical variables: Horse Power and Fuel Economy (MPG), which are essential for our analysis.

b) Conduct data visualization and preliminary analysis to identify trends, patterns, and anomalies within the dataset.

```python
df.describe()
```
```python
df.info()
```
```python
# Visualize data 
sns.set_theme(style='darkgrid')
sns.jointplot(x='Horse Power', y='Fuel Economy (MPG)', data=df, kind='reg', truncate=False)
plt.show()
```
![](https://github.com/DataVizStory/Project_Fuel-Consumption_Simple-Linear-Regression/blob/main/Images/Chart1.png)

```python
sns.set_theme(style="ticks")
sns.pairplot(df, kind='reg')
plt.show()
```
![](https://github.com/DataVizStory/Project_Fuel-Consumption_Simple-Linear-Regression/blob/main/Images/Chart2.png)
```python
sns.set_theme(style="ticks")
sns.lmplot(x='Horse Power', y='Fuel Economy (MPG)', data=df)
plt.ylabel('MPG')
plt.grid()
plt.show()
```
![](https://github.com/DataVizStory/Project_Fuel-Consumption_Simple-Linear-Regression/blob/main/Images/Chart3.png)

Insights:
Preliminary analysis indicates a negative correlation between horsepower and fuel economy, suggesting that higher horsepower generally leads to lower MPG.

### 2) Create Testing and Training Dataset
Split the dataset into training (75%) and testing (25%) subsets to facilitate model training and evaluation.

```python
X = df['Horse Power']  # Input to the Model
y = df['Fuel Economy (MPG)']  # Output to the Model
```
```python
# Convert X and y into arrays
X = np.array(X).reshape(-1, 1)
y = np.array(y).reshape(-1, 1)
```
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)  
```
Insights:
The data split ensures that we have a robust training and testing setup, allowing us to assess the model's performance accurately.

### 3) Model Development
Utilize Scikit-Learn to implement a simple linear regression model that predicts fuel economy based on horsepower.

```python
from sklearn.linear_model import LinearRegression
```
```python
# Training model 
SimpleLinearRegression = LinearRegression(fit_intercept=True)
SimpleLinearRegression.fit(X_train, y_train)
```

```python
print('Linear Model Coefficient (m):', SimpleLinearRegression.coef_)
print('Linear Model Coefficient (b):', SimpleLinearRegression.intercept_)
```
Insights:
The coefficients from the linear regression model quantify the relationship between horsepower and fuel economy, indicating how much MPG decreases per unit increase in horsepower.


### 4) Test Model
Assess the performance of the trained model using relevant metrics to ensure its accuracy and reliability.

```python
y_predict = SimpleLinearRegression.predict(X_test)
y_predict
```
```python
y_test
```

```python
# Visualize Train Set Results
plt.scatter(X_train, y_train, color='gray')
plt.plot(X_train, SimpleLinearRegression.predict(X_train), color='red')
plt.ylabel('MPG')
plt.xlabel('Horse Power (HP)')
plt.title('HP vs. MPG (Training dataset)')
plt.show()
```
![](https://github.com/DataVizStory/Project_Fuel-Consumption_Simple-Linear-Regression/blob/main/Images/HP%20vs.%20MPG%20(Training%20dataset).png)

```python
# Visualize Test Set Results
plt.scatter(X_test, y_test, color='gray')
plt.plot(X_test, SimpleLinearRegression.predict(X_test), color='red')
plt.ylabel('MPG')
plt.xlabel('Horse Power (HP)')
plt.title('HP vs. MPG (Testing dataset)')
plt.show()
```
![](https://github.com/DataVizStory/Project_Fuel-Consumption_Simple-Linear-Regression/blob/main/Images/HP%20vs.%20MPG%20(Testing%20dataset).png)

### 5) Model Evaluation and Accuracy Metrics
1) ##### R²-Score:
1R²-Score: The R²-Score, also known as the coefficient of determination, tells us how well our model fits the data. It gives us the proportion of the variance in the dependent variable (MPG) that can be predicted from the independent variable (horsepower). A higher R² indicates a better model fit.
```python
# First way to calculate the coefficient of determination (R²)
# A built-in method of the model that returns the coefficient of determination (R²) based on the test data
accuracy_LinearRegression = SimpleLinearRegression.score(X_test, y_test)
print('Model Accuracy:', accuracy_LinearRegression)
```
```python
# Second way to calculate the coefficient of determination (R²)
# Manually calculates R²  by comparing the true values and predicted values
from sklearn.metrics import r2_score
r2=r2_score(y_test,y_predict)
print('Model Accuracy:', r2)
```
Insights:
The model achieved an R² of approximately 93%, indicating that 93% of the variance in Fuel Economy (MPG) can be explained by Horse Power (HP). This suggests the model is quite effective in predicting MPG based on HP.

2) ##### Mean Absolute Error (MAE)
Mean Absolute Error (MAE): The Mean Absolute Error (MAE) is the average of the absolute differences between the predicted and actual values. It is a common metric used to evaluate the accuracy of regression models, where smaller values indicate better performance.
```python
#This metric quantifies the average magnitude of errors. Lower MAE suggests a better model.
mae = np.mean(np.absolute(y_predict - y_test))
print("Mean absolute error: %.2f" % mae)
```
Insights: The MAE for this model is 0.98. This suggests that, on average, the model's prediction is off by 0.98 MPG. This is a relatively small error, meaning the model's predictions are fairly accurate and close to the true values.

3) ##### Root Mean Squared Error (RMSE)
Root Mean Squared Error (RMSE): Root Mean Squared Error (RMSE) is another commonly used metric that can be derived by taking the square root of MSE. This helps bring the error value back to the original units (MPG) and is often easier to interpret.
```python
mse = np.mean((y_predict - y_test) ** 2)
print("Residual sum of squares (MSE): %.2f" % mse)
rmse = np.sqrt(mse)
print("Root Mean Squared Error (RMSE): %.2f" % rmse) 
```
Insights: Given that RMSE (1.22) is relatively small compared to the overall range of MPG values (which typically range from 10 to 50 MPG in this dataset), it indicates that the model is reasonably accurate.
The model does not make large errors in its predictions, and an average error of 1.22 MPG suggests that the model is performing well.

### 6) Generate the predictions
```python
# Use the trained Model to generate the predictions
HP = np.array([240]).reshape(-1, 1)
MPG = SimpleLinearRegression.predict(HP)
print('Predicted MPG for 240 HP:', MPG)
```
Insights:
The prediction for a vehicle with 240 HP is approximately 21.4 MPG, demonstrating the practical application of the model.

## What I Learned
The project has demonstrated the utility of simple linear regression in predicting fuel economy (MPG) based on vehicle horsepower (HP). The model is able to produce relatively accurate predictions with a 93% accuracy, 0.98 mean absolute error, and a MSE of 1.48, which suggests good model fit and reliability.

## Overall Insights
1. The relationship between horsepower and fuel economy is inversely proportional: as horsepower increases, fuel economy decreases. This is reflected in the negative correlation observed during exploratory data analysis.
2. The model is fairly accurate in predicting MPG based on HP. With an R² score of 93%, it shows strong predictive power, and the mean absolute error of 0.98 suggests that the predictions are, on average, close to the actual values.
3. The MSE of 1.48 and RMSE value 1.22 indicate that the model performs reasonably well and can be useful for decision-making in automotive design, specifically in predicting fuel efficiency.

## Challenges I Faced
Ensuring the dataset was clean and free of anomalies.
Balancing model complexity and interpretability while achieving good predictive performance.

## Conclusions
This project successfully demonstrated the ability to predict fuel economy using a simple linear regression model, with vehicle horsepower as the predictor. The insights gained from the model can aid automotive manufacturers in making data-driven decisions to improve vehicle performance and fuel efficiency. The model's good accuracy, paired with relatively low error metrics, shows that the relationship between horsepower and fuel economy is both strong and predictable, making it valuable for practical applications in the automotive industry.
With this model, we have established a reliable method to estimate fuel economy based on a vehicle's horsepower, which is crucial for manufacturers aiming to optimize both performance and environmental impact.

