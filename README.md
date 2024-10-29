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
Split the dataset into training (80%) and testing (20%) subsets to facilitate model training and evaluation.

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
accuracy_LinearRegression = SimpleLinearRegression.score(X_test, y_test)
print('Model Accuracy:', accuracy_LinearRegression)
```
Insights:
The model achieved an accuracy of approximately 92%, suggesting it is effective in predicting fuel economy based on horsepower.

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
```python
# Use the trained Model to generate the predictions
HP = np.array([240]).reshape(-1, 1)
MPG = SimpleLinearRegression.predict(HP)
print('Predicted MPG for 240 HP:', MPG)
```
Insights:
The prediction for a vehicle with 240 HP is approximately 21.4 MPG, demonstrating the practical application of the model.

## What I Learned
The relationship between vehicle horsepower and fuel economy is quantifiable.
Simple linear regression can effectively model this relationship and provide useful predictions.

## Overall Insights
Higher horsepower generally correlates with lower fuel economy, which is significant for automotive design.
The model's high accuracy indicates that horsepower is a crucial factor in predicting MPG.

## Challenges I Faced
Ensuring the dataset was clean and free of anomalies.
Balancing model complexity and interpretability while achieving good predictive performance.

## Conclusion
This project successfully demonstrates the ability to predict fuel economy using vehicle horsepower as a predictor variable. The insights gained can assist automotive manufacturers in making data-driven decisions to enhance vehicle efficiency.

