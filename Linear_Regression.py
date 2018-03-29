# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 20:15:28 2018

@author: Lenovo
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 

import statsmodels.api as sm
from statsmodels.graphics import regressionplots
import seaborn as sns

# splitting data into train and test 
X_train,X_test,y_train,y_test = train_test_split(black_friday,target_variable,test_size=0.35)

df_train = pd.merge(X_train,y_train,right_index=True,left_index=True)
df_test = pd.merge(X_test,y_test,right_index=True,left_index=True)

# added polynomial and interaction term
mean_use_2 = df_train['User_mean']**2
prod_user = df_train['Product_mean'] * df_train['User_mean']

# fitting linear regession using formula 
model = smf.ols(formula = 'Purchase ~ Gender +Occupation +Stay_In_Current_City_Years+Marital_Status+Product_Category_2+ Product_Category_3 + Age+Product_Category_1+Product_mean+User_mean_x++User_mean_y',
                data =df_train)
model= model.fit()
print(model.summary())

# Evaluating model
pred_train = model.predict(df_train)
pred_test = model.predict(df_test)
MSE_train = np.mean((pred_train-y_train)**2)
MSE_test = np.mean((pred_test-y_test)**2)
RSE_train = abs(MSE_train)**0.5
RSE_test = abs(MSE_test)**0.5
print("RSE_train =",RSE_train)
print("RSE_test =",RSE_test)

# plotting residuals plots

residual_plot_var = pd.DataFrame({'resid':model.resid,'std_resids': model.resid_pearson,
                        'fitted': model.predict()})


# residual vs fitted value
residvsfitted = plt.plot(residual_plot_var['fitted'],residual_plot_var['resid'],'+')
l= plt.axhline(y=0,color='black',linestyle = 'dashed')
plt.xlabel('Fitted_values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted_value')
plt.show(residvsfitted)


# Q-Q Plot
qqplot= sm.qqplot(residual_plot_var['std_resids'],line = 's')
plt.show(qqplot)

# Scalelocation plot
scalelocplot = plt.plot(residual_plot_var['fitted'],abs(residual_plot_var['std_resids'])**0.5,
                        'o')
plt.xlabel('Fitted_values')
plt.ylabel('Square Root of |standardized residuals|')
plt.title('Scale-Location')
plt.show(scalelocplot)

# Residual vs leverage plot
from statsmodels.graphics import regressionplots
residsvlevplot = regressionplots.influence_plot(model, criterion = 'Cooks')
plt.show(residsvlevplot)


