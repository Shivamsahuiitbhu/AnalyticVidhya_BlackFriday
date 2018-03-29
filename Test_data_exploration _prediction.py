# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 14:47:51 2018

@author: Lenovo
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

black_frame_test = pd.read_csv("test.csv")

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

le= LabelEncoder()
enc = OneHotEncoder(sparse=False)


columns = ['Gender','City_Category']
for col in columns:
    black_frame_test[col] = le.fit_transform(black_frame_test[col])
    

black_friday_test = black_frame_test.copy()
col = "City_Category"
 # fit one hot encoding
temp = enc.fit_transform(black_frame_test[[col]])
# changing the encoded features into dataframe
temp = pd.DataFrame(temp,columns= [(col+"_"+str(i)) for i in black_frame_test[col].value_counts().index])
# setting index value similar to original data frames to join
temp=temp.set_index(black_frame_test.index.values)
# adding the dataframe to original dataframe
black_friday_test= pd.concat([black_friday_test,temp],axis=1)

stay = pd.Series(black_friday_test.Stay_In_Current_City_Years)

for i in range(0,len(stay)):
    if(stay[i]=='0'):
        stay[i]= 1
    elif(stay[i]=='1'):
        stay[i]=2
    elif(stay[i]=='2'):
        stay[i]=3
    elif(stay[i]=='3'):
        stay[i]=4
    elif(stay[i]=='4+'):
        stay[i]= 8
        
stay = pd.to_numeric(stay)
black_friday_test = black_friday_test.merge(stay.to_frame(),right_index=True,left_index= True)

del(black_friday_test['Stay_In_Current_City_Years_y'])
black_friday_test.columns.values[5] = "Stay_In_Current_City_Years"

age= pd.Series(black_friday_test.Age)
age.head(6)

for i in range(0,len(age)):
    if(age[i]=='0-17'):
        age[i]='15'
    elif(age[i]=='18-25'):
        age[i]='21'
    elif(age[i]=='26-35'):
        age[i]='30'
    elif(age[i]=='36-45'):
        age[i]='40'
    elif(age[i]=='46-50'):
        age[i]='48'
    elif(age[i]=='51-55'):
        age[i]='53'
    elif(age[i]=='55+'):
        age[i]='60'
 
del(black_friday_test['Age'])
# converting age variable to numeric       
Age = pd.to_numeric(age)

# converting series to dataframe and merging to original dataframe
black_friday_test=black_friday_test.merge(Age.to_frame(),right_index=True,left_index=True)

black_friday_test.apply(lambda x: sum(x.isnull()),axis= 0)

black_friday_test['Product_Category_2'].fillna(0,inplace=True)
black_friday_test['Product_Category_3'].fillna(0,inplace=True)

table_1 = black_friday.pivot_table(values='Purchase',index = ['Product_ID'],aggfunc= np.mean)
table_1.columns.values[0] = "Product_mean"

#black_friday_test = black_friday_test.set_index('Product_ID')
black_friday_test = black_friday_test.join(table_1,on= 'Product_ID')
table2 = black_friday.pivot_table(values = 'Purchase',index=['User_ID'],aggfunc =np.mean)
table2.columns.values[0] = "User_mean"

black_friday_test = black_friday_test.join(table2,on= 'User_ID')

dftest = black_friday_test.copy()
del(dftest['Product_ID'])

mean_use_2 = dftest['User_mean']**2
prod_user = dftest['Product_mean'] * dftest['User_mean']
dftest = dftest.merge(mean_use_2.to_frame(),right_index=True,left_index=True)
dftest = dftest.merge(prod_user.to_frame(),right_index=True,left_index=True)

dftest['Product_mean'].fillna(np.mean(dftest['Product_mean']),inplace=True)
dftest['0'].fillna(np.mean(dftest['0']),inplace=True)


Purchase=  model.predict(dftest)
Purchase = pd.DataFrame(Purchase)
pieces = [dftest['User_ID'],black_friday_test['Product_ID'],Purchase]
Purchase_1 = pd.concat(pieces,axis = 1)
Purchase_1.columns.values[2] = 'Purchase'
Purchase_1.to_csv("summit_1.csv",header = True,encoding = 'utf-8',index=False)











