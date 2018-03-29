import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

black_frame = pd.read_csv("train.csv")
black_frame.head(5)
black_frame.describe()

# checking the missing value
black_frame.apply(lambda x: sum(x.isnull()),axis= 0)


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

le= LabelEncoder()
enc = OneHotEncoder(sparse=False)


columns = ['Gender','City_Category']
for col in columns:
    black_frame[col] = le.fit_transform(black_frame[col])
    

black_friday = black_frame.copy()
col = "City_Category"
 # fit one hot encoding
temp = enc.fit_transform(black_frame[[col]])
# changing the encoded features into dataframe
temp = pd.DataFrame(temp,columns= [(col+"_"+str(i)) for i in black_frame[col].value_counts().index])
# setting index value similar to original data frames to join
temp=temp.set_index(black_frame.index.values)
# adding the dataframe to original dataframe
black_friday= pd.concat([black_friday,temp],axis=1)


stay = pd.Series(black_friday.Stay_In_Current_City_Years)

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
black_friday = black_friday.merge(stay.to_frame(),right_index=True,left_index= True)

del(black_friday['Stay_In_Current_City_Years_y'])
black_friday.columns.values[5] = "Stay_In_Current_City_Years"


age= pd.Series(black_friday.Age)
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
 
del(black_friday['Age'])
# converting age variable to numeric       
Age = pd.to_numeric(age)

# converting series to dataframe and merging to original dataframe
black_friday=black_friday.merge(Age.to_frame(),right_index=True,left_index=True)

# Detecting outlier
from scipy.stats import mode
x = np.percentile(black_friday['Product_Category_1'],[25,75])
lower_bound = x[0]-(1.5*(x[1]-x[0]))
upper_bound = x[1]+(1.5*(x[1]-x[0]))
f=mode(black_friday['Product_Category_1']).mode[0]

prc1= pd.Series(black_friday.Product_Category_1)

for i in range(0,len(prc1)):
    if(prc1[i]>upper_bound):
        prc1[i]=f

del(black_friday['Product_Category_1'])
Product_Category_1 = prc1

# filling missing value
black_friday=black_friday.merge(Product_Category_1.to_frame(),right_index=True,left_index=True)
black_friday['Product_Category_2'].fillna(0,inplace=True)
black_friday['Product_Category_3'].fillna(0,inplace=True)

# Creating new features
table = black_friday.pivot_table(values='Purchase',index = ['Product_ID'],aggfunc= np.mean)
Product_mean = table['Purchase']


black_friday = black_friday.set_index('Product_ID')
black_friday = black_friday.join(table)


table2 = black_friday.pivot_table(values = 'Purchase',index=['User_ID'],aggfunc =np.mean)
user_id = black_friday['User_ID']

black_friday = black_friday.set_index('User_ID')
black_friday = black_friday.join(table2)

user_id.to_csv("user_id.csv",index = False,header = True)
user_id = pd.read_csv("user_id.csv")
black_friday = black_friday.merge(user_id,right_index = True,left_index=True)
target_variable = black_friday['Purchase']
del(black_friday['Purchase'])
