#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
df = pd.read_csv('strokes.csv', delimiter = ';', encoding = 'utf-8')
df_og=df


import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
import sklearn.metrics as sm
from sklearn.preprocessing import OrdinalEncoder
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('strokes.csv', delimiter = ';', encoding = 'utf-8')

df_NoNull=df.copy()
df_NoNull=df_NoNull.dropna()

#encode categorical values
encoder = OrdinalEncoder()

cols=['smoking_status']
def encode(data):
    '''function to encode non-null data and replace it in the original data'''
    #retains only non-null values
    nonulls = np.array(data.dropna())
    #reshapes the data for encoding
    impute_reshape = nonulls.reshape(-1,1)
    #encode date
    impute_ordinal = encoder.fit_transform(impute_reshape)
    #Assign back encoded values to non-null values
    data.loc[data.notnull()] = np.squeeze(impute_ordinal)
    return data

for i in cols:
    encode(df_NoNull[i])
    encode(df[i])


# adding missing values to bmi column
df_NoNull.loc[df_NoNull.sample(frac=0.033).index, 'bmi'] = np.nan

# adding missing values to smoking status column
df_NoNull.loc[df_NoNull.sample(frac=0.3).index, 'smoking_status'] = np.nan

#evaluation
df_recover=df_NoNull.copy()
print('--> Imputation Methods for BMI numerical Feature. \n')

print('Imputation method 1 : Mean imputation for BMI values.')
#mean imputation method
values = df_recover[['bmi']]
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
transformed_values = imputer.fit_transform(values)

#arranging true and prediction values
mean_res=pd.DataFrame()
mean_res['True Value'] =df.dropna()['bmi'].tolist()
mean_res['Missing'] =df_recover['bmi'].tolist()
mean_res['Prediction'] =transformed_values.flatten().tolist()
mean_res = mean_res[mean_res['Missing'].isna()]
mean_res[['True Value', 'Prediction']]

#performance
mean_mse=sm.mean_squared_error(mean_res['True Value'], mean_res['Prediction'])
print("Mean squared error =", round(mean_mse, 2) )
print('\n')


#median imputation
print('Imputation method 2: Median imputation for BMI values.')
#bmi 
values = df_recover[['bmi']]
imputer2 = SimpleImputer(missing_values=np.nan, strategy='median')
transformed_values2 = imputer2.fit_transform(values)

median_res=pd.DataFrame()
median_res['True Value'] =df.dropna()['bmi'].tolist()
median_res['Missing'] =df_recover['bmi'].tolist()
median_res['Prediction'] =transformed_values2.flatten().tolist()
median_res= median_res[median_res['Missing'].isna()]
median_res[['True Value', 'Prediction']]

bmi_median_mse = sm.mean_squared_error(median_res['True Value'], median_res['Prediction'])
print("Mean squared error =", round(bmi_median_mse, 2)) 
print('\n')


#Mode imputation
#BMI

print('Imputation method 3: Mode imputation for BMI values.')
values = df_recover[['bmi']]
imputer3 = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
transformed_values3 = imputer3.fit_transform(values)

mode_res=pd.DataFrame()
mode_res['True Value'] =df.dropna()['bmi'].tolist()
mode_res['Missing'] =df_recover['bmi'].tolist()
mode_res['Prediction'] =transformed_values3.flatten().tolist()
mode_res = mode_res[mode_res['Missing'].isna()]
mode_res[['True Value', 'Prediction']]

bmi_mode_mse=sm.mean_squared_error(mode_res['True Value'], mode_res['Prediction'])

print("Mean squared error =", round(bmi_mode_mse, 2)) 
print("\n")

#KNN imputation 

print('Imputation method 4: KNN imputation for BMI values.')

df_knn= df_recover.copy()

#instantiate both packages to use
encoder = OrdinalEncoder()
imputer = KNNImputer(n_neighbors=5)
# create a list of categorical columns to iterate over
cat_cols = ['gender','work_type','Residence_type','ever_married','smoking_status']

def encode(data):
    '''function to encode non-null data and replace it in the original data'''
    #retains only non-null values
    nonulls = np.array(data.dropna())
    #reshapes the data for encoding
    impute_reshape = nonulls.reshape(-1,1)
    #encode date
    impute_ordinal = encoder.fit_transform(impute_reshape)
    #Assign back encoded values to non-null values
    data.loc[data.notnull()] = np.squeeze(impute_ordinal)
    return data

#create a for loop to iterate through each column in the data
for columns in cat_cols:
    encode(df_recover[columns])

#data used to impute
impute_data=df_recover
impute_data=impute_data.drop(columns=['smoking_status','stroke'])

encode_data = pd.DataFrame(np.round(imputer.fit_transform(impute_data)),columns = impute_data.columns)

knn_bmi=pd.DataFrame()
knn_bmi['True Value'] =df.dropna()['bmi'].tolist()
knn_bmi['Missing'] =df_recover['bmi'].tolist()
knn_bmi['Prediction'] =encode_data['bmi'].tolist()
knn_bmi= knn_bmi[knn_bmi['Missing'].isna()]
knn_bmi=knn_bmi.round({'True Value':0})
knn_bmi.head(40)[['True Value', 'Prediction']]

print("Mean squared error =", round(sm.mean_squared_error(knn_bmi['True Value'], knn_bmi['Prediction']), 2)) 
print("\n \n")
#Smoking Status Imputations:

# median imputation for smoking data 

print('--> Imputation Methods for Smoking Status Categorical Feature. \n')
print('Imputation method 1: Median imputation')
values = df_recover[['smoking_status']]
imputer2 = SimpleImputer(missing_values=np.nan, strategy='median')
transformed_values2 = imputer2.fit_transform(values)

median_res=pd.DataFrame()
median_res['True Value'] =df.dropna()['smoking_status'].tolist()
median_res['Missing'] =df_recover['smoking_status'].tolist()
median_res['Prediction'] =transformed_values2.flatten().tolist()
median_res= median_res[median_res['Missing'].isna()]
median_res[['True Value', 'Prediction']]

print("Imputation Accuracy =", round(sm.accuracy_score(median_res['True Value'], median_res['Prediction']), 2)) 
print("\n")


# mode imputation
print('Imputation method 2: Mode imputation')
values = df_recover[['smoking_status']]
imputer2 = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
transformed_values2 = imputer2.fit_transform(values)

mode_res=pd.DataFrame()
mode_res['True Value'] =df.dropna()['smoking_status'].tolist()
mode_res['Missing'] =df_recover['smoking_status'].tolist()
mode_res['Prediction'] =transformed_values2.flatten().tolist()
mode_res= mode_res[mode_res['Missing'].isna()]
mode_res[['True Value', 'Prediction']]


print("Imputation Accuracy =", round(sm.accuracy_score(mode_res['True Value'], mode_res['Prediction']), 2)) 
print("\n")


print("imputation method 3: KNN (k=3)")


impute_data2=df_recover.drop(columns=['bmi','stroke'])
impute_data2['bmi']=df.dropna()['bmi']
imputer = KNNImputer(n_neighbors=9)
encode_data2 = pd.DataFrame(np.round(imputer.fit_transform(impute_data2)),columns = impute_data2.columns)



#impute_data2['bmi']=transformed_values.flatten().tolist()

#DataFrame(np.round(imputer.fit_transform(impute_data)),columns = impute_data.columns)

#imputer = KNNImputer(n_neighbors=2)
knn_smoke=pd.DataFrame()
knn_smoke['True Value'] =df.dropna()['smoking_status'].tolist()
knn_smoke['Missing'] =df_recover['smoking_status'].tolist()
knn_smoke['Prediction'] =encode_data2['smoking_status'].tolist()
knn_smoke= knn_smoke[knn_smoke['Missing'].isna()]
knn_smoke.head(40)[['True Value', 'Prediction']]


print("Imputation accuracy =", round(sm.accuracy_score(knn_smoke['True Value'], knn_smoke['Prediction']), 2)) 


# In[ ]:




