#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import KNNImputer, SimpleImputer


from sklearn.model_selection import train_test_split


# SAMPLING 

from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.under_sampling import NearMiss, RandomUnderSampler, AllKNN, NeighbourhoodCleaningRule
from imblearn.over_sampling import SMOTE, SMOTENC, ADASYN



# TRAINING
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier

# METRICS
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from xgboost.sklearn import XGBClassifier
from lightgbm.sklearn import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier


from scipy.stats import uniform
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import cross_validate


from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


import warnings
warnings.filterwarnings('ignore')


# In[5]:


def preprocess():
    print('Data Preprocessing in progress ... ')
    df = pd.read_csv('strokes.csv', delimiter = ';', encoding = 'utf-8')


    df_onehot = df.copy()

    df_onehot = pd.get_dummies(df_onehot, columns=['work_type'])
    encoder = OrdinalEncoder()

    cols=['smoking_status','ever_married', 'Residence_type', 'gender']
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
        encode(df_onehot[i])
    
    print('Categorical data succesfully encoded!')
    X,y=df_onehot.drop(columns=['stroke']), df_onehot[['id','stroke']]
    
    print('Training(0.8) and Testing(0.2) data succesfully split!')
    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=23)
    df_train= pd.merge(x_train, y_train, on='id')
    Test= pd.merge(x_test, y_test, on='id')   

    print('Missing values Imputation in progress ...')
    imputer = KNNImputer(n_neighbors=3)
    impute_data=df_train
    impute_data=impute_data.drop(columns=['bmi'])
    x_train2 = pd.DataFrame(np.round(imputer.fit_transform(impute_data)),columns = impute_data.columns)
    x_train2['id'] = x_train2['id'].astype(np.int64)
    df_train= pd.merge(x_train2, df_train[['id','bmi']], on='id')
    
    print('Smoking status entries succesfully imputed!')



    df_train = pd.get_dummies(df_train, columns=['smoking_status'])
    x_test= pd.get_dummies(x_test, columns=['smoking_status'])

    X_test= pd.merge(x_test, y_test, on='id')

    from sklearn.impute import SimpleImputer

    #mean imputation
    values = df_train[['bmi']]
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    impute_bmi = imputer.fit_transform(values)
    df_train[['bmi']]=impute_bmi
    
    print('BMI entries succesfully imputed!')


    df_train = df_train[['id', 'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
           'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status_0.0','smoking_status_1.0','smoking_status_2.0',
           'work_type_Govt_job', 'work_type_Never_worked', 'work_type_Private',
           'work_type_Self-employed', 'work_type_children', 'stroke']]

    X_test= pd.merge(x_test, y_test, on='id')
    df_test = X_test[['id', 'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
           'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status_0.0','smoking_status_1.0','smoking_status_2.0',
           'work_type_Govt_job', 'work_type_Never_worked', 'work_type_Private',
           'work_type_Self-employed', 'work_type_children', 'stroke']]

    numerical=['age','avg_glucose_level','bmi']

    categorical=['id', 'gender','hypertension', 'heart_disease', 'ever_married',
           'Residence_type', 'smoking_status_0.0',
           'smoking_status_1.0', 'smoking_status_2.0', 'work_type_Govt_job',
           'work_type_Never_worked', 'work_type_Private',
           'work_type_Self-employed', 'work_type_children', 'stroke']
    for i in numerical:
        df_train[i]=df_train[i].astype(np.float64)

    for i in categorical:
        df_train[i]=df_train[i].astype(np.int64)

    for i in categorical[1:]:
        df_train[i]=df_train[i].astype(np.int64)
        df_test[i]=df_test[i].astype(np.int64)

    df_test=df_test.drop(columns='id')
    df_train=df_train.drop(columns='id')

    df_test=df_test.dropna()



    #val, test = train_test_split(df_test, train_size=0.5, test_size=0.5, random_state=123)
    train_y = df_train['stroke']
    test_y = df_test['stroke']
    #val_y = val['stroke']

    df_train.drop(['stroke'], axis=1, inplace=True)
    df_test.drop(['stroke'], axis=1, inplace=True)
    #val.drop(['stroke'], axis=1, inplace=True)


    for i in df_train.columns:
        if df_train[i].dtype=='int64':
            df_train[i]=df_train[i].astype('object')


    cat_cols = df_train.loc[:,df_train.dtypes == "object"].columns
    num_cols = df_train.loc[:,df_train.dtypes != "object"].columns

    
    num_pipeline = Pipeline([
            ('std_scaler', StandardScaler())
        ])

    cat_pipeline = Pipeline([
            ('one_hot', OneHotEncoder(handle_unknown='ignore'))
        ])

    full_pipeline = ColumnTransformer([
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols)
        ])


    train = full_pipeline.fit_transform(df_train, train_y)
    test = full_pipeline.fit_transform(df_test)
    print('Numerical features succesfully scaled!')
    train_y=train_y.astype('int')

    sampling = RandomUnderSampler()
    train, train_y = sampling.fit_resample(train, train_y.ravel())
    print('Data classes succesfully balanced!')
    
    
    print('------ The Data has been preprocessed ------ ')
    print('There are ', str(len(train)), ' entries in the training data and', str(len(test)),'entries in the testing data.')
    return train, train_y, test, test_y

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = preprocess()


# In[ ]:




