#!/usr/bin/env python
# coding: utf-8

# In[211]:


import numpy as np
# TRAINING
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier

# METRICS
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score, recall_score, precision_recall_fscore_support, f1_score

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from xgboost.sklearn import XGBClassifier
from lightgbm.sklearn import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV



from scipy.stats import uniform
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import cross_validate

import warnings
warnings.filterwarnings('ignore')


# In[235]:


from Preprocessing_stroke_data import preprocess
X_train, y_train, X_test, y_test=preprocess()

def test_threshold(probas, test_y):
    results = []
    for i in range(48, 52):
        result = (probas > i / 100).astype(int)
        results.append((roc_auc_score(test_y, result), i / 100))
    return sorted(results, key=(lambda x : x[0]), reverse=True)

def tuned_RF(grid_search=False, threshold=0.5):
    if grid_search:

        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start = 90, stop = 140, num = 7)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(2, 70, num = 25)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [5, 10, 12, 15, 17, 20]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 3, 4, 5, 6, 7, 8, 9 ,10, 11, 12, 13, 14]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}
        print(random_grid)

        # Use the random grid to search for best hyperparameters
        # First create the base model to tune
        rf = RandomForestClassifier()
        # Random search of parameters, using 3 fold cross validation, 
        # search across 100 different combinations, and use all available cores
        rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
        # Fit the random search model
        rf_random.fit(X_train, y_train)
        pr=train_and_evaluate(rf_random.best_estimator_, X_train, y_train, X_test, y_test,train_model=False)
        return pr,rf_random.best_estimator_
    else:
        
        #tuned random forest model with saved best parameters 
        rf_best=RandomForestClassifier(n_estimators= 97, min_samples_split= 17, min_samples_leaf= 9, max_features= 'auto', max_depth= 58, bootstrap= True)
        rf_best.fit(X_train, y_train)
        return train_and_evaluate(rf_best, X_train, y_train, X_test, y_test,train_model=False, threshold=threshold),rf_best

def tuned_LGBM(grid_search=False):
    if grid_search:

        a = LGBMClassifier()
        a_random = RandomizedSearchCV(a, {'min_child_weight': [1, 5, 10],'gamma': [0.5, 1, 1.5, 2, 5],'subsample': [0.6, 0.8, 1.0],'colsample_bytree': [0.6, 0.8, 1.0],'max_depth': [2, 4, 6]}, n_iter=3, scoring='roc_auc', n_jobs=5, verbose=3, random_state=42)

        a_random.fit(train, train_y)
        return train_and_evaluate(a_random.best_estimator_, train, train_y, test, test_y, train_model=False), a_random.best_estimator_
    else:
        
        #tuned random forest model with saved best parameters 
        lgbm_best=LGBMClassifier(max_depth=2, min_child_weight=1, subsample=0.6)
        lgbm_best.fit(X_train, y_train)
        return train_and_evaluate(lgbm_best, X_train, y_train, X_test, y_test,train_model=False), lgbm_best

def tuned_SVC(grid_search=False):
    if grid_search:
        
        parameters = [
            {
            'C': [0.01, 0.5, 1, 2, 5, 10],
            'kernel' : ['poly'],
            'degree' : [2,3],
            'gamma': ['scale', 'auto'],
            'coef0': [0.5, 1, 2, 3],
            'class_weight': ['balanced', None]    
            },
            {
            'C': [0.01, 0.5, 1, 2, 5, 10],
            'kernel' : ['rbf', 'sigmoid'],
            'gamma': ['scale', 'auto'],
            'class_weight': ['balanced', None]    
            },
            {
            'C': [0.01, 0.5, 1, 2, 5, 10],
            'kernel' : ['linear'],
            'class_weight': ['balanced', None] 
            }
        ]

        model = SVC(probability=True)
        grid_search = GridSearchCV(model,
                                   param_grid=parameters,
                                   cv=3,
                                   scoring='roc_auc',
                                   refit='roc_auc',
                                   )

        r = grid_search.fit(X_train, y_train)

        return train_and_evaluate(r.best_estimator_, X_train, y_train, X_test, y_test, train_model=False), r
    else:
        svc=SVC(C=0.01, class_weight='balanced', kernel='linear', probability=True)
        svc.fit(X_train, y_train)
        return train_and_evaluate(svc, X_train, y_train, X_test, y_test, train_model=False), svc
    
def train_and_evaluate(model, train, train_y, test, test_y, eq=None, train_model=True, threshold=0.5):
    r=[]
    a=[]
    auc=[]
    for i in range(3):
        if train_model:
            model.fit(train, train_y)

        results = model.predict_proba(test)
        proba = results[:,1]
        results = (results[:,1] > threshold).astype(int)


        cm=confusion_matrix(test_y, results)

        acc=(cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][1]+cm[1][0])
        roc=roc_auc_score(test_y, proba)
        recall=precision_recall_fscore_support(test_y, results)[1][1]
        a.append(acc)
        r.append(recall)
        auc.append(roc)
    #print(classification_report(test_y, results))
    print('Recall :', str(round(sum(r)/3,4)), '; Accuracy :',str(round(sum(a)/3,4)),'; ROC AUC :',str(round(sum(auc)/3,4)))
    
    return proba

def train_models():
    print('---------   Initial model selection :   ---------\n')
    
    initial_models=[RandomForestClassifier(), XGBClassifier(eval_metric='logloss'), SVC(probability=True), LogisticRegression(), LGBMClassifier(), DecisionTreeClassifier(),KNeighborsClassifier()]
    ini_names=['Random Forest Classifier','XGBoost Classifier', 'Support Vector Classifier','Logistic regression', 'Light Gradient Boost Classifier', 'Decision Tree Classifier', 'KNN Classifier',]


    for i in range(len(initial_models)):
        if ini_names[i]=='Logistic regression':
            print(ini_names[i])
            lg_pr=train_and_evaluate(initial_models[i], X_train, y_train, X_test, y_test
        )
        else:
            print(ini_names[i])
            train_and_evaluate(initial_models[i], X_train, y_train, X_test, y_test
            )

    print('\n---- Parameter tuning using Random Grid Search : ----\n')
    
    print('Tuned Random Forest Classifier:')
    rf_pr, rf_mo=tuned_RF()
    
    print('Tuned LGBM Classifier:')
    lgbm_pr, lgbm_mo =tuned_LGBM()
    
    print('Tuned SVC Classifier:')
    svc_pr, svc_mo =tuned_SVC()
    
    print('\n ---------  Optimal Treshold Tuning :   ---------\n')
    
    print('Tuned Random Forest :')
    forest_best_rec_score = test_threshold(rf_pr, y_test)[0]
    train_and_evaluate(rf_mo, X_train, y_train, X_test, y_test, train_model=False, threshold=forest_best_rec_score[1])

    print('Tuned LGBM Classifier :')
    lgbm_best_rec_score = test_threshold(lgbm_pr, y_test)[0]
    train_and_evaluate(lgbm_mo, X_train, y_train, X_test, y_test, train_model=False, threshold=lgbm_best_rec_score[1])

    print('Tuned SVC Forest :')
    SVC_best_rec_score = test_threshold(svc_pr, y_test)[0]
    train_and_evaluate(svc_mo, X_train, y_train, X_test, y_test, train_model=False, threshold=SVC_best_rec_score[1])

    print('Tuned Logistic Regression :')
    lg_best_rec_score = test_threshold(lg_pr, y_test)[0]
    train_and_evaluate(LogisticRegression(), X_train, y_train, X_test, y_test, train_model=True, threshold=lg_best_rec_score[1])

    
    
if __name__ == "__main__":
    train_models()

