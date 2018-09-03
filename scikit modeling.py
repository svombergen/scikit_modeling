# Update pip and packages:
# python -m pip install --upgrade pip
# pip list --outdated --format=freeze | grep -v '^\-e' | cut -d = -f 1  | xargs -n1 pip install -U

# Import packages
import numpy as np
import pandas as pd
import pyodbc
import getpass
import imblearn
import matplotlib as plt

# Connection details
driver= '{ODBC Driver 13 for SQL Server}'
server = ''
port = 1433
username = ''
database = ''
pwd = getpass.getpass()

cnxn = pyodbc.connect('Driver=' + driver + 
                      ';Server='+ server + 
                      ',' + str(port) + 
                      ';Uid=' + username + 
                      ';Pwd=' + pwd + 
                      ';Database=' + database + 
                      ';Encrypt=yes;TrustServerCertificate=no;' + 
                      'Connection Timeout=30;Authentication=ActiveDirectoryPassword;') 

query = "SELECT * FROM [MOD].vw_abt_final_undersampled"
df = pd.read_sql_query(sql=query, con=cnxn)
cnxn.close()

# Remove index keys and useless columns
df.set_index(keys=['weekid','DL_kode'], drop=True, append=False, inplace=True, verify_integrity=True) # set index columns
cols_to_remove = [c for c in df.columns if '_flag_' in c and c != 'seventy_flag_in_four_weeks'] 
df.drop(columns=cols_to_remove, inplace=True) # remove alle target vars behalve 1

# Create X and y
X = df.drop(columns='seventy_flag_in_four_weeks', inplace=False) # predictors
y = df['seventy_flag_in_four_weeks'] # target

# Prerequisites to train model
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import TomekLinks

def trainModelWithResults(model, X, y,rd_state=None,autoscale=1,usetomeklinks=1):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, stratify=y, random_state=rd_state) # stratify the split because we have unbalanced target
    if autoscale==1:
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
    if usetomeklinks==1:
        tl = TomekLinks(return_indices=False)
        X_train, y_train = tl.fit_sample(X_train, y_train)
    mfitted = model.fit(X_train,y_train)
    predictions = mfitted.predict(X_test)
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))

def printFeatureImportances(model, X):
    important_features = pd.Series(data=model.feature_importances_,index=X.columns)
    important_features.sort_values(ascending=False,inplace=True)
    print(important_features)

# Classification with scikit GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(n_estimators=300,learning_rate=0.25, max_depth=10)
trainModelWithResults(gb,X,y, autoscale=1)

# Classification with scikit RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100)
trainModelWithResults(rf,X,y, autoscale=1)

# Neural net
from sklearn.neural_network import MLPClassifier
nncl = MLPClassifier(solver='lbfgs')
trainModelWithResults(nncl,X,y, autoscale=1)

# xgboost trials, seems no extra value as addition to sklearn GBC
# Run cell
#%%
import xgboost as xgb
# xgb = xgboost.DMatrix(df,label='seventy_flag_in_four_weeks')
param = {'max_depth': 5, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic','nthread':4,'eval_metric':'auc'}
num_round = 10
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, stratify=y) # stratify the split because we have unbalanced target
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
evallist = [(dtest, 'eval'), (dtrain, 'train')]
bst = xgb.train(param,dtrain,num_round,evallist)
xgb.plot_importance(bst)
xgb.plot_tree(bst, num_trees=2)
plt.pyplot.show()

# GridSearchCV
from sklearn.model_selection import GridSearchCV
parameters = {'learning_rate':[0.05, 0.1, 0.25, 0.5, 0.75, 1]
    ,'max_depth':[1, 3, 5, 10]
    ,'n_estimators':[25, 100, 500]}
gcv = GridSearchCV(estimator = GradientBoostingClassifier(), param_grid=parameters, n_jobs=4, scoring='f1')
gcv.fit(X,y)
trainModelWithResults(gcv.best_estimator_,X,y, autoscale=1)
printFeatureImportances(gcv.best_estimator_,X)
print(gcv.best_estimator_)

# Smote alleen op je training, tomek links
from imblearn.under_sampling import TomekLinks
tl = TomekLinks(return_indices=False)
X_resampled, y_resampled = tl.fit_sample(X_train, y_train)



# Ensembling evt voting classifier




# Threshold tweaking -> function with input target precision/recall/f1 and 
# Categorical values -> onehotencoder

