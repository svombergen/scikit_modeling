# Import packages
import numpy as np
import pandas as pd
import pyodbc
import getpass
import imblearn

# Connection details
driver= '{ODBC Driver 13 for SQL Server}'
server = 'eviscdmsrv.database.windows.net'
port = 1433
username = 's.vanombergen@vanlanschot.com'
database = 'evioutflow'
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

def trainModelWithResults(model, X, y,rd_state=None,autoscale=1):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, stratify=y, random_state=rd_state) # stratify the split because we have unbalanced target
    if autoscale==1:
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
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
gb = GradientBoostingClassifier(n_estimators=100,learning_rate=0.25, max_depth=5)
trainModelWithResults(gb,X,y, autoscale=0)

# Classification with scikit RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100)
trainModelWithResults(rf,X,y, autoscale=0)

# Neural net
from sklearn.neural_network import MLPClassifier
nncl = MLPClassifier(solver='lbfgs')
trainModelWithResults(nncl,X,y, autoscale=0)

# GridSearchCV
learning_rates = [0.25]                 #learning_rates = [0.05, 0.1, 0.25, 0.5, 0.75, 1]
n_estimators = [150]                    #n_estimators = [25, 100, 250, 600]
subsample = [1]                         #subsample = [0.1, 0.25, 0.5, 1]
max_depth = [5]                        #max_depth = [1, 3, 5, 10]  
from sklearn.model_selection import GridSearchCV
parameters = {'learning_rate':[0.05, 0.1, 0.25, 0.5, 0.75, 1]}
gcv = GridSearchCV(estimator = gb, param_grid=parameters, n_jobs=4, scoring='f1')
gcv.fit(X,y)
# Smote
gcv.cv_results_

# Threshold tweaking -> function with input target precision/recall/f1 and 

