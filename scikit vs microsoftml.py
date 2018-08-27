
# Import packages
import numpy as np
import pandas as pd
import pyodbc
import getpass
import microsoftml

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

query = "SELECT * FROM [MOD].[ABT_2016_2018_final_reduced2] --WHERE weekid IN ('2018-12','2018-08','2018-04','2017-52')"
df = pd.read_sql_query(sql=query, con=cnxn)
cnxn.close()

# Info on data
df.info()
print(df)

# Remove index keys and useless columns
df.set_index(keys=['weekid','DL_kode'], drop=True, append=False, inplace=True, verify_integrity=True) # set index columns
cols_to_remove = [c for c in df.columns if '_flag_' in c and c != 'seventy_flag_in_four_weeks'] 
df.drop(columns=cols_to_remove, inplace=True) # remove alle target vars behalve 1

# Train/test split
X = df.drop(columns='seventy_flag_in_four_weeks', inplace=False) # predictors
X_reduced = df[['sl_monthly_perf_30','AUM','dagperformance','Age','loyalty','Avg_cumm_nr_calls','Avg_cumm_nr_emails','emo_score','TransExSpeed_avg','Heeft_kennis','PC3_Indexes','nr_products','IsCautious']]
X = X_reduced
y = df['seventy_flag_in_four_weeks'] # target
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, stratify=y) # stratify the split because we have unbalanced target

# Classification with scikit GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

learning_rates = [0.25]                 #learning_rates = [0.05, 0.1, 0.25, 0.5, 0.75, 1]
n_estimators = [100]                    #n_estimators = [25, 100, 250, 600]
subsample = [0.1, 0.25, 0.5, 1]

for lr in learning_rates:
    for est in n_estimators:
        for ss in subsample:
            print(str(lr) + ',' + str(est)+ ',' + str(ss))
            gb = GradientBoostingClassifier(n_estimators=est,learning_rate=lr,subsample=ss).fit(X_train, y_train)
            predictions = gb.predict(X_test)
            print(confusion_matrix(y_test, predictions))
            print(classification_report(y_test, predictions))


important_features = pd.Series(data=gb.feature_importances_,index=X.columns)
important_features.sort_values(ascending=False,inplace=True)
important_features