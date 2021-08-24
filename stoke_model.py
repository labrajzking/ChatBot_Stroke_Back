import numpy as np
import pandas as pd
import pickle
data=pd.read_csv(r"C:\Users\oussema\Desktop\python\flask\one\healthcare-dataset-stroke-data.csv")
data['bmi'] = data['bmi'].fillna(data['bmi'].mean())
data = data.drop(columns ='id')
data['gender'] = data['gender'].replace('Other', list(data.gender.mode().values)[0])
bmi_outliers=data.loc[data['bmi']>50]
bmi_outliers['bmi'].shape
data["bmi"] = pd.to_numeric(data["bmi"])
data["bmi"] = data["bmi"].apply(lambda x: 50 if x>50 else x)
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
le = LabelEncoder()
hot=OneHotEncoder()
data['gender'] = le.fit_transform(data['gender'])
data['ever_married'] = le.fit_transform(data['ever_married'])
data['Residence_type'] = le.fit_transform(data['Residence_type'])
data['work_type'] = le.fit_transform(data['work_type'])
data['smoking_status'] = le.fit_transform(data['smoking_status'])
data = data.drop(['ever_married'], axis = 1)
from sklearn.preprocessing import StandardScaler
s = StandardScaler()
columns = ['avg_glucose_level','bmi','age']
stand_scaled = s.fit_transform(data[['avg_glucose_level','bmi','age']])
stand_scaled = pd.DataFrame(stand_scaled,columns=columns)

data=data.drop(columns=columns,axis=1)
#enc_work_type=pd.DataFrame(hot.fit_transform(data[['work_type']]).toarray())
#enc_smoking_status=pd.DataFrame(hot.fit_transform(data[['smoking_status']]).toarray())
#df_en =pd.concat([data,enc_work_type],axis=1)
#df_en=pd.concat([df_en,enc_smoking_status],axis=1)
#data.drop(columns=['work_type','smoking_status'],inplace=True)
df = pd.concat([data, stand_scaled], axis=1)
x=df.drop(['stroke'], axis=1)
y=df['stroke']
print (x.head(5))
# Models
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Evaluation
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state= 124)
#smote=SMOTE()
#x_train,y_train=smote.fit_resample(x_train,y_train)
#x_test,y_test=smote.fit_resample(x_test,y_test)
#xgc=XGBClassifier(objective='binary:logistic',n_estimators=1000,max_depth=5,learning_rate=0.001,n_jobs=-1)
#xgc.fit(x_train,y_train)
model=RandomForestClassifier()
model.fit(x_train,y_train)
# Pickle serializes objects so they can be saved to a file, and loaded in a program again later on.
pickle.dump(model , open('model.pkl','wb'))
