import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split 
from sklearn import metrics
data = pd.read_csv('bill_authentication.csv')

x = data[['Variance','Skewness','Curtosis','Entropy']]
y = data['Class']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.20)

svc = SVC(kernel = 'linear')
svc.fit(x_train,y_train)

y_pred = svc.predict(x_test)

print ('Confusion Matrix')
print metrics.confusion_matrix(y_pred,y_test)

print('Report')
print metrics.classification_report(y_pred,y_test)

