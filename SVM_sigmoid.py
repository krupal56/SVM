import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn import metrics
iris = load_iris()

x = pd.DataFrame(iris.data,columns = iris.feature_names[:])
y = pd.DataFrame(iris.target,columns = ['Species'])

x_train,x_test,y_train,y_test = train_test_split(x,y)

print 'Sigmoid kernel'

svc = SVC(kernel = 'sigmoid')
svc.fit(x_train,y_train)

y_pred = svc.predict(x_test)

print ('Confusion Matrix')
print metrics.confusion_matrix(y_test,y_pred)


print ('Classification report')
print metrics.classification_report(y_test,y_pred)

