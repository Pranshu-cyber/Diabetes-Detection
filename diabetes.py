from numpy import *
from pandas import *
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

#Load Dataset
data=read_csv("diabetes.csv")

#Seperate dataset into test and train data
X=data.drop(columns='Outcome', axis=1)
Y=data['Outcome']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=2, test_size=.2, stratify=Y)


#Data Standardization
scalar=StandardScaler()
scalar.fit_transform(X)

#Training Model
model=svm.SVC(kernel='rbf')
result=model.fit(X_train,Y_train)

#Check Accuracy Score
X_train_prediction=model.predict(X_train)
acc1=accuracy_score(X_train_prediction,Y_train)
print("Training Data:", acc1)
X_test_prediction=model.predict(X_test)
acc2=accuracy_score(X_test_prediction, Y_test)
print("Test Data:", acc2)

#Run The Model
input_data=[4,173,70,14,168,29.7,0.361,33]
input_data=asarray(input_data)
reshaped_data=input_data.reshape(1,-1)
std_data=scalar.fit_transform(reshaped_data)
result=model.predict(std_data)
if result==0:
    print("Not Diabetic")
elif result==1:
    print("Diabetic")
else:
    print("Error")