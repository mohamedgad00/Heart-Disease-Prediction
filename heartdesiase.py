# Importing Libaries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Importing Dataset
dataset=pd.read_csv('heart.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

print(x)
print(y)


#Data Preprocissing
from sklearn.preprocessing import LabelEncoder
le1 = LabelEncoder()
le2 = LabelEncoder()
le6 = LabelEncoder()
le8 = LabelEncoder()
le10 = LabelEncoder()
x[:,1]=le1.fit_transform(x[:,1])
x[:,2]=le2.fit_transform(x[:,2])
x[:,6]=le6.fit_transform(x[:,6])
x[:,8]=le8.fit_transform(x[:,8])
x[:,10]=le10.fit_transform(x[:,10])
print(x)

#Spliting DataSet Into Traning & Test Set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
print (x_train)
print(y_test)

# Applying Dimensionality Reduction
from sklearn.decomposition import PCA
pca = PCA(n_components=5)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)
print(x_train_pca)
print(x_test_pca)

#Training DataSet
#SVM
from sklearn.svm import SVC
model_svm=SVC()
model_svm.fit(x_train,y_train)
y_pred_svm=model_svm.predict(x_test)

# Training DataSet After Dimensionality Reduction
model_svm_pca = SVC()
model_svm_pca.fit(x_train_pca, y_train)

# Testing the Model
y_pred_svm_pca = model_svm_pca.predict(x_test_pca)
from sklearn.metrics import accuracy_score
Svm_Acc=accuracy_score(y_test,y_pred_svm)
print(Svm_Acc)

# Testing Accuracy After Dimensionality Reduction
svm_Acc_pca = accuracy_score(y_test, y_pred_svm_pca)
print(svm_Acc_pca)
# new patient input (already encoded values as the model won't accept strings  as it only accept numerical values only )
# ONLY 11 features (not 13! as PCA decresed the  amount of  features  )
new_patient = np.array([[63, 1, 0, 145, 233, 1, 1, 150, 0, 2.3, 2]])
# Now it's safe to transform
new_patient_pca = pca.transform(new_patient)
# Predict
prediction_pca = model_svm_pca.predict(new_patient_pca)
print("Prediction with PCA:", prediction_pca)
if (prediction_pca == 0):
    print("Patient does not suffer from  any heart proplems ")
else:
    print("Patient does suffer from a heart proplem")



# Visualization of Accuracy Comparison
accuracies = [Svm_Acc, svm_Acc_pca]
labels = ['Before Dimensionality Reduction', 'After Dimensionality Reduction']
print(f"Accuracy without Dimensionality Reduction: {Svm_Acc}")
print(f"Accuracy after Dimensionality Reduction (PCA): {svm_Acc_pca}")
plt.bar(labels, accuracies, color=['blue', 'green'])
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison')
plt.ylim(0, 1)
plt.show()



