import numpy as np
import pandas as pd
import seaborn as sns
sns.set_palette('husl')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Normalizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

from sklearn.metrics import  ConfusionMatrixDisplay

# %%
#Partie I – Acquisition des données

#1 /lecture du dataset
col_name = ['sepal-lenght','sepal-width','petal-lenght','petal-width','specie']

iris = pd.read_csv('./iris/iris.data', names = col_name)
iris = iris.replace({"specie":  {"Iris-setosa":1,"Iris-versicolor":2, "Iris-virginica":3}})

print(iris.shape)
print(iris.head())

array = iris.values

X = array[:,0:4]
y = array[:,4]


print(X)

print(y)

# %%
# 2/ Definir la fonction distance()

def distance(x_train, x_test_point):

  
  #- x_train: correspondant aux données d'entraînement
  #- x_test_point: correspondant au point de test


  distances= []  ## crée une liste vide appelée distances
  for row in range(len(x_train)): ## Boucle sur les lignes de x_train
      current_train_point= x_train[row] #Obtenez-les point par point
      current_distance= 0 ## initialiser la distance à zéro

      for col in range(len(current_train_point)): ##Boucle sur les colonnes de la ligne
          
          current_distance += (current_train_point[col] - x_test_point[col]) **2
       
      current_distance= np.sqrt(current_distance)

      distances.append(current_distance) ## Append les distances

  # Storer les distances dans un dataframe
  distances= pd.DataFrame(data=distances,columns=['dist'])
  return distances


# %%
# 3 /statistiques sur les données
print("quelque statistiques sur les données")
print(iris.describe())
print(iris.groupby('specie').size())




# %%
#Partie II – Classification à partir de plusieurs méthodes de machine learning





# %%
# creation des histogrammes selon chaque caractéristique

Y_Data = np.array([np.average(X[:, i][y==j].astype('float32')) for i in range (X.shape[1])
 for j in (np.unique(y))])
Y_Data_reshaped = Y_Data.reshape(4, 3)
Y_Data_reshaped = np.swapaxes(Y_Data_reshaped, 0, 1)
X_axis = np.arange(len(col_name)-1)
width = 0.25

plt.bar(X_axis, Y_Data_reshaped[0], width, label = 'Setosa')
plt.bar(X_axis+width, Y_Data_reshaped[1], width, label = 'Versicolour')
plt.bar(X_axis+width*2, Y_Data_reshaped[2], width, label = 'Virginica')
plt.xticks(X_axis, col_name[:4])
plt.xlabel("Features")
plt.ylabel("Value in cm.")
plt.legend(bbox_to_anchor=(1.3,1))
plt.show()


# diviser les données en ensembles de train et de test
x_train, x_test, y_train, y_test= train_test_split(X, y,test_size= 0.5,random_state= 42)
x_train= np.asarray(x_train)
y_train= np.asarray(y_train)

x_test= np.asarray(x_test)
y_test= np.asarray(y_test)
print(f'training set size: {x_train.shape[0]} samples \ntest set size: {x_test.shape[0]} samples')
# %%
# Normalisation des donné


scaler= Normalizer().fit(x_train) # scalar est adapté à l'ensemble d'entraînement
normalized_x_train= scaler.transform(x_train) # scaler est appliqué à l'ensemble d'apprentissage
normalized_x_test= scaler.transform(x_test) # scalar est appliqué à l'ensemble de test
print('x train before Normalization')
print(x_train[0:5])
print('\nx train after Normalization')
print(normalized_x_train[0:5])
# %%
## avant normalisation
# Afficher les relations entre les variables ; code couleur par type d'espèce
di= {0.0: 'Setosa', 1.0: 'Versicolor', 2.0:'Virginica'} # dictionary

before= sns.pairplot(iris.replace({'specie': di}), hue= 'specie')
before.fig.suptitle('Pair Plot of the dataset Before normalization', y=1.08)
# %%
## apres la normalisation
iris_df_2= pd.DataFrame(data= np.c_[normalized_x_train, y_train],columns= ['sepal-lenght','sepal-width','petal-lenght','petal-width','specie'] )
di= {0.0: 'Setosa', 1.0: 'Versicolor', 2.0: 'Virginica'}
after= sns.pairplot(iris_df_2.replace({'specie':di}), hue= 'specie')
after.fig.suptitle('Pair Plot of the dataset After normalization', y=1.08)
# %%


#knn
#Fitting knn au Training set

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit( x_train, y_train)
# %%

#Prediction du Test set resultats
y_pred = classifier.predict(x_test)
score =accuracy_score(y_test,y_pred)
print("Knn predition" , y_pred)
print("Knn score" ,score)
# %%
#matrice de confision
cm = confusion_matrix(y_test, y_pred)
print('knn confusion matrix',cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=classifier.classes_)
disp.plot()
plt.title("Knn confusion matrix")
plt.show()
print("À travers cette matrice, nous voyons qu'il y a 2 versicolor virginica que nous prédisons être versicolor.")
#%%
#raport du classification
print(classification_report(y_test, y_pred))

# %%
# précision du classificateur
accuracies = cross_val_score(estimator = classifier, X =  x_train, y = y_train, cv = 10)
# %%
# Prediction des probabilités
print("knn Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("knn Standard Deviation: {:.2f} %".format(accuracies.std()*100))
# %%


#%%
#Regression Logistic
# Fitting  Regression Logistic  au Training set
classifier = LogisticRegression(random_state = 0, solver='lbfgs', multi_class='auto',max_iter=1000)
classifier.fit( x_train, y_train)
# %%
# Prediction du Test set resultats
y_pred = classifier.predict(x_test)
score =accuracy_score(y_test,y_pred)
print("LR predition" , y_pred)
print("LR score" ,score)
# %%
#matrice de confision
cm = confusion_matrix(y_test, y_pred)
print('LR confusion matrix',cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=classifier.classes_)
disp.plot()
plt.title("LR confusion matrix")
plt.show()
# %%
#raport du classification
print(classification_report(y_test, y_pred))
# %%
# précision du classificateur
accuracies = cross_val_score(estimator = classifier, X =  x_train, y = y_train, cv = 10)

# %%
# Prediction des probabilités
print("LR Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("LR Standard Deviation: {:.2f} %".format(accuracies.std()*100))

# %%
#Random Forest
# Fitting RandomForest au Training set

classifier=RandomForestClassifier(n_estimators=100)
classifier.fit( x_train, y_train)
# %%
# Prediction du Test set resultats

y_pred = classifier.predict(x_test)
score =accuracy_score(y_test,y_pred)
print("RF predition" , y_pred)
print("RF score" ,score)
# %%

#raport du classification
print(classification_report(y_test, y_pred))
# %%
#matrice de confision
cm = confusion_matrix(y_test, y_pred)
print('RF confusion matrix',cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=classifier.classes_)
disp.plot()
plt.title("RF confusion matrix")
plt.show()
print("À travers cette matrice, nous voyons qu'il y a 2 versicolor virginica que nous prédisons être versicolor.")
# %%

# Prediction des probabilités
accuracies = cross_val_score(estimator = classifier, X =  x_train, y = y_train, cv = 10)
print("RF Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("RFS Standard Deviation: {:.2f} %".format(accuracies.std()*100))
# %%


#%%
#SVM
# Fitting svm au Training set
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(x_train, y_train)
# %%
# Prediction du Test set resultats
y_pred = classifier.predict(x_test)
score =accuracy_score(y_test,y_pred)
print("svm predition" , y_pred)
print("svm score" ,score)
# %%
#raport du classification
print(classification_report(y_test, y_pred))
# %%
#matrice de confision
cm = confusion_matrix(y_test, y_pred)
print('svm confusion matrix',cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=classifier.classes_)
disp.plot()
plt.title("svm confusion matrix")
plt.show()
# %%

# Prediction des probabilités
accuracies = cross_val_score(estimator = classifier, X =  x_train, y = y_train, cv = 10)
print("svm Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("svm Standard Deviation: {:.2f} %".format(accuracies.std()*100))

# %%
