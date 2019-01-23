#wont use numpy here, we'll just work out with scikit-learn.

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

# sklearn has iris dataset preloaded, so just call for the dataset.
iris1 = load_iris()
type(iris1)

print iris1.target 
print iris1.feature_names 
print iris1.data

print type(iris1.target) 
print type(iris1.data)
print iris1.target.shape 
print iris1.data.shape

X = iris1.data
y = iris1.target

print X.shape
print y.shape

knn = KNeighborsClassifier(n_neighbors=1)

#print knn
knn.fit(X, y)
#X_new = [[3,5,4,2],[5,4,3,2]]
print (knn.predict([[3,5,4,2]]))

