from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
clf  = tree.DecisionTreeClassifier() #classifier 1
clfs = SVC(gamma='auto')  #classifier 2
clfk = KNeighborsClassifier(n_neighbors=3) #classifier 3
clfr = RandomForestClassifier() #classifier 4
clfa = AdaBoostClassifier() #classifier 5

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],[190, 90, 47], [175, 64, 39], [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female','female', 'male', 'male']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.3)

#training model
clf  = clf.fit(X_train, Y_train)
clfs = clfs.fit(X_train,Y_train)
clfk = clfk.fit(X_train,Y_train)
clfr = clfr.fit(X_train,Y_train)
clfa = clfa.fit(X_train,Y_train)
#checking accuracy
print(clf.score(X_test,Y_test))
print(clfs.score(X_test,Y_test))
print(clfk.score(X_test,Y_test))
print(clfr.score(X_test,Y_test))
print(clfa.score(X_test,Y_test))
