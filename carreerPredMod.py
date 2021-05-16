from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

#clf = tree.DecisionTreeClassifier()
clf = RandomForestClassifier()

#[ 'Sub1', 'Sub2', 'Sub3', 'Sub4',    'Sub5',    'Sub6',    'Sub7',    'Sub8',    'Sub9',    'Sub10', 'Sub11', 'Sub12']
X=[
 [82,  67,    92,    82,    71,    92,    82,    87,    69,    61,    98,    92,],
 [66,  65,    90,    82,    72,    92,    82,    78,    70,    63,    95,    90],
 [67,  63,    88,    62,    73,    72,    82,    87,    71,    65,    92,    88],
 [82,  61,    86,    82,    74,    92,    82,    87,    72,    67,    89,    86],
 [82,  77,    84,    82,    75,    82,    82,    79,    77,    69,    86,    84],
 [62,  68,    82,    82,    76,    72,    82,    87,    78,    71,    83,    86],
 [82,  59,    80,    82,    77,    92,    82,    87,    79,    73,    80,    85],
 [72,  50,    78,    82,    78,    68,    82,    87,    80,    75,    77,    84],
 [82,  41,    76,    82,    79,    92,    82,    87,    81,    77,    74,    83]
 ]

Y=['C1','C2', 'C3','C4','C4','C1','C2','C4','C4']

predict=[87, 77, 84, 77, 71, 82, 62, 47, 89, 71, 98, 92]

clf = clf.fit(X, Y)
prediction = clf.predict([[87, 77, 84, 77, 71, 82, 62, 47, 89, 71, 98, 92]])
print("----------------- Decision Tree Classification  Without Normallization----------------")
print(prediction)

probs = clf.predict_proba([[87, 77, 84, 77, 71, 82, 62, 47, 89, 71, 98, 92]])
print(probs)
print("------------------ Decision Tree Classification With MiniMax Scaller-----------------")
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X)
NEW_X=scaler.transform(X)
predict=scaler.transform([[87, 77, 84, 77, 71, 82, 62, 47, 89, 71, 98, 92]])
clf = clf.fit(NEW_X, Y)



prediction = clf.predict(predict)

print(prediction)
probs = clf.predict_proba(predict)
print(probs)
Decision_Matrix={'Decision-Tree':probs}
print("---------------K-NN lassification Without Normallization-----------------------------")



import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
X=[
 [82,  67,    92,    82,    71,    92,    82,    87,    69,    61,    98,    92,],
 [66,  65,    90,    82,    72,    92,    82,    78,    70,    63,    95,    90],
 [67,  63,    88,    62,    73,    72,    82,    87,    71,    65,    92,    88],
 [82,  61,    86,    82,    74,    92,    82,    87,    72,    67,    89,    86],
 [82,  77,    84,    82,    75,    82,    82,    79,    77,    69,    86,    84],
 [62,  68,    82,    82,    76,    72,    82,    87,    78,    71,    83,    86],
 [82,  59,    80,    82,    77,    92,    82,    87,    79,    73,    80,    85],
 [72,  50,    78,    82,    78,    68,    82,    87,    80,    75,    77,    84],
 [82,  41,    76,    82,    79,    92,    82,    87,    81,    77,    74,    83]
 ]

Y=['C1','C2', 'C3','C4','C4','C1','C2','C4','C4']

prediction=[87, 77, 84, 77, 71, 82, 62, 47, 89, 71, 98, 92]

DF_X=pd.DataFrame(X, columns = ['Sub1', 'Sub2', 'Sub3', 'Sub4',    'Sub5',    'Sub6',    'Sub7',    'Sub8',    'Sub9',    'Sub10', 'Sub11', 'Sub12'])

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(DF_X, Y)
y_pred = knn.predict([prediction])
print(y_pred)

print("------------------- K-NN lassification With Normallization----------------------")
scaler = MinMaxScaler()
scaler.fit(X)
NEW_X=scaler.transform(X)
predict=scaler.transform([[87, 77, 84, 77, 71, 82, 62, 47, 89, 71, 98, 92]])
knn1 = KNeighborsClassifier(n_neighbors=3)
knn1.fit(DF_X, Y)
y_pred = knn.predict(predict)
probs1= knn.predict_proba(predict)
print(y_pred)

Decision_Matrix={'Decision-Tree':probs, 'K-NN':probs1}
print(Decision_Matrix)
