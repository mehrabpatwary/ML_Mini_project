import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
from pydotplus import graph_from_dot_data


training_data = pd.read_csv("play.csv")
X = training_data.iloc[:, :-1]
# print(X)

y = training_data.iloc[:, 4]
# print(y)
label_encoder_X = LabelEncoder()

X = X.apply(LabelEncoder().fit_transform)
print(X)

regressor = DecisionTreeClassifier()
print(regressor.fit(X, y))

X_in = np.array([0, 1, 1, 0])

y_pred = regressor.predict([X_in])

print(y_pred)

dot_data = StringIO()
export_graphviz(regressor, out_file=dot_data, filled=True, rounded=True, special_characters=True)
graph = graph_from_dot_data(dot_data.getvalue())
graph.write_png('tree.png')
