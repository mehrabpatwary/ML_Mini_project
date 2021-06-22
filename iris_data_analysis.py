# Check the version of libraries
# Python version
import matplotlib
import numpy
import pandas
import scipy
import sklearn
import sys
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

print('Python: {}'.format(sys.version))
print('Scipy: {}'.format(scipy.__version__))
print('Numpy: {}'.format(numpy.__version__))
print('Matplotlib: {}'.format(matplotlib.__version__))
print('Pandas: {}'.format(pandas.__version__))
print('Sklearn: {}'.format(sklearn.__version__))

url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)
print(dataset.shape)
print(dataset.head(30))
print(dataset.describe())
print(dataset.groupby('class').size())

dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
plt.show()

dataset.hist()
plt.show()

scatter_matrix(dataset)
plt.show()

array = dataset.values
X = array[:, 0:4]
Y = array[:, 4]
validation_size = 0.20
seed = 1

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=validation_size, random_state=seed)

scoring = 'accuracy'

# Spot check Algorithms
models = [('LR', LogisticRegression(solver="liblinear", multi_class='ovr')), ('LDA', LinearDiscriminantAnalysis()),
          ("KNN", KNeighborsClassifier()), ("CART", DecisionTreeClassifier()), ('NB', GaussianNB()), ("SVM", SVC(gamma='auto'))]
# Evaluate each model in turn
results = []
names = []
seed = 1
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=seed, shuffle=True)
    cv_result = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_result)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_result.mean(), cv_result.std())
    print(msg)

