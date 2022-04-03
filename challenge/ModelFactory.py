from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


class ModelFactory:

    def __init__(self):
        pass

    def createModel(self, kind, data=None):
        if kind == "SVM":
            kernel_kind, gamma = data
            return SVC(kernel=kernel_kind, gamma=gamma, cache_size=7000)
        elif kind == "LogisticReg":
            return LogisticRegression()
        elif kind == "KNearest":
            return KNeighborsClassifier(n_neighbors=data)
        elif kind == "DecTree":
            max_depth, min_samples_leaf = data
            return DecisionTreeClassifier(criterion="gini", random_state=42, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        elif kind == "RandForest":
            return RandomForestClassifier(*data)
        elif kind == "MLP":
            return MLPClassifier(hidden_layer_sizes=data, max_iter=500)
        elif kind == "LinearReg":
            return LinearRegression()
        elif kind == "RandForestReg":
            return RandomForestRegressor(n_estimators=500, random_state=0)
        else:
            return None
