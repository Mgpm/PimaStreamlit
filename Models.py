from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

class Models():
    def __init__(self):
        pass

    def getModels(self):
        models = []
        models.append(('LogisticRegression',LogisticRegression()))
        models.append(('LinearDiscriminantAnalysis',LinearDiscriminantAnalysis()))
        models.append(('KNeighborsClassifier',KNeighborsClassifier()))
        models.append(('DecisionTreeClassifier',DecisionTreeClassifier()))
        models.append(('GaussianNB',GaussianNB()))
        models.append(('SVM',SVC()))
        models.append(('MLPClassifier',MLPClassifier()))
        models.append(('RandomForestClassifier',RandomForestClassifier()))
        return models