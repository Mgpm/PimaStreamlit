from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix,precision_score,recall_score,accuracy_score
import joblib

class grichSearch():
    def __init__(self):
        pass

    def gridsearch(self, data_rr, X_train,y_train,models):
        i = data_rr['accuracy'].values.argmax()
        modelStr = data_rr['model'].values[i]
        modelName = [i for i in models if i[0] == modelStr]
        model = GridSearchCV(modelName[0][1], {}, cv=5)
        model.fit(X_train,y_train)
        self.model_final = model.best_estimator_
        joblib.dump(self.model_final, 'model.pkl')

    def accuracyScoreModel(self,X_test,y_test):
        pred = self.model_final.predict(X_test)
        return accuracy_score(pred,y_test)


    def prediction(self, arr):
        model = joblib.load('model.pkl')
        pred = model.predict([arr])
        return pred[0]




