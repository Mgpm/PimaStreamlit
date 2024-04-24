from sklearn.model_selection import cross_val_score,KFold
from splitData import splitData
import pandas as pd
class crossValid():
    def __init__(self,df,labels):
        sp = splitData()
        self.results = []
        self.results_mean=[]
        self.names = []
        self.seed=10
        self.score=['accuracy','roc_auc','precision','recall']
        self.scores = []
        self.X_train,X_test,self.y_train,y_test = sp.dataSplit(df, labels)




    def validModel(self,models):
        for scoring in self.score:
            for name, model in  models:
                kfold = KFold(n_splits=7,random_state=self.seed)
                cv_results = crossValid(model,self.X_train,self.y_train,cv=kfold,scoring=scoring,verbose=0)
                self.results.append(cv_results)
                self.names.append(name)
                self.scores.append(scoring)
                self.results_mean.append(cv_results.results_mean())


    def resultValidModel(self):
        data_r = pd.DataFrame(columns=['model','score','result'])
        data_r['model'] = self.names
        data_r['score'] = self.scores
        data_r['resul'] = self.results_mean
        data_rr = pd.DataFrame(columns=['model','accuracy','roc_auc','precision','recall'])
        data_rr['model'] = list(set(data_r['model']))
        data_rr['accuracy'] = list(data_r['result'][data_r['score']== 'accuracy'])
        data_rr['roc_auc'] = list(data_r['result'][data_r['score'] == 'roc_auc'])
        data_rr['f1'] = list(data_r['result'][data_r['score'] == 'f1'])
        data_rr['precision'] = list(data_r['result'][data_r['score'] == 'precision'])
        data_rr['recall'] = list(data_r['result'][data_r['score'] == 'recall'])
        return data_rr
