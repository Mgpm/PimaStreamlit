from sklearn.model_selection import cross_val_score,KFold
import pandas as pd

class crossValid():
    def __init__(self,X_train,y_train):
        
        self.results = []
        self.results_mean=[]
        self.names = []
        self.score=['accuracy','roc_auc','f1','precision','recall']
        self.scores = []
        self.X_train = X_train
        self.y_train = y_train
        


    def validModel(self,models):
        for scoring in self.score:
            for name, model in  models:
                kfold = KFold(n_splits=7,shuffle=True)
                cv_results = cross_val_score(model,self.X_train,self.y_train.values,cv=kfold,scoring=scoring,verbose=0)
                self.results.append(cv_results)
                self.names.append(name)
                self.scores.append(scoring)
                self.results_mean.append(cv_results.mean())


    def resultValidModel(self):
        self.data_r = pd.DataFrame(columns=['model','score','result'])
        self.data_r['model'] = self.names
        self.data_r['score'] = self.scores
        self.data_r['result'] = self.results_mean
        self.data_rr = pd.DataFrame(columns=['model','accuracy','roc_auc','f1','precision','recall'])
        self.data_rr['model'] = list(set(self.data_r['model']))
        self.data_rr['accuracy'] = list(self.data_r['result'][self.data_r['score']== 'accuracy'])
        self.data_rr['roc_auc'] = list(self.data_r['result'][self.data_r['score'] == 'roc_auc'])
        self.data_rr['f1'] = list(self.data_r['result'][self.data_r['score'] == 'f1'])
        self.data_rr['precision'] = list(self.data_r['result'][self.data_r['score'] == 'precision'])
        self.data_rr['recall'] = list(self.data_r['result'][self.data_r['score'] == 'recall'])
       
