from sklearn.preprocessing import StandardScaler, MinMaxScaler

class normalize():

    def __init__(self):
        pass

    def MinMaxSc(self,df):
        return MinMaxScaler(feature_range=(-1,1)).fit_transform(df)

    def StandSc(self,df):
        return StandardScaler().fit_transform(df)



