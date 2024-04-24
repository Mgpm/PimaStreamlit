from sklearn.model_selection import train_test_split

class splitData():

    def __init__(self):
        pass

    def dataSplit(self,df, labels):
        return train_test_split(df,labels, test_size=0.2)