import pandas as pd
class vizualisation():
    def __init__(self):
        pass

    def plotDF(self,df,c1,c2):
        return df.plot(x=c1,y=c2)

    def scatterDF(self,df,c1,c2):
        return df.plot(kind='scatter',x=c1, y=c2)

    def barDF(self,df,c1,c2):
        return df.plot(kind='bar',x=c1,y=c2)

    def histData(self,df):
        return df.hist(column=df.columns[:-1],figsize=(12,12))

    def visualizeData(self,arr):
        df = pd.DataFrame(data=arr,columns=['x','y'])
        return  df.plot(kind='scatter',x=df.columns[0],y=df.columns[1])


