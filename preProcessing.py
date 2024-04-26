from typing import *
import pandas as pd
class preProcessing():
    def __init__(self):
        pass

    def load(self) -> None:
        self.df = pd.read_csv('diabete.csv')

    def getDF(self) -> pd.DataFrame:
        return self.df

    def shapeDF(self) -> tuple[int,int]:
        return self.df.shape

    def headDF(self) -> pd.DataFrame:
        return self.df.head()

    def infoDF(self) -> None:
        return self.df.info()

    def getDataWithoutLabels(self) -> pd.DataFrame:
        return self.df[self.df.columns[:-1]]

    def getLabels(self) -> pd.DataFrame:
        return self.df[self.df.columns[-1]]

    def getLabelSeries(self) -> pd.Series:
        return self.df[self.df.columns[-1]].value_counts()

    def describeDF(self) -> pd.DataFrame:
        return self.getDataWithoutLabels().describe()

    def corrDF(self) -> pd.DataFrame:
        return self.getDataWithoutLabels().corr()



