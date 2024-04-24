
from sklearn.decomposition import PCA

class decomposition():

    def __init__(self):
        pass

    def Pca(self, df):
        pca = PCA(n_components=2)
        pcaM=pca.fit(df)
        self.Variances = pcaM.explained_variance_ratio_
        return pcaM.transform(df)

