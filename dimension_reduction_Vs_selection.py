# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 14:18:03 2020

@author: Dio VB
"""

def plotting(X_transformed,y,f,plt):
    if f=="dr":
        fdr = plt.figure()
        ax1 = fdr.add_subplot(111)
        plt.title("Dimensionality Reduction(ISOMAP) 2 Dim")

    elif f=="fs":
        ffs = plt.figure()
        ax2 = ffs.add_subplot(111)
        plt.title("Feature Selection 2 Feature")
  
    plt.scatter(X_transformed[:, 0], X_transformed[:, 1], lw=0.1,
       c=y, cmap=plt.cm.get_cmap('Spectral', 11))
   
    plt.colorbar(ticks=range(11), label='digit value')
    
def plotting_original(plt,digits):
    fig, ax = plt.subplots(8, 8, figsize=(10, 10))
    for i, axi in enumerate(ax.flat):
        axi.imshow(digits.images[i], cmap='binary')
        axi.set(xticks=[], yticks=[])
            
def dimentional_reduction(X):
    from sklearn.manifold import Isomap
    dim_red = Isomap(n_components=2)
    X_transformed = dim_red.fit_transform(X)
    return X_transformed

def feature_selection(X,y):
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
    return X_new,y
    
if __name__ == '__main__':
    import seaborn as sns; sns.set()
    from sklearn.datasets import load_digits
    import matplotlib.pyplot as plt
    digits = load_digits()
    X=digits.data
    y=digits.target
    plotting_original(plt,digits);
    X_rd=dimentional_reduction(X);
    plotting(X_rd,y,"dr",plt);
    X_fs,y_fs=feature_selection(X,y);
    plotting(X_fs,y_fs,"fs",plt);
