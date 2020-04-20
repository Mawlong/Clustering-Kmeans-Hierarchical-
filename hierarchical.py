from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

def singleLinkage(df,k=2):
    print("\nHierarchical Clustering using single linkage\n")
    clustering = AgglomerativeClustering(n_clusters = k, linkage='single' ).fit(df)
    score_avg = silhouette_score(df, clustering.labels_)
    clusteringFinal = clustering.fit_predict(df)
    df['cluster'] = clusteringFinal
    singleLinkageCSV = df.copy()
    singleLinkageCSV.to_csv(".\generatedCSV\singleLinkageCSV.csv", index = False, header=True)
    return(score_avg)

def completeLinkage(df,k=2):
    print("\nHierarchical Clustering using complete linkage\n")
    clustering = AgglomerativeClustering(n_clusters=k, linkage='complete').fit(df)
    score_avg = silhouette_score(df, clustering.labels_)
    clusteringFinal = clustering.fit_predict(df)
    df['cluster'] = clusteringFinal
    completeLinkageCSV = df.copy()
    completeLinkageCSV.to_csv(".\generatedCSV\completeLinkageCSV.csv", index=False, header=True)
    return (score_avg)

def averageLinkage(df,k=2):
    print("\nHierarchical Clustering using average linkage\n")
    clustering = AgglomerativeClustering(n_clusters=k, linkage='average').fit(df)
    score_avg = silhouette_score(df, clustering.labels_)
    clusteringFinal = clustering.fit_predict(df)
    df['cluster'] = clusteringFinal
    averageLinkageCSV = df.copy()
    averageLinkageCSV.to_csv(".\generatedCSV\averageLinkageCSV.csv", index=False, header=True)
    return (score_avg)