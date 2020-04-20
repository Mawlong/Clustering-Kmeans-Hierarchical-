from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn import metrics

def kmpp(df,k=2):
    km = KMeans(n_clusters=k, init='k-means++')
    predicted = km.fit_predict(df)
    df['cluster'] = predicted
    kmppCSV = df.copy()
    kmppCSV.to_csv("./generatedCSV/kmppCSV.csv", index=False, header=True)
    labelsR = km.labels_
    print("K-means++ initialisation Silhouette Analysis: ", metrics.silhouette_score(df, labelsR, metric='euclidean'))
    return(metrics.silhouette_score(df, labelsR, metric='euclidean'))

def kmrand(df,k=2):
    km = KMeans(n_clusters=k, init='random')
    predicted = km.fit_predict(df)
    df['cluster'] = predicted
    kmRandCSV = df.copy()
    kmRandCSV.to_csv("./generatedCSV/kmRandCSV.csv", index=False, header=True)
    labelsR = km.labels_
    print("Random initialisation Silhouette Analysis: ", metrics.silhouette_score(df, labelsR, metric='euclidean'))
    return (metrics.silhouette_score(df, labelsR, metric='euclidean'))

def elbow(df):

    print("\n\n\nElbow Method visualisation using SSE")

    sseR = []
    sse = []
    k_rng = range(1, 10)

    for k in k_rng:
        kmR = KMeans(n_clusters=k, init='random')
        km = KMeans(n_clusters=k, init='k-means++')
        kmR.fit_predict(df)
        sseR.append(kmR.inertia_)
        km.fit_predict(df)
        sse.append(km.inertia_)
        print('Cluster(Random init):\t ', (k + 1), '| Inertia(SSE):', kmR.inertia_)
        print('Cluster(k-means++ init): ', (k + 1), '| Inertia(SSE):', km.inertia_)
        print('Random is better\n' if kmR.inertia_ >= km.inertia_ else 'k-means++ is better\n')

    print("The corresponding graphs are shown in the figure generated.")

    fig, (ax1, ax2) = plt.subplots(2, sharey=True)
    ax1.plot(k_rng, sseR, 'ko-')
    ax1.set(title='Elbow Method', ylabel='Random')
    ax2.plot(k_rng, sse, 'ko-')
    ax2.set(xlabel='number of clusters K', ylabel='K-means++')
    plt.draw()
    plt.pause(0.001)
    input("Press [enter] to continue.")
    plt.savefig('./figurePlots/elbow.png')

    print("\n\n***********************************************************************************")