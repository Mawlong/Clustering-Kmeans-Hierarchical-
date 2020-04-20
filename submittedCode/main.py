import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import about as about

def kmpp(df,k=2):
    km = KMeans(n_clusters=k, init='k-means++')
    predicted = km.fit_predict(df)
    df['cluster'] = predicted
    kmppCSV = df.copy()
    kmppCSV.to_csv(".\generatedCSV\kmppCSV.csv", index=False, header=True)
    labelsR = km.labels_
    print("K-means++ initialisation Silhouette Analysis: ", metrics.silhouette_score(df, labelsR, metric='euclidean'))
    return(metrics.silhouette_score(df, labelsR, metric='euclidean'))

def kmrand(df,k=2):
    km = KMeans(n_clusters=k, init='random')
    predicted = km.fit_predict(df)
    df['cluster'] = predicted
    kmRandCSV = df.copy()
    kmRandCSV.to_csv(".\generatedCSV\kmRandCSV.csv", index=False, header=True)
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
    plt.savefig('elbow.png')

    print("\n\n***********************************************************************************")

def singleLinkage(df,k=2):
    print("\nHierarchical Clustering using single linkage\n")
    clustering = AgglomerativeClustering(n_clusters = k, linkage='single' ).fit(df)
    score_avg = silhouette_score(df, clustering.labels_)
    clusteringFinal = clustering.fit_predict(df)
    df['cluster'] = clusteringFinal
    print(df.head())
    return(score_avg)

def completeLinkage(df,k=2):
    print("\nHierarchical Clustering using complete linkage\n")
    clustering = AgglomerativeClustering(n_clusters=k, linkage='complete').fit(df)
    score_avg = silhouette_score(df, clustering.labels_)
    clusteringFinal = clustering.fit_predict(df)
    df['cluster'] = clusteringFinal
    print(df.head())
    return (score_avg)

def averageLinkage(df,k=2):
    print("\nHierarchical Clustering using average linkage\n")
    clustering = AgglomerativeClustering(n_clusters=k, linkage='average').fit(df)
    score_avg = silhouette_score(df, clustering.labels_)
    clusteringFinal = clustering.fit_predict(df)
    df['cluster'] = clusteringFinal
    print(df.head())
    return (score_avg)

def clusterMenu():
      check = int(1)

      while(check):

            print("Select from the following (type 1 or 2):\n"
                  "\t1. K-means clustering"
                  "\t2. Agglomerative hierarchical clustering"
                  "\t3. exit")

            c = int(input().strip())


            if (c == 1):
                  loop = int(1)

                  while(loop):
                        print("Would you like to view the dataset using the elbow method to determine the number of clusters? (yes = 1, no = 0)")
                        kchoice = int(input().strip())
                        if(kchoice == 1):
                              elbow(df_scaled)

                        print("\n\n Which method of cluster initialisation to view?"
                              "\n\t1. k-means++"
                              "\n\t2. random"
                              "\n\t3. exit")
                        kchoice = int(input().strip())

                        print("\n\nEnter the number of clusters: ")
                        clusters = int(input().strip())

                        if(kchoice == 1):
                              kplusplus = kmpp(df_scaled,clusters)
                              print("\n\nWould you like to see for Random as well? (yes = 1, no = 0)")
                              secondChoice = int(input().strip())
                              if(secondChoice == 1):
                                    krandom = kmrand(df_scaled,clusters)
                                    

                        elif(kchoice == 2):
                              krandom = kmrand(df_scaled,clusters)
                              print("Would you like to see for k-means++ as well? (yes = 1, no = 0)")
                              secondChoice = int(input().strip())
                              if (secondChoice == 1):
                                    kplusplus = kmpp(df_scaled, clusters)
                                    

                        elif(kchoice ==3):
                              break
                        else:
                             print("wrong choice")

                        print("\n\nWould you like to check for kmeans again? (yes = 1, no = 0)? ")
                        loop  = int(input().strip())

            elif(c == 2):

                  loop = int(1)

                  while(loop):

                        print("\n\n Which method of Agglomerative hierarchical clustering to view?"
                              "\n\t1. Single Linkage"
                              "\n\t2. Compete Linkage"
                              "\n\t3. Group Average"
                              "\n\t4. exit")

                        kchoice = int(input().strip())

                        print("Enter the number of clusters (default = 2): ")
                        clusters = int(input().strip())

                        if(kchoice == 1):
                              output1 = singleLinkage(df_scaled,clusters)
                              print("silhouette_score: ", output1)
                              print("Would you like to see for Complete and Average as well? (yes = 1, no = 0)")
                              secondChoice = int(input().strip())
                              if (secondChoice == 1):
                                    output2 = completeLinkage(df_scaled,clusters)
                                    print("silhouette_score: ", output2)
                                    output3 = averageLinkage(df_scaled, clusters)
                                    print("silhouette_score: ", output3)
                                    

                        elif(kchoice == 2):
                              output2 = completeLinkage(df_scaled,clusters)
                              print("silhouette_score: ", output2)
                              print("Would you like to see for Single and Average as well? (yes = 1, no = 0)")
                              secondChoice = int(input().strip())
                              if (secondChoice == 1):
                                    output1 = singleLinkage(df_scaled, clusters)
                                    print("silhouette_score: ", output1)
                                    output3 = averageLinkage(df_scaled, clusters)
                                    print("silhouette_score: ", output3)
                                    


                        elif(kchoice == 3):
                              output3 = averageLinkage(df_scaled,clusters)
                              print("silhouette_score: ", output3)
                              print("Would you like to see for Complete and Single as well? (yes = 1, no = 0)")
                              secondChoice = int(input().strip())
                              if (secondChoice == 1):
                                    output2 = completeLinkage(df_scaled, clusters)
                                    print("silhouette_score: ", output2)
                                    output1 = singleLinkage(df_scaled, clusters)
                                    print("silhouette_score: ", output1)
                                    

                        elif(kchoice == 4):
                              break

                        else:
                              print("Wrong choice!")

                        print("\n\nWould you like to check for Agglomerative hierarchical clustering again? (yes = 1, no = 0)? ")
                        loop = int(input().strip())

            elif (c == 3):
                   break
            else:
                  print("Wrong choice")

            print("\n\nWould you like to redo your choices? (yes = 1, no = 0)? ")
            check = int(input().strip())



if __name__ == '__main__':

      about.intoduction()

      #Reading the dataset
      df = pd.read_csv("https://dataset-ten.now.sh/dataset.csv")

      #scalling the Dataset
      scaler = MinMaxScaler()
      scaled_data = scaler.fit_transform(df)
      df_scaled = pd.DataFrame(data = scaled_data,columns = ['STG','SCG','STR','LPR','PEG'])
      print("***********************************************************************************")
      print("The scaled data-set")
      print(df_scaled)
      clusterMenu()