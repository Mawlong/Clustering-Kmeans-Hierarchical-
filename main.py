import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import hierarchical as h
import about as about
import kmeans as km


def plotter(plus, rand):
      values = [plus,rand]
      label = ['K-means++', 'Random']
      plt.bar(label,values)
      plt.title("Comparison of Silhouette Analysis")
      plt.draw()
      plt.pause(0.001)
      plt.savefig('./figurePlots/randVsKMeansPP.png')


def plotterAgg(o1,o2,o3):
      values = [o1,o2,o3]
      label = ['Single', 'Complete', 'Group Average']
      plt.bar(label,values)
      plt.title("Comparison of Silhouette Analysis")
      plt.draw()
      plt.pause(0.001)
      plt.savefig('./figurePlots/AgglomerativeHierarchical.png')


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
                              km.elbow(df_scaled)

                        print("\n\n Which method of cluster initialisation to view?"
                              "\n\t1. k-means++"
                              "\n\t2. random"
                              "\n\t3. exit")
                        kchoice = int(input().strip())

                        print("\n\nEnter the number of clusters: ")
                        clusters = int(input().strip())

                        if(kchoice == 1):
                              kplusplus = km.kmpp(df_scaled,clusters)
                              print("\n\nWould you like to see for Random as well? (yes = 1, no = 0)")
                              secondChoice = int(input().strip())
                              if(secondChoice == 1):
                                    krandom = km.kmrand(df_scaled,clusters)
                                    plotter(kplusplus,krandom)

                        elif(kchoice == 2):
                              krandom = km.kmrand(df_scaled,clusters)
                              print("Would you like to see for k-means++ as well? (yes = 1, no = 0)")
                              secondChoice = int(input().strip())
                              if (secondChoice == 1):
                                    kplusplus = km.kmpp(df_scaled, clusters)
                                    plotter(kplusplus, krandom)

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
                              output1 = h.singleLinkage(df_scaled,clusters)
                              print("silhouette_score: ", output1)
                              print("Would you like to see for Complete and Average as well? (yes = 1, no = 0)")
                              secondChoice = int(input().strip())
                              if (secondChoice == 1):
                                    output2 = h.completeLinkage(df_scaled,clusters)
                                    print("silhouette_score: ", output2)
                                    output3 = h.averageLinkage(df_scaled, clusters)
                                    print("silhouette_score: ", output3)
                                    plotterAgg(output1,output2,output3)

                        elif(kchoice == 2):
                              output2 = h.completeLinkage(df_scaled,clusters)
                              print("silhouette_score: ", output2)
                              print("Would you like to see for Single and Average as well? (yes = 1, no = 0)")
                              secondChoice = int(input().strip())
                              if (secondChoice == 1):
                                    output1 = h.singleLinkage(df_scaled, clusters)
                                    print("silhouette_score: ", output1)
                                    output3 = h.averageLinkage(df_scaled, clusters)
                                    print("silhouette_score: ", output3)
                                    plotterAgg(output1, output2, output3)


                        elif(kchoice == 3):
                              output3 = h.averageLinkage(df_scaled,clusters)
                              print("silhouette_score: ", output3)
                              print("Would you like to see for Complete and Single as well? (yes = 1, no = 0)")
                              secondChoice = int(input().strip())
                              if (secondChoice == 1):
                                    output2 = h.completeLinkage(df_scaled, clusters)
                                    print("silhouette_score: ", output2)
                                    output1 = h.singleLinkage(df_scaled, clusters)
                                    print("silhouette_score: ", output1)
                                    plotterAgg(output1, output2, output3)

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
      df = pd.read_csv("./dataset/dataset.csv")

      #scalling the Dataset
      scaler = MinMaxScaler()
      scaled_data = scaler.fit_transform(df)
      df_scaled = pd.DataFrame(data = scaled_data,columns = ['STG','SCG','STR','LPR','PEG'])
      print("***********************************************************************************")
      print("The scaled data-set")
      print(df_scaled)
      clusterMenu()