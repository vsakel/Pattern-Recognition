import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, silhouette_score
from sklearn.decomposition import PCA


data = pd.read_csv("songs.csv")
#print(data.info()) 
data = data.dropna()
data = data.drop_duplicates()
# check for imbalanced data
# print("instances of class 1: "+str(len(data.loc[(data.Genre == 'Pop')])))
# print("instances of class 0: "+str(len(data.loc[(data.Genre == 'Rock')]))) 

# tha xwrisw ta dedomena sta kathgorika kai ta numerical wste na ta kanw preproccesing
# tha kanw one hot encoding sta categorical
# kai ta numerical tha ta kanw standarize, gia na ta xrhsimopoihsw se classifiers pou douleuoun me apostaseis shmeiwn

# ta numerical tha ta xrhsimopoihsw kai meta sto clustering
numerical = ["Duration", "Year", "Danceability", "Intensity", "Loudness", "Acousticness", "Instrumentalness", "Liveness", "Tempo"]
categorical = ["Song","ExplicitContent", "Popularity", "SpokenWords"]


y = data.Genre # target class
X = data.drop(columns=["Genre"]) 

# train - test split so we can evaluate our models/ check generalization skill
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

cat_train = X_train.loc[:,categorical]
num_train = X_train.loc[:,numerical]
cat_test = X_test.loc[:,categorical]
num_test = X_test.loc[:,numerical]

scaler = StandardScaler()

encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
cat_train = encoder.fit_transform(cat_train)

# kanw one hot encoding kai sta categorical tou test set xrhsimopoiwntas ton encoder tou train set gia na mhn exw data leakage

cat_test= encoder.transform(cat_test)

# concatenate numerical and categorical attributes and make X_train, X_test
X_train = np.concatenate([num_train, cat_train], axis=1)
X_test = np.concatenate([num_test, cat_test], axis=1)


# # X_train and X_test with scaled numerical for algorithms that handle distances of points
# scaled_Xtrain = np.concatenate([scaler.fit_transform(num_train), cat_train], axis=1)
# scaled_Xtest = np.concatenate([scaler.transform(num_test), cat_test], axis=1)


from sklearn.tree import DecisionTreeClassifier

# clf = DecisionTreeClassifier(max_depth=3)
# # # bazw parametro max_depth giati ta dentra otan ta afhnoume xwris periorismo 
# # # kanoun eukola overfit, sto sygkekrimeno petyxainoume au3hsh twn metrikwn an baloume max_depth=3 anti gia None
# clf = clf.fit(X_train,y_train)
# pred = clf.predict(X_test)
# print(accuracy_score(y_test,pred))
# print(f1_score(y_test,pred,average='weighted')) # pio swsto tha htan na parw to f1 weighted 

# ## accuracy: 0.8116343490304709
# ## f1_score: 0.8940809968847352

# ##sth synexeia dokimazw mpl classifier alla epeidh douleuei me apostaseis tha kanw kai ena scale ta dedomena (standarize)

# # from sklearn.neural_network import MLPClassifier

# clf = MLPClassifier(hidden_layer_sizes=(20,20,20),max_iter=10000)
# clf = clf.fit(scaled_Xtrain,y_train)
# pred = clf.predict(scaled_Xtest)
# print(accuracy_score(y_test,pred))
# print(f1_score(y_test,pred,pos_label='Pop'))

# # # accuracy:0.7839335180055401
# # # f1_score:0.8729641693811074


# # Let's try SVM classifier kai edw xrhsimopoiw ta scaled dedomena giati douleuei me apostaseis
# from sklearn.svm import SVC

# accuracy_svm =[]
# f1_svm = []

# gammas = [0.1,1,10]
# for gamma in gammas:
#     clf = SVC(kernel="rbf",gamma=gamma)
#     clf = clf.fit(scaled_Xtrain,y_train)
#     pred = clf.predict(scaled_Xtest)
#     accuracy_svm.append(accuracy_score(y_test,pred))
#     f1_svm.append(f1_score(y_test,pred,pos_label='Pop'))

# # print("to kalytero modelo einai gia gamma =",gammas[np.argmax(accuracy_svm)])
# # print(accuracy_svm)
# # print(f1_svm)

# ## gia gamma = 1 pernoume ta kalytera apotelesmata me 
# # # accuracy: 0.8227146814404432
# # # f1_score: 0.9027355623100305


# # dokimazw kai knn classifer

# from sklearn.neighbors import KNeighborsClassifier
# clf = KNeighborsClassifier(n_neighbors=10)
# clf = clf.fit(scaled_Xtrain,y_train)
# pred = clf.predict(scaled_Xtest)
# accuracy_knn = accuracy_score(y_test,pred)
# f1_knn = f1_score(y_test,pred,pos_label='Pop')
# print(accuracy_knn)
# print(f1_knn)

# # # accuracy: 0.8116343490304709
# # # f1_score: 0.8940809968847352


## gia to clustering, thelw na omadopoihsw ta tragoudia pou emfanizoun koina xarakthristika
# tha xrhsimopoihsw mono ta synexh dedomena mia kai oi algorithmoi pou tha xrhsimopoihsw douleuoun me apostaseis shmeiwn
# ara einai anagkh kai pali na kanw standarize ta dedomena

X = data.loc[:,numerical]
Song = data.Song # song name

scaler = StandardScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X,columns=numerical)


# we do this to see how dimension reduction impacts on information loss
# to na paroume tis PC pou sygkentrwnoun thn perissoterh plhroforia den shmainei anagkastika oti tha kanoume kalytero clustering
# pantws se algorithmous me polles diastaseis, sigoura bohthaei sthn polylokothta

# pca = PCA()
# pca_data = pca.fit_transform(X)
# eigenvalues = pca.explained_variance_
# eigenvectors = pca.components_
# print("eigen values are :",eigenvalues)
# print("eigenvectors are: ",eigenvectors)
# # grafhma pou deixnei thn plhroforia pou exei h kathe synistwsa PC 
# plt.bar(range((len(eigenvalues))),eigenvalues/sum(eigenvalues))
# plt.show()
# print("info loss by taking 5 PC",sum(eigenvalues[5:])/sum(eigenvalues))

# tha dokimasw kmeans me diaforetiko ari8mo kentrwn/omadwn 
# kai tha epile3w ws kalytero ari8mo me bash to elbow method

from sklearn.cluster import KMeans, AgglomerativeClustering

# sse = []
# for i in range(1, 11):
#     kmeans = KMeans(n_clusters=i, init="k-means++").fit(X)
#     sse.append(kmeans.inertia_)
# plt.plot(range(1, 11), sse)
# plt.scatter(range(1, 11), sse, marker="o")
# plt.show()

# h epilogh twn kentrwn ephreazei ton kmeans, giauto xrhsimopoioume ton e3ypno tropo epiloghs twn kentrwn pou diathetei
# h ylopoihsh ths sklearn
# epilegoume n=6 clusters

# kmeans = KMeans(n_clusters=6,init='k-means++').fit(X)
# print(silhouette_score(X, kmeans.labels_))

# from yellowbrick.cluster import SilhouetteVisualizer
# visualizer = SilhouetteVisualizer(kmeans, colors='yellowbrick')
# visualizer.fit(X)
# visualizer.show() 

# # blepoume wstoso poly xamhlh timh tou silhouette gia th synolikh omadopoihsh
# # epishs blepoume ta silhouette twn omadwn na einai anomoiomorfa kai arketa dedomena na exoun arnhtiko silhouette pou shmainei 
# # oti omadopoiountai lathos

# # # gia na optikopoihsoume to apotelesma tou clustering sto xwro kanoume PCA kai pame sto subspace R^3
# pca = PCA(n_components=3)
# pca_data = pca.fit_transform(X)
# pca_data = pd.DataFrame(pca_data)
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(pca_data.loc[:,0],pca_data.loc[:,1],pca_data.loc[:,2],c=kmeans.labels_,cmap='Accent')
# plt.show()

#blepoume ti paei strava (xontrika giati exw kanei pca)
#ta dedomena einai arketa pykna kai den exoun sfairiko sxhma gia na doulepsi o kmeans



# dokimazw Hierarchical Clustering me linkage = "average"
from sklearn.cluster import AgglomerativeClustering

# slc = []
# for i in range(2, 21):
#     clustering = AgglomerativeClustering(n_clusters=i, linkage="average").fit(X)
#     slc.append(silhouette_score(X, clustering.labels_))

# plt.plot(range(2, 21), slc)
# plt.xticks(range(2, 21), range(2, 21))
# plt.title('Hierarchical clustering')
# plt.xlabel('Number of clusters')
# plt.ylabel('Silhouette score')
# plt.show()

# me bash th metrikh tou silhouette tha parw kai tha kopsw 3 clusters

clustering = AgglomerativeClustering(n_clusters=3, linkage="average").fit(X)
print(silhouette_score(X, clustering.labels_))

# ## gia na optikopoihsoume to apotelesma tou clustering sto xwro kanoume PCA kai pame sto subspace R^3
# pca = PCA(n_components=3)
# pca_data = pca.fit_transform(X)
# pca_data = pd.DataFrame(pca_data)
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(pca_data.loc[:,0],pca_data.loc[:,1],pca_data.loc[:,2],c=clustering.labels_,cmap='Accent')
# plt.show()

# # final dataframe

# check = pd.DataFrame({"Songs":Song, "labels":clustering.labels_})
# print(check.to_string())

# edw blepoume oti ousiastika ta 3 clusters einai 1 megalo kai poly pukno kai ta ypoloipa 2 einai poly mikra
# ta alla dyo mporei na einai kai thorybos kathws o hierarchical clustering einai arketa euais8htos