from gensim.models import Doc2Vec
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from parameter import Parameter

for i in range(0, 20):
    model = Doc2Vec.load('models/%s.d2v' % i)

    # convert sequence to array
    docvecs = []
    for num in range(len(model.docvecs)):
        docvecs.append(np.array(model.docvecs[num]))

    silhouette_scores = []
    calinski_scores = []
    for index in Parameter.K:
        kmeans_model = KMeans(n_clusters=index, random_state=1).fit(docvecs)
        labels = kmeans_model.labels_

        silhouette_scores.append(metrics.silhouette_score(docvecs, labels))
        calinski_scores.append(metrics.calinski_harabaz_score(docvecs, labels))

    plt.subplot(1, 2, 1)
    plt.plot(Parameter.K, silhouette_scores, label=i)
    plt.legend()
    plt.title("silhouette_scores")

    plt.subplot(1, 2, 2)
    plt.plot(Parameter.K, calinski_scores, label=i)
    plt.legend()
    plt.title("calinski_scores")

plt.savefig("result/Kmeans/Kmeans.png")
