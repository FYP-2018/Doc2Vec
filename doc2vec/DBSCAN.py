from gensim.models import Doc2Vec
from sklearn.cluster import DBSCAN
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from parameter import Parameter

for i in range(0, 20):
    model = Doc2Vec.load('models/%s.d2v' % i)
    plt.figure()

    # convert sequence to array
    docvecs = []
    for num in range(len(model.docvecs)):
        docvecs.append(np.array(model.docvecs[num]))
    for _min in Parameter.min_samples:
        silhouette_scores = []
        calinski_scores = []

        for _eps in Parameter.eps:
            dbscan_model = DBSCAN(eps=_eps, min_samples=_min).fit(docvecs)
            labels = dbscan_model.labels_

            silhouette_scores.append(metrics.silhouette_score(docvecs, labels))
            calinski_scores.append(metrics.calinski_harabaz_score(docvecs, labels))

        plt.subplot(1, 2, 1)
        plt.plot(Parameter.eps, silhouette_scores, label=_min)
        plt.legend()
        plt.title("silhouette_scores")

        plt.subplot(1, 2, 2)
        plt.plot(Parameter.eps, calinski_scores, label=_min)
        plt.legend()
        plt.title("calinski_scores")

    plt.savefig("result/DBSCAN/DBSCAN_%s.png" % i)
