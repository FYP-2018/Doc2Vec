from gensim.models import Doc2Vec
from sklearn.cluster import Birch
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
        # print(num)
        # print(model.docvecs[num])
        docvecs.append(np.array(model.docvecs[num]))

    for branching in Parameter.branching_factor:
        silhouette_scores = []
        calinski_scores = []

        for thres in Parameter.threshold:
            Birch_model = Birch(branching_factor=branching, n_clusters=None, threshold=thres, compute_labels=True).fit(docvecs)
            labels = Birch_model.labels_

            silhouette_scores.append(metrics.silhouette_score(docvecs, labels))
            calinski_scores.append(metrics.calinski_harabaz_score(docvecs, labels))

        plt.subplot(1, 2, 1)
        plt.plot(Parameter.threshold, silhouette_scores, label=str(branching))
        plt.legend()
        plt.title("silhouette_scores")

        plt.subplot(1, 2, 2)
        plt.plot(Parameter.threshold, calinski_scores, label=str(branching))
        plt.legend()
        plt.title("calinski_scores")

    plt.savefig("result/Birch/Birch_%s.png" % i)
