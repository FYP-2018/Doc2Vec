from gensim.models import Doc2Vec
from pyclustering.cluster.cure import cure
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

    for Rpoint in Parameter.represent_point:
        silhouette_scores = []
        calinski_scores = []

        for index in Parameter.K:
            cure_model = cure(docvecs, index, number_represent_points=Rpoint)
            cure_model.process()
            clusters = cure_model.get_clusters()
            labels = [1] * len(docvecs)
            for ind in range(len(clusters)):
                for element in clusters[ind]:
                    labels[element] = ind


            print("Performance with threshold %d:" % i)
            silhouette_scores.append(metrics.silhouette_score(docvecs, labels))
            calinski_scores.append(metrics.calinski_harabaz_score(docvecs, labels))

        plt.subplot(1, 2, 1)
        plt.plot(Parameter.K, silhouette_scores, label=str(Rpoint))
        plt.legend()
        plt.title("silhouette_scores")

        plt.subplot(1, 2, 2)
        plt.plot(Parameter.K, calinski_scores, label=str(Rpoint))
        plt.legend()
        plt.title("calinski_scores")

    plt.savefig("result/Cure/Cure_%s.png" % i)
