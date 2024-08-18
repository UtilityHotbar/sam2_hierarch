import pickle
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

import matplotlib.pyplot as plt
import numpy as np

def cluster_embeds(images: dict):
    embeds = []
    names = []
    for name in images:
        embeds.append(np.array(images[name]))
        names.append(name)
    ld = linkage(embeds, method='single', metric='euclidean', optimal_ordering=False)
    x = fcluster(ld, t=0.6, criterion='distance')
    c_n_ls = {}
    for i in range(len(x)):
        if x[i] not in c_n_ls:
            c_n_ls[x[i]] = []
        c_n_ls[x[i]].append(names[i])
    for cluster in c_n_ls:
        print(cluster)
        print(','.join(c_n_ls[cluster]))

def main():
    print('loading images')
    with open('img_embeds.pickle', 'rb') as handle:
        images = pickle.load(handle)
    print('loaded')
    cluster_embeds(images)
    

if __name__ == '__main__':
    main()