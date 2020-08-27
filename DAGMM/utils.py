import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import logging
import logging.handlers
from time import time
from datetime import datetime

def x_means(
        X,
        num_init_clusters=8,
        visualize=True
):
    from pyclustering.cluster.kmeans import kmeans, kmeans_visualizer
    from pyclustering.cluster.xmeans import xmeans
    from pyclustering.cluster import cluster_visualizer
    from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
    from pyclustering.cluster import cluster_visualizer_multidim

    X = list(X)

    start_centers = kmeans_plusplus_initializer(X, num_init_clusters).initialize()

    xmeans_instance = xmeans(X, start_centers, 32, ccore=True, criterion=0)

    # Run cluster analysis and obtain results.
    xmeans_instance.process()
    clusters = xmeans_instance.get_clusters()
    centers = xmeans_instance.get_centers()
    print('Number of cluster centers calculated :', len(centers))

    if visualize:
        visualizer = cluster_visualizer_multidim()
        visualizer.append_clusters(clusters, X)
        visualizer.show()
    return centers, clusters



def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)



def get_logger(LOG_FILE):

    logger = logging.getLogger('main')
    logger.setLevel(logging.INFO)
    OP_DIR = os.path.join('Logs')

    if not os.path.exists(OP_DIR):
        os.mkdir(OP_DIR)

    handler = logging.FileHandler(os.path.join(OP_DIR, LOG_FILE))
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.info('Log start :: ' + str(datetime.now()))
    return logger


def log_time(logger):
    logger.info(str(datetime.now()) + '| Time stamp ' + str(time()))


def close_logger(logger):
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)
    logging.shutdown()
    return