# Copyright 2018 Stanislav Pidhorskyi
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#  http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import print_function
import torch.utils.data
from torch import optim
from torchvision.utils import save_image
from net import *
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import json
import pickle
from utils.torch_cuda_helper import *
from utils import batch_provider
import random
from torch.autograd.gradcheck import zero_gradients
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import scipy.stats
import os
import math
from sklearn.metrics import roc_auc_score
# Importing svm
from sklearn import svm
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.externals import joblib


folding_id = -1
class_fold = -1
z_size = 32
batch_size = 64

title_size = 16
axis_title_size = 14
ticks_size = 18

power = 2.0


def process_batch(batch):
    label = [x[0] for x in batch]
    data = [x[1] for x in batch]
    y = numpy2torch(np.asarray(label)).type(LongTensor)
    x = numpy2torch(np.asarray(data, dtype=np.float32)) / 255.0
    return x, y


def compute_jacobian(inputs, output):
    """
    :param inputs: Batch X Size (e.g. Depth X Width X Height)
    :param output: Batch X Classes
    :return: jacobian: Batch X Classes X Size
    """
    assert inputs.requires_grad

    num_classes = output.size()[1]

    jacobian = torch.zeros(num_classes, *inputs.size())
    grad_output = torch.zeros(*output.size())
    if inputs.is_cuda:
        grad_output = grad_output.cuda()
        jacobian = jacobian.cuda()

    for i in range(num_classes):
        zero_gradients(inputs)
        grad_output.zero_()
        grad_output[:, i] = 1
        output.backward(grad_output, retain_graph=True)
        jacobian[i] = inputs.grad.data

    return torch.transpose(jacobian, dim0=0, dim1=1)


def r_pdf(x, bins, count):
    if x < bins[0]:
        return max(count[0], 1e-308)
    if x >= bins[-1]:
        return max(count[-1], 1e-308)
    id = np.digitize(x, bins) - 1
    return max(count[id], 1e-308)


def get_f1(precision, recall):
    if precision == 0.0 or recall == 0.0:
        return 0
    return 2.0 * precision * recall / (precision + recall)


def gpnd(data, run_gpnd=False, gennorm_param=None, bin_edges=0, counts=0):
    G = Generator(z_size).to(device)
    E = Encoder(z_size).to(device)
    setup(E)
    setup(G)
    G.eval()
    E.eval()

    G.load_state_dict(torch.load("Gmodel.pkl"))
    E.load_state_dict(torch.load("Emodel.pkl"))

    sample = torch.randn(64, z_size).to(device)
    sample = G(sample.view(-1, z_size, 1, 1)).cpu()
    save_image(sample.view(64, 1, 32, 32), 'sample.png')

    zlist = []
    labellist = []
    rlist = []
    result = []
    batches = batch_provider.batch_provider(data, batch_size, process_batch, report_progress=True)
    for x, y in batches:
        x = Variable(x.data, requires_grad=True)
        z = E(x.view(-1, 1, 32, 32))
        recon_batch = G(z)
        z = z.squeeze()
        recon_batch = recon_batch.squeeze().cpu().detach().numpy()
        xn = x.squeeze().cpu().detach().numpy()

        distances = []
        for i in range(x.shape[0]):
            distance = np.sum(np.power(recon_batch[i].flatten() - xn[i].flatten(), power))
            rlist.append(distance)
            distances.append(distance)

        zlist.append(z.cpu().detach().numpy())
        labellist.append(y)

        if run_gpnd:
            #J = compute_jacobian(x, z)

            #J = J.cpu().numpy()

            z = z.cpu().detach().numpy()

            for i in range(x.shape[0]):
                #u, s, vh = np.linalg.svd(J[i, :, :], full_matrices=False)
                #logD = np.sum(np.log(np.abs(s)))

                p = scipy.stats.gennorm.pdf(z[i], gennorm_param[0, :], gennorm_param[1, :], gennorm_param[2, :])
                logPz = np.sum(np.log(p))

                # Sometimes, due to rounding some element in p may be zero resulting in Inf in logPz
                # In this case, just assign some large negative value to make sure that the sample
                # is classified as unknown.
                if not np.isfinite(logPz):
                    logPz = -1000

                distance = distances[i]

                logPe = np.log(r_pdf(distance, bin_edges, counts))
                logPe -= np.log(distance) * (32 * 32 - z_size)

                P = logPz + logPe

                result.append(P)

    zlist = np.concatenate(zlist)
    labellist = np.concatenate(labellist)
    return rlist, zlist, labellist, result


def compute_result(dataset, train_classes, inliner_classes, gennorm_param, bin_edges, counts, threshhold=None):
    rlist, zlist, labellist, result = gpnd(dataset, run_gpnd=True, gennorm_param=gennorm_param, bin_edges=bin_edges,
                                           counts=counts)

    def compute_f1(t):
        true_positive = 0
        false_positive = 0
        false_negative = 0

        known_samples = 0
        classified_as_known = 0

        for i in range(len(dataset)):
            label = labellist[i]
            known = label in inliner_classes

            #  SVM should predict this
            predicted_class = 0

            if result[i] > t:
                correct = train_classes[predicted_class] == label
                classified_as_known += 1
            else:
                correct = not known

            known_samples += known

            true_positive += known and correct
            false_positive += (not known) and (not correct)
            false_negative += known and (not correct)

        recall = true_positive / known_samples#(true_positive + false_negative)

        if classified_as_known > 0:
            precision = true_positive / classified_as_known#(true_positive + false_positive)

            F1 = get_f1(precision, recall)
            return F1
        else:
            return 0

    if not threshhold is None:
        return compute_f1(threshhold), threshhold
    else:

        minP = min(result) - 1
        maxP = max(result) + 1

        best_e = 0
        best_f = 0
        best_e_ = 0
        best_f_ = 0

        print(minP, maxP)

        for e in np.arange(minP, maxP, 10):
            f = compute_f1(e)

            if f > best_f:
                best_f = f
                best_e = e
            if f >= best_f_:
                best_f_ = f
                best_e_ = e

        best_e = (best_e + best_e_) / 2.0

        print("Best e: ", best_e)
        return best_f_, best_e


def main(_folding_id, opennessid, _class_fold, folds=5):
    mnist_train = []
    mnist_valid = []
    #define svm classifier
    clf = svm.SVC(decision_function_shape='ovr')

    global folding_id
    global class_fold
    folding_id = _folding_id
    class_fold = _class_fold
    class_data = json.load(open('class_table_fold_%d.txt' % class_fold))

    train_classes = class_data[0]["train"]
    test_classes = class_data[opennessid]["test_target"]
    inliner_classes = train_classes
    outlier_classes = [x for x in test_classes if x not in inliner_classes]

    openness = 1.0 - math.sqrt(2 * len(train_classes) / (len(train_classes) + len(test_classes)))
    print("\tOpenness: %f" % openness)

    for i in range(folds):
        if i != folding_id:
            with open('data_fold_%d.pkl' % i, 'rb') as pkl:
                fold = pickle.load(pkl)
            if len(mnist_valid) == 0:
                mnist_valid = fold
            else:
                mnist_train += fold

    with open('data_fold_%d.pkl' % folding_id, 'rb') as pkl:
        mnist_test = pickle.load(pkl)

    random.shuffle(mnist_train)
    random.shuffle(mnist_valid)

    #keep only train classes
    mnist_train = [x for x in mnist_train if x[0] in train_classes]

    #keep only test classes
    mnist_valid = [x for x in mnist_valid if x[0] in test_classes]
    mnist_test = [x for x in mnist_test if x[0] in test_classes]

    print("Train set size:", len(mnist_train))

    rlist, zlist, labellist, _ = gpnd(mnist_train, run_gpnd=False)

    counts, bin_edges = np.histogram(rlist, bins=30, normed=True)

    plt.plot(bin_edges[1:], counts, linewidth=2)
    plt.xlabel(r"Distance, $\left \|\| I - \hat{I} \right \|\|$", fontsize=axis_title_size)
    plt.ylabel('Probability density', fontsize=axis_title_size)
    plt.title(r"PDF of distance for reconstruction error, $p\left(\left \|\| I - \hat{I} \right \|\| \right)$", fontsize=title_size)
    plt.grid(True)
    plt.xticks(fontsize=ticks_size)
    plt.yticks(fontsize=ticks_size)
    plt.tight_layout(rect=(0.0, 0.0, 1, 0.95))
    plt.savefig('mnist_randomsearch.pdf')
    plt.clf()
    plt.cla()
    plt.close()

    for i in range(z_size):
        plt.hist(zlist[:, i], bins='auto', histtype='step')

    plt.xlabel(r"$z$", fontsize=axis_title_size)
    plt.ylabel('Probability density', fontsize=axis_title_size)
    plt.title(r"PDF of embeding $p\left(z \right)$", fontsize=title_size)
    plt.grid(True)
    plt.xticks(fontsize=ticks_size)
    plt.yticks(fontsize=ticks_size)
    plt.tight_layout(rect=(0.0, 0.0, 1, 0.95))
    plt.savefig('mnist_embeding.pdf')
    plt.clf()
    plt.cla()
    plt.close()

    gennorm_param = np.zeros([3, z_size])
    for i in range(z_size):
        betta, loc, scale = scipy.stats.gennorm.fit(zlist[:, i])
        gennorm_param[0, i] = betta
        gennorm_param[1, i] = loc
        gennorm_param[2, i] = scale

    # train SVM on extracted z
    print("SVM!")
    clf.fit(zlist, labellist)
    print("SVM Done!")

    _, best_th = compute_result(mnist_valid, train_classes, inliner_classes, gennorm_param, bin_edges, counts, threshhold=None)

    F1, _ = compute_result(mnist_test, train_classes, inliner_classes, gennorm_param, bin_edges, counts, best_th)

    print("F1: %f" % (F1))
    return F1, best_th


if __name__ == '__main__':
    main(0, 1, 0, 5)
