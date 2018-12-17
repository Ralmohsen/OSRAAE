# OSRAAE

Open-set Recognition with Adversarial Autoencoders

A deep learning approach to solving the problem of open-set recognition, by
leveraging an encoder-decoder network architecture in conjunction with a
multi-class classifier. The network enables learning a novelty detector that
computes the probability of a sample to belong to one of the known classes
versus being unknown. If known, the multi-class classifiers assigns the class
label to the sample.


|       F1 Score on MNIST Dataset                 |
| --------|-------|-------| ------|-------|-------|
| Ours    | 0.994 | 0.924 | 0.876 | 0.869 | 0.875 |
| GOpenMax| 0.994 | 0.921 | 0.872 | 0.825 | 0.812 |
| OpenMax | 0.994 | 0.910 | 0.842 | 0.810 | 0.773 |
