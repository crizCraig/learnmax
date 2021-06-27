# import numpy as np
# import scipy
# from scipy.special import rel_entr
#
# p = [0.1, 0.1, 0.8]
# q = [0.0, 0.0, 1.0]
#
# vec = rel_entr(p, q)
# vec = np.ma.masked_invalid(vec).compressed()
#
# kl_div = np.sum(vec)
#
# print(kl_div)


import numpy as np


def KL(P, Q):
    """ Epsilon is used here to avoid conditional code for
    checking that neither P nor Q is equal to 0. """
    epsilon = 0.00001

    # You may want to instead make copies to avoid changing the np arrays.
    P = P + epsilon
    Q = Q + epsilon

    divergence = np.sum(P * np.log(P / Q))
    return divergence


# Should be normalized though
values1 = np.asarray([1.346112, 1.337432, 1.246655])
values2 = np.asarray([1.033836, 1.082015, 1.117323])
# values1 = np.asarray([0.1,0.1,0.8])
# values2 = np.asarray([0., 0., 1.])

values1 /= np.sum(values1)
values2 /= np.sum(values2)

# Note slight difference in the final result compared to Dawny33
print(KL(values1, values2))  # 0.775278939433)
