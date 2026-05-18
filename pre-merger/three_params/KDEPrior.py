import numpy as np
from bilby.core.prior.joint import BaseJointPriorDist


class KDEJointDist(BaseJointPriorDist):
    """
    Joint prior backed by a scipy gaussian_kde.

    Parameters
    ----------
    names : list[str]
    kde   : scipy.stats.gaussian_kde
    bounds : list[tuple]  [(lo, hi), ...]  one per name
    """

    def __init__(self, names, kde, bounds):
        super().__init__(names=names, bounds=bounds)
        self.kde = kde

    # bilby calls this with samp shape (n, d), outbounds shape (n,)
    def _ln_prob(self, samp, lnprob, outbounds):
        mask = ~outbounds
        if mask.any():
            lnprob[mask] = self.kde.logpdf(samp[mask].T)
        lnprob[outbounds] = -np.inf
        return lnprob

    def _sample(self, size, **kwargs):
        lo = np.array([self.bounds[n][0] for n in self.names])
        hi = np.array([self.bounds[n][1] for n in self.names])
        samples = np.empty((size, len(self.names)))
        drawn = 0
        while drawn < size:
            batch = self.kde.resample(size * 5).T          # (5*size, d)
            in_bounds = np.all((batch >= lo) & (batch <= hi), axis=1)
            candidates = batch[in_bounds]
            take = min(len(candidates), size - drawn)
            samples[drawn : drawn + take] = candidates[:take]
            drawn += take
        return samples



# import numpy as np
# from bilby.core.prior.joint import BaseJointPriorDist

# class KDEJointDist(BaseJointPriorDist):
#     def __init__(self, names, kde, bounds):
#         super().__init__(names=names, bounds=bounds)
#         self.kde = kde

#     def _ln_prob(self, samp, lnprob, outbounds):
#         lnprob[~outbounds] = self.kde.logpdf(samp[~outbounds].T)
#         lnprob[outbounds] = -np.inf
#         return lnprob

#     def _sample(self, size, **kwargs):
#         samples = np.zeros((size, len(self)))
#         drawn = 0
#         while drawn < size:
#             batch = self.kde.resample(size * 5).T
#             for s in batch:
#                 if drawn >= size:
#                     break
#                 if all(self.bounds[name][0] <= val <= self.bounds[name][1]
#                        for name, val in zip(self.names, s)):
#                     samples[drawn] = s
#                     drawn += 1
#         return samples