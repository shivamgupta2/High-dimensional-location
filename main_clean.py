import numpy as np
from scipy import stats

NUM_SAMPLES_I = 10**7  # Number of samples used to estimate fisher information


class Mixture:
    def __init__(self, d, rs, ps=None, centers=None):
        self.d = d
        self.k = len(rs)
        if centers is None:
            centers = np.random.randn(self.k, d) / d**.5
        self.centers = np.array(centers)
        self.rs = np.array(rs, dtype='float64')
        self.Iests = {}
        if ps is None:
            ps = np.array([1]*self.k)
        assert len(ps) == len(rs)
        self.ps = ps / np.sum(ps)

    def sample(self, n):
        samples = np.random.randn(n, self.d)
        mix = stats.rv_discrete(values=(np.arange(self.k), self.ps))
        choices = mix.rvs(size=n)
        result = self.centers[choices] + self.rs[choices].reshape([-1, 1]) * samples
        return result

    def pmf(self, xs):
        assert xs.shape[-1] == self.d
        xs = xs.reshape((-1, 1, self.d))
        offsets = xs - self.centers  #n x k x d
        pdf = np.zeros(len(xs))
        for i in range(self.k):
            pdf += stats.multivariate_normal.pdf(offsets[:,i,:], cov=np.identity(self.d)*self.rs[i])
        return pdf

    def score(self, xs):
        """
        xs: n x d
        """
        xs = np.array(xs)
        n = len(xs)
        # Single gaussian: -(Sigma^{-1})(x-mu)
        assert xs.shape[-1] == self.d
        xs = xs.reshape((-1, 1, self.d))
        offsets = xs - self.centers  #n x k x d

        # Compute weighted average of individual scores. (sum p)'/(sum p) = (sum (p'/p) * p)/sum p
        # Use logpdfs for better conditioning (shift so max prob. is constant)
        logpdfs = []
        scores = []
        for i in range(self.k):
            logpdfs.append(stats.multivariate_normal.logpdf(offsets[:,i,:], cov=np.identity(self.d)*self.rs[i]**2))
            scores.append( - self.rs[i]**(-2) * offsets[:,i,:])
        logpdfs = np.array(logpdfs) #k x n
        logpdfs -= np.max(logpdfs, axis=0)
        pdfs = np.exp(logpdfs).reshape(self.k, n, 1)
        pdfs /= np.sum(pdfs, axis=0)
        scores = np.array(scores) # k x n x d
        score = np.sum(pdfs * scores, axis=0)
        return score


    def mean(self):
        return np.sum([c * p for c, p in zip(self.centers, self.ps)], axis=0)

    def var(self):
        mu = self.mean()
        Sigma = sum(p * ((c-mu).reshape([-1,1])*(c-mu).reshape([1,-1]) + np.identity(self.d)*r) for p, c, r in zip(self.ps, self.centers, self.rs))
        return Sigma

    def getI(self, N):
        maxbatch = max(1, 10**6 // self.d**2)
        ans = 0
        n = N
        while n > 0:
            batch = min(n, maxbatch)
            n -= batch
            # batch samples from each gaussian
            samples = np.random.randn(batch, 1, self.d) * self.rs.reshape([1, -1, 1]) + self.centers.reshape([1, -1, self.d])
            # batch x k x d
            for i in range(self.k):
                scores = self.score(samples[:,i,:]) #batch x d
                SST = scores.reshape((batch, -1, 1)) * scores.reshape((batch, 1, -1))
                ans += np.sum(SST, axis=0) * self.ps[i]
        return ans / N

    def getI_old(self, N):
        # This is less accurate than getI, which reweights to sample each gaussian equally
        # Probably one could figure out a more optimal weighting scheme though
        maxbatch = max(1, 10**6 // self.d**2)
        ans = 0
        n = N
        while n > 0:
            batch = min(n, maxbatch)
            n -= batch
            samples = self.sample(batch)
            scores = self.score(samples) # batch x d
            SST = scores.reshape((batch, -1, 1)) * scores.reshape((batch, 1, -1))
            ans += np.sum(SST, axis=0)
        return ans / N

    def getsmoothedI(self, N, r):
        m2 = Mixture(self.d, np.sqrt(self.rs**2 + r**2), self.ps, self.centers)
        return m2.getI(N)

    def runalg(self, samples, r, Iest=None, steps=1):
        samples = np.array(samples)
        n = samples.shape[0]
        m2 = Mixture(self.d, np.sqrt(self.rs**2 + r**2), self.ps, self.centers)
        smoothsamples = samples + np.random.randn(n, self.d)*r

        if Iest is None:
            Iest = self.Iests.get(r, None)
        if Iest is None:
            print(f"Getting I for {r}...")
            Iest = m2.getI(NUM_SAMPLES_I)
            self.Iests[r] = Iest
            print("...done")

        mean = np.mean(samples, axis=0) - self.mean()
        for i in range(steps):
            meanscore = np.mean(m2.score(smoothsamples - mean), axis=0)
            # Todo: cache the inverted matrix
            shift = np.linalg.inv(Iest).dot(meanscore)
            mean = mean - shift
        return mean


    def repeatedMLE(self, samples, r, Iest, count=100):
        ans = []
        for i in range(count):
            ans.append(self.runalg(samples, r, Iest))
        return np.mean(ans, axis=0)

    def test_recovery(self, samples, r):
        ans = {}
        scale = len(samples)**.5  #scale up to C
        ans['Ours, r=.1'] = np.linalg.norm(self.runalg(samples, .1)) *scale
        ans['Ours, r=0'] = np.linalg.norm(self.runalg(samples, 0)) *scale

        ans['Ours-10, r=0'] = np.linalg.norm(self.runalg(samples, 0,steps=10)) *scale
        ans['mean'] = np.linalg.norm(np.mean(samples, axis=0)-self.mean())*scale
        return ans

    def test_recovery_repeat(self, N, k, r):
        ans = {}
        for i in range(k):
            result = self.test_recovery(self.sample(N), r)
            for key in result:
                ans.setdefault(key, []).append(result[key])
        for key in ans:
            ans[key].sort()
        return ans


if __name__ == '__main__':
    r = 0.1
    d = 20
    m = Mixture(d, [1, 3, 1e-3], [1, 1, 0.0001], [[-1]+[0]*(d-1),[1]+[0]*(d-1), [0,10000]+[0]*(d-2)])
    print(f"Running: {m.rs} {m.ps}")
    results = {}
    repetitions = 100
    for i in range(1, 7):
        ans = m.test_recovery_repeat(10**i, repetitions, r)
        results[i] = ans
        print(f"10**{i}")
        for key in ans:
            print(key, np.median(ans[key]), ans[key][(len(ans[key])*9)//10], np.mean(ans[key]), np.std(ans[key]), np.max(ans[key]))

    print()
    print("Scaled errors [L2 error is C/sqrt(n) for the given C]")
    print('sqrt trace:')
    for r in m.Iests:
        print(r, np.sqrt(np.trace(np.linalg.inv(m.Iests[r]))))

    print(f"Reminder: {m.rs} {m.ps}")
    print()
    for (label, func) in [('Medians:', np.median),
                          ('90th percentile', lambda x: x[(9*len(x))//10]),
                          ('Means:', np.mean),
                          ('Maxes:', np.max),
                          ]:
        print(label)
        print(f'{"N":11}\t' + '\t'.join(f'10^{k}' for k in results.keys()))
        for key in ans:
            vals = [func(results[i][key]) for i in results]
            line = f'{key:11}\t' + '\t'.join(f'{val:0.2f}' for val in vals)
            print(line)
        print()
