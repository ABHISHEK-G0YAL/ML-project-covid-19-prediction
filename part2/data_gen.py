import numpy as np

def gen_points(num_samples, mu, r):
    y = np.random.multivariate_normal(mu, r, size=num_samples)
    a = y[:,0]
    t = y[:,1]
    a = sorted(a, reverse=True)
    return a

# for infection rate
num_samples = 30
# The desired mean values of the sample.
mu = np.array([22.75, 42548.0])
# The desired covariance matrix.
r = np.array([
        [  0.1225, -11106.55],
        [ -11106.55,  1006983289.0]
    ])
gen_points(num_samples, mu, r)

# for death rate
num_samples = 30
# The desired mean values of the sample.
mu = np.array([22.75, 1384.0])
# The desired covariance matrix.
r = np.array([
        [  0.1225, -360.85],
        [ -360.85,  1062961.0]
    ])