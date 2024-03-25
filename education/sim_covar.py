import numpy as np

# X = pd.DataFrame()
# X["gender"] = np.random.binomial(1, 0.5, n)
# X["age"] = np.random.normal(20, 4, n)
# xb = -9 + 3.5 * X.gender + 0.2 * X.age
# p = 1/(1 + np.exp(-xb))
# y = np.random.binomial(1, p, n)
# clf = LogisticRegression(random_state=0).fit(X, y)
# clf.coef_


# Define a vector of means and a matrix of covariances
mean = [0.25, 0.60]
Sigma = [[1, 0.70],
         [0.70, 1]]

# Generate 100 cases
X = np.random.default_rng().multivariate_normal(mean, Sigma, 100).T

# Subtract the mean from each variable
X = np.apply_along_axis(lambda x: x - np.mean(x), 1, X)
# for n in range(X.shape[0]):
#     X[n] = X[n] - X[n].mean()

# # Make each variable in X orthogonal to one another
L_inv = np.linalg.cholesky(np.cov(X, bias = True))
L_inv = np.linalg.inv(L_inv)
X = np.dot(L_inv, X)

# # Rescale X to exactly match Sigma
L = np.linalg.cholesky(Sigma)
X = np.dot(L, X)

# # Add the mean back into each variable
# X = np.apply_along_axis(lambda x: x + np.mean(x), 1, X)
for n in range(X.shape[0]):
    X[n] = X[n] + mean[n]

# # The covariance of the generated data should match Sigma
print(np.cov(X, bias = True))

# pd.DataFrame(X.T).applymap(lambda x: 1 if x > 0 else 0).corr()