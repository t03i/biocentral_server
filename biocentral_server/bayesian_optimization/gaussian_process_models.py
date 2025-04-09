import gpytorch
import torch


class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def marginalization(mvn_dist):
    return torch.distributions.Normal(
        mvn_dist.mean, mvn_dist.covariance_matrix.diag().sqrt()
    )


def MC_sampling(mvn_dist, lb, ub, n_samples=100000):
    """
    Calculate probability using Monte Carlo sampling.
    Args:
        mvn_dist: gpytorch.distributions.MultivariateNormal
        lb: torch.Tensor lower bounds
        ub: torch.Tensor upper bounds
        n_samples: Number of samples to use
    Returns:
        probability: (dim,)
    """
    with torch.no_grad():
        samples = mvn_dist.sample(torch.Size([n_samples]))
        prob = ((samples >= lb) & (samples <= ub)).float().mean(dim=0)
    return prob

# train model: training set => 
def trainGPRegModel(trainset: dict, lr: float = 0.3, epoch: int = 120):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPRegressionModel(trainset['X'], trainset['y'], likelihood)
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.3)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for i in range(epoch):
        optimizer.zero_grad()
        output = model(trainset['X'])
        loss = -mll(output, trainset['y'])
        loss.backward()
        optimizer.step()
    model.eval()
    likelihood.eval()
    return model, likelihood

def trainGPClsModel(trainset: dict, lr: float = 0.3, epoch: int = 120):
    pass