from botorch.models.gpytorch import GPyTorchModel
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior
from botorch.fit import fit_gpytorch_model
import random
import numpy as np
from ax import SearchSpace, RangeParameter, ParameterType, SimpleExperiment
from ax.modelbridge import get_sobol
from ax.modelbridge.factory import get_botorch


class SimpleCustomGP(ExactGP, GPyTorchModel):

    _num_outputs = 1  # to inform GPyTorchModel API
    
    def __init__(self, train_X, train_Y):
        # squeeze output dim before passing train_Y to ExactGP
        super().__init__(train_X, train_Y.squeeze(-1), GaussianLikelihood())
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            base_kernel=RBFKernel(ard_num_dims=train_X.shape[-1]),
        )
        self.to(train_X)  # make sure we're on the right device/dtype
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

def get_and_fit_gpytorch_model(
    Xs: List[Tensor],
    Ys: List[Tensor],
    Yvars: List[Tensor],
    state_dict: Optional[Dict[str, Tensor]] = None,
    **kwargs: Any,
) -> Model:

    def _get_and_fit_simple_custom_gp(Xs, Ys, **kwargs):
        model = SimpleCustomGP(Xs[0], Ys[0])
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)
        return model

def branin(parameterization, *args):
    x1, x2 = parameterization["x1"], parameterization["x2"]
    y = (x2 - 5.1 / (4 * np.pi ** 2) * x1 ** 2 + 5 * x1 / np.pi - 6) ** 2
    y += 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10
    # let's add some synthetic observation noise
    y += random.normalvariate(0, 0.1)
    return {"branin": (y, 0.0)}

search_space = SearchSpace(
    parameters=[
        RangeParameter(
            name="x1", parameter_type=ParameterType.FLOAT, lower=-5, upper=10
        ),
        RangeParameter(
            name="x2", parameter_type=ParameterType.FLOAT, lower=-5, upper=10
        ),
    ]
)

exp = SimpleExperiment(
    name="test_branin",
    search_space=search_space,
    evaluation_function=branin,
    objective_name="branin",
    minimize=True,
)

for i in range(50):
    print(f"Running optimization batch {i + 1}/5...")
    model = get_botorch(
        experiment=exp,
        data=exp.eval(),
        search_space=exp.search_space,
        model_constructor=_get_and_fit_simple_custom_gp,
    )
    batch = exp.new_trial(generator_run=model.gen(1))
print("Done!")
