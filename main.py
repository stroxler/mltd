from dataclasses import dataclass

import torch
from torch import Tensor
from jaxtyping import jaxtyped, Float
import beartype


# Intro ------------

@jaxtyped(typechecker=beartype.beartype)
def determinant_2x2(
    x: Float[Tensor, "2 2"]
) -> Float[Tensor, ""]:
    return x[0, 0] * x[1, 1] - x[0, 1] * x[1, 0]


@jaxtyped(typechecker=beartype.beartype)
def matrix_multiply(
    x: Float[Tensor, "a b"],
    y: Float[Tensor, "b c"],
) -> Float[Tensor, "a c"]:
    return x @ y


@jaxtyped(typechecker=beartype.beartype)
def tensor_multiply(
    x: Float[Tensor, "*a b"],
    y: Float[Tensor, "b c"],
) -> Float[Tensor, "*a c"]:
    return x @ y


@jaxtyped(typechecker=beartype.beartype)
def pointwise_multiply(
    x: Float[Tensor, "*a"],
    y: Float[Tensor, "#*a"],
) -> Float[Tensor, "*a"]:
    return x * y


@jaxtyped(typechecker=beartype.beartype)
def random_chisq(
    n: int
) -> Float[Tensor, "{n}"]:
    # pyrefly: ignore (TODO: Pyrefly can't model descriptor behavior on this)
    return torch.randn(n) ** 2


@jaxtyped(typechecker=beartype.beartype)
def tail(
    x: Float[Tensor, "dim"],
) -> Float[Tensor, "dim-1"]:
    return x[1:]


# Regression example ------------


@jaxtyped(typechecker=beartype.beartype)
def multiple_linear_regression(
    x: Float[Tensor, "n p"],
    y: Float[Tensor, "n k"]
) -> Float[Tensor, "p k"]:
    return torch.linalg.solve(x.t() @ x, x.t() @ y)


@jaxtyped(typechecker=beartype.beartype)
def predict(
    beta_hat: Float[Tensor, "p k"],
    x: Float[Tensor, "n p"],
) -> Float[Tensor, "n k"]:
    return x @ beta_hat


# Dataclass example ------------


@beartype.beartype
@dataclass(frozen=True)
class Synthetic:
    beta: Float[Tensor, "{p} {k}"]
    x: Float[Tensor, "{n} {p}"]
    y: Float[Tensor, "{n} {k}"]


@jaxtyped(typechecker=beartype.beartype)
def generate(
    p: int,
    k: int,
    n: int,
    sigma: float,
) -> Synthetic:
    """
    Note: as of Python 3.13 it's not really yet feasible to make this a
    classmethod because that would require a forward reference.
    """
    beta = torch.rand(p, k)
    x = torch.rand(n, p)
    epsilon = torch.rand(n, k)
    y = x @ beta + sigma * epsilon
    return Synthetic(beta, x, y)

# Some other demo?



# What happens if we try to use it with Pydantic?
# This is a possible hack project for me