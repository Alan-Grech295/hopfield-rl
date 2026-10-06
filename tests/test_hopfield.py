import math

import torch
import pytest

from hopfield_rl.hopfield import HopfieldNetwork


@pytest.fixture(params=range(10))
def seed(request):
    return request.param


@pytest.fixture(params=[0.1, 0.5, 1.0, 2.0])
def temperature(request):
    return request.param


@pytest.fixture
def network_case(seed, temperature):
    torch.manual_seed(seed)

    dims = 16

    network = HopfieldNetwork(
        dims,
        16,
        temperature=temperature,
    )

    x = torch.randn(dims, dtype=torch.float64)
    drive = torch.randn(dims, dtype=torch.float64)

    return network, x, drive


def test_energy_gradient_matches_update(network_case):
    network, x, drive = network_case
    x = x.clone().requires_grad_(True)

    energy = network.energy(x, drive)
    energy_grad, = torch.autograd.grad(energy, x)

    x_next = network.update(x, drive)
    expected_grad = x - x_next

    assert energy.ndim == 0
    assert energy_grad.shape == x.shape
    assert torch.isfinite(energy)
    assert torch.isfinite(energy_grad).all()

    assert torch.allclose(
        energy_grad,
        expected_grad,
        atol=1e-10,
        rtol=1e-10,
    )


def test_jacobian_matches_finite_difference(network_case):
    network, x, drive = network_case
    dims = len(x)

    eps = 1e-6

    J_fd = torch.zeros(dims, dims, dtype=torch.float64)

    for j in range(dims):
        x_plus = x.clone()
        x_minus = x.clone()

        x_plus[j] += eps
        x_minus[j] -= eps

        T_plus = network.update(x_plus, drive)
        T_minus = network.update(x_minus, drive)

        J_fd[:, j] = (T_plus - T_minus) / (2 * eps)

    J_analytic = network.jacobian(x)
    difference = J_analytic - J_fd
    rel_error = torch.linalg.norm(difference) / max(torch.linalg.norm(J_fd), eps)

    assert rel_error < 1e-3


def settle(network, x, drive, max_iterations=100, tol=1e-4):
    converged = False
    energies = []
    residuals = []
    for i in range(max_iterations):
        energy = network.energy(x, drive)
        x_next = network.update(x, drive)
        residual = torch.linalg.norm(x_next - x)
        energies.append(energy)
        residuals.append(residual)

        assert torch.isfinite(x).all()
        assert torch.isfinite(energy)

        x = x_next
        if residual < tol:
            converged = True
            break

    return {
        "state": x,
        "converged": converged,
        "iterations": i + 1,
        "energies": torch.stack(energies),
        "residuals": torch.stack(residuals)
    }


def test_network_converges(network_case):
    network, x, drive = network_case

    results = settle(network, x, drive, max_iterations=1000, tol=1e-4)

    assert results["converged"]

    current_x = results["state"]
    x_next = network.update(current_x, drive)
    residual = torch.linalg.norm(x_next - current_x)
    assert residual < 1e-4


def test_network_update_does_not_increase_energy(network_case):
    network, x, drive = network_case

    results = settle(network, x, drive, max_iterations=1000, tol=1e-4)

    assert torch.all(results["energies"][1:] <= results["energies"][:-1] + 1e-6)

def test_tied_scores(temperature):
    dims = 16

    v = torch.randn(dims, dtype=torch.float64)
    v = torch.nn.functional.normalize(v, dim=0)

    network = HopfieldNetwork(
        dims,
        2,
        temperature=temperature,
        memories=torch.stack([v, -v]),
    )

    x = torch.zeros(dims, dtype=torch.float64)
    drive = torch.zeros(dims, dtype=torch.float64)

    weights = network.weights(x)
    assert torch.allclose(weights, torch.ones(2, dtype=torch.float64) * 0.5)

    energy = network.energy(x, drive)
    assert torch.isclose(energy, torch.tensor(0, dtype=torch.float64))

    J = network.jacobian(x)
    eigenvalues = torch.linalg.eigvalsh(J)
    largest = eigenvalues[-1]

    assert torch.isclose(largest, torch.tensor(1 / temperature, dtype=torch.float64))

def test_largest_jacobian_eigen_is_beta(temperature):
    dims = 16

    v = torch.randn(dims, dtype=torch.float64)
    v = torch.nn.functional.normalize(v, dim=0)

    network = HopfieldNetwork(
        dims,
        2,
        temperature=temperature,
        memories=torch.stack([v, -v]),
    )

    x = torch.zeros(dims, dtype=torch.float64)

    J = network.jacobian(x)
    eigenvalues = torch.linalg.eigvalsh(J)
    largest = eigenvalues[-1]

    assert torch.isclose(largest, torch.tensor(1 / temperature, dtype=torch.float64))

@pytest.mark.parametrize("temperature", [0.1, 0.5])
@pytest.mark.parametrize("deviation", [0.0001, -0.0001])
def test_fixed_point_is_unstable(temperature, deviation):
    dims = 16

    v = torch.randn(dims, dtype=torch.float64)
    v = torch.nn.functional.normalize(v, dim=0)

    network = HopfieldNetwork(
        dims,
        2,
        temperature=temperature,
        memories=torch.stack([v, -v]),
    )

    x = torch.zeros(dims, dtype=torch.float64)
    x += v * deviation

    results = settle(network, x, torch.zeros(dims, dtype=torch.float64), 1000)

    assert results["converged"]

    final_projection = torch.dot(results["state"], v)

    assert torch.sign(final_projection) == math.copysign(1, deviation) # State moved to the correct side
    assert abs(final_projection) > abs(deviation) # State moved away significantly