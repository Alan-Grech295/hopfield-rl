import torch

from hopfield import HopfieldNetwork

dims = 16

x = torch.randn(dims, dtype=torch.float64, requires_grad=True)
drive = torch.randn(dims, dtype=torch.float64)

network = HopfieldNetwork(dims, 16, 1)

energy = network.energy(x, drive)
print("Energy:", energy)

energy_grad, = torch.autograd.grad(energy, x)
print(energy_grad.shape)

x_next = network.update(x, drive)

expected_grad = x - x_next

difference = energy_grad - expected_grad

print("Difference:", difference)
print("Maximum error:", difference.abs().max())

x = torch.randn(dims, dtype=torch.float64, requires_grad=True)
drive = torch.randn(dims, dtype=torch.float64)

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

print(difference.abs().max())
print("Relative error:", rel_error)
