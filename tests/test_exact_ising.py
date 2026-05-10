import math

from tvan.exact_ising import exact_ising_free_energy


def test_exact_ising_critical_regression():
    beta_c = 0.5 * math.log(1.0 + math.sqrt(2.0))
    f120 = exact_ising_free_energy(L=120, beta=beta_c, J=1.0) / (120 ** 2)
    f128 = exact_ising_free_energy(L=128, beta=beta_c, J=1.0) / (128 ** 2)
    assert abs(f120 - (-2.10975198)) < 5e-8
    assert abs(f128 - (-2.10973977)) < 5e-8
