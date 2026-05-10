import torch

from tvan.physics import ising_energy


def brute_force_energy(spins: torch.Tensor, J: float = 1.0) -> torch.Tensor:
    B, L, _ = spins.shape
    values = []
    for b in range(B):
        total = 0.0
        for x in range(L):
            for y in range(L):
                total += -J * spins[b, x, y].item() * (
                    spins[b, (x + 1) % L, y].item() + spins[b, x, (y + 1) % L].item()
                )
        values.append(total)
    return torch.tensor(values, dtype=torch.float32)


def test_ising_energy_matches_bruteforce():
    torch.manual_seed(0)
    spins = torch.randint(0, 2, (7, 4, 4), dtype=torch.long).mul(2).sub(1)
    expected = brute_force_energy(spins, J=1.7)
    actual = ising_energy(spins.float(), J=1.7)
    assert torch.allclose(actual, expected)
