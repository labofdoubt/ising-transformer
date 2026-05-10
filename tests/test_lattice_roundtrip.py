import torch

from tvan.lattice import lattice_to_tokens, tokens_to_lattice


def test_lattice_roundtrip():
    torch.manual_seed(0)
    spins = torch.randint(0, 2, (5, 8, 8), dtype=torch.long).mul(2).sub(1)
    for r, c in [(1, 1), (2, 2), (2, 4), (4, 2)]:
        tokens = lattice_to_tokens(spins, r, c)
        recovered = tokens_to_lattice(tokens, 8, r, c)
        assert torch.equal(spins, recovered)
