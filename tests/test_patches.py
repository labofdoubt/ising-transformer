import torch

from tvan.patches import all_token_patches, patch_to_token, token_to_patch


def test_patch_roundtrip_all_tokens():
    for r, c in [(1, 1), (1, 3), (2, 2), (2, 4), (3, 4)]:
        if r * c > 12:
            continue
        vocab = 2 ** (r * c)
        tokens = torch.arange(vocab, dtype=torch.long)
        patches = token_to_patch(tokens, r, c)
        recovered = patch_to_token(patches)
        assert torch.equal(tokens, recovered)


def test_all_token_patches_matches_decode():
    patches = all_token_patches(2, 3)
    tokens = torch.arange(2 ** 6, dtype=torch.long)
    assert torch.equal(patches, token_to_patch(tokens, 2, 3))
