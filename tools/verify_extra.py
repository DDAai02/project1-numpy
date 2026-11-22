"""Additional invariants beyond the supplied reference images.

Uses new arrays, small synthetic attention weights and the supplied cat PNG.
Run with --starter pointing to the teacher's extracted student directory.
"""
from pathlib import Path
import argparse
import json
import sys
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--starter', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.starter.resolve()))
    from image_encoder import ImageEncoder, MultiHeadAttention
    from runtime import load_weights, load_images

    checks = {}
    encoder = ImageEncoder(load_weights())
    images = np.arange(2 * 3 * 224 * 224, dtype=np.float32).reshape(2, 3, 224, 224)
    expected = np.stack([
        images[:, :, row:row + 32, col:col + 32].reshape(2, -1)
        for row in range(0, 224, 32) for col in range(0, 224, 32)
    ], axis=1)
    np.testing.assert_array_equal(encoder.patchify(images), expected)
    checks['patch_order_unique_pixels'] = 'PASS'

    # A small width/head count verifies that attention is not hard-coded to 768/12.
    identity = np.eye(8, dtype=np.float32)
    weights = {
        'toy.in_proj_weight': np.concatenate((identity, identity, identity)),
        'toy.in_proj_bias': np.zeros(24, dtype=np.float32),
        'toy.out_proj.weight': identity,
        'toy.out_proj.bias': np.zeros(8, dtype=np.float32),
    }
    attention = MultiHeadAttention(weights, 'toy', 2)
    tokens = np.arange(2 * 5 * 8, dtype=np.float32).reshape(2, 5, 8) / 40
    split = attention.split_heads(tokens)
    assert split.shape == (2, 2, 5, 4)
    for head in range(2):
        np.testing.assert_array_equal(split[:, head], tokens[:, :, head * 4:(head + 1) * 4])
    checks['generic_head_partition'] = 'PASS'

    changed = tokens.copy()
    changed[:, -1] += 3
    before = attention(tokens, causal=True)
    after = attention(changed, causal=True)
    np.testing.assert_allclose(before[:, :-1], after[:, :-1], atol=1e-6, rtol=1e-6)
    assert not np.allclose(attention(tokens)[:, 0], attention(changed)[:, 0])
    assert after.shape == tokens.shape and after.dtype == np.float32
    checks['causal_future_independence'] = 'PASS'
    checks['noncausal_future_visibility'] = 'PASS'

    rng = np.random.default_rng(20261110)
    fresh = rng.normal(size=(3, 3, 224, 224)).astype(np.float32)
    batch = encoder(fresh)
    single = encoder(fresh[1:2])
    np.testing.assert_allclose(single, batch[1:2], atol=2e-5, rtol=2e-4)
    np.testing.assert_allclose(np.sum(batch * batch, axis=-1), 1, atol=2e-4, rtol=2e-4)
    assert batch.shape == (3, 512) and batch.dtype == np.float32 and np.isfinite(batch).all()
    checks['unseen_random_images'] = 'PASS'
    checks['single_batch_equivalence'] = 'PASS'
    checks['batch_max_abs'] = float(np.max(np.abs(single - batch[1:2])))

    cat = args.starter / 'lecture-demos' / 'images' / 'uw_cat.png'
    extra = encoder(load_images([cat, cat]))
    np.testing.assert_allclose(extra[0], extra[1], atol=2e-5, rtol=2e-4)
    assert extra.shape == (2, 512) and np.isfinite(extra).all()
    checks['non_reference_png_and_duplicates'] = 'PASS'
    for key, value in checks.items():
        print(key + ': ' + str(value))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(checks, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
