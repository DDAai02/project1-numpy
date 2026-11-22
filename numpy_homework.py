"""NumPy exercises NP-E1..E8 and NP-M1..M4.

Run this file to print all intermediate results. Calculations use full precision;
six decimal places are used only for display. Codex assisted implementation.
"""
import numpy as np


def _vector(value, name):
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or result.size == 0 or not np.isfinite(result).all():
        raise ValueError(name + ' must be a finite, nonempty 1D vector')
    return result


def _nonzero_length(vector):
    length = np.sqrt(np.sum(vector * vector))
    if length == 0:
        raise ValueError('zero vector has no direction')
    return length


def angle_deg(a, b):
    """NP-E3: angle in degrees; zero vectors are invalid."""
    a, b = _vector(a, 'a'), _vector(b, 'b')
    if a.shape != b.shape:
        raise ValueError('vector dimensions must match')
    cosine = (a @ b) / (_nonzero_length(a) * _nonzero_length(b))
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def unit_vector(a):
    """NP-E5: a new unit vector; unlike the model helper, reject zero."""
    a = _vector(a, 'a')
    return a / _nonzero_length(a)


def _rows(query, rows):
    query = _vector(query, 'query')
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != query.size or not np.isfinite(rows).all():
        raise ValueError('rows must be a finite 2D array matching query width')
    return query, rows, _nonzero_length(query)


def angles_to(query, X):
    """NP-M1: vectorized angles; zero rows retain their place as NaN."""
    query, X, query_length = _rows(query, X)
    lengths = np.sqrt(np.sum(X * X, axis=1))
    valid = lengths > 0
    result = np.full(X.shape[0], np.nan)
    cosine = (X[valid] @ query) / (lengths[valid] * query_length)
    result[valid] = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return result


def top_k_cosine(query, gallery, k):
    """NP-M2: descending scores, stable ties, original gallery indices."""
    query, gallery, query_length = _rows(query, gallery)
    lengths = np.sqrt(np.sum(gallery * gallery, axis=1))
    valid_indices = np.flatnonzero(lengths > 0)
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)):
        raise ValueError('k must be a positive integer')
    if not 1 <= k <= valid_indices.size:
        raise ValueError('k must not exceed the number of nonzero rows')
    scores = (gallery[valid_indices] @ query) / (lengths[valid_indices] * query_length)
    order = np.argsort(-scores, kind='stable')[:k]
    return valid_indices[order], scores[order]


def pairwise_distances(A, B):
    """NP-M3: [M,1,D] - [1,N,D] -> [M,N,D] -> [M,N]."""
    A, B = np.asarray(A, dtype=np.float64), np.asarray(B, dtype=np.float64)
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[1]:
        raise ValueError('A and B must be 2D arrays with matching coordinate width')
    differences = A[:, None, :] - B[None, :, :]
    return np.sqrt(np.sum(differences * differences, axis=-1))


def _show(name, value):
    print(f'{name}: shape={np.shape(value)}\n{value}')


def _boundary_checks():
    # Boundaries required by the exercises: zeros, stable ties and invalid k.
    for call in (
        lambda: angle_deg([0, 0], [1, 0]),
        lambda: unit_vector([0, 0]),
        lambda: angles_to([0, 0], [[1, 0]]),
        lambda: top_k_cosine([0, 0], [[1, 0]], 1),
    ):
        try:
            call()
        except ValueError:
            pass
        else:
            raise AssertionError('zero direction must raise ValueError')
    gallery = np.array([[0., 0.], [1., 0.], [2., 0.], [-1., 0.]])
    indices, scores = top_k_cosine([1, 0], gallery, 3)
    np.testing.assert_array_equal(indices, [1, 2, 3])
    np.testing.assert_allclose(scores, [1, 1, -1])
    for k in (0, -1, 4, 1.5, True):
        try:
            top_k_cosine([1, 0], gallery, k)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid k must raise ValueError')
    angles = angles_to([1, 0], [[0, 0], [1, 0], [-1, 0], [0, 1]])
    assert np.isnan(angles[0])
    np.testing.assert_allclose(angles[1:], [0, 180, 90], atol=1e-6)
    source = np.array([3., 4.])
    normalized = unit_vector(source)
    np.testing.assert_array_equal(source, [3., 4.])
    np.testing.assert_allclose(normalized, [.6, .8])
    np.testing.assert_allclose(pairwise_distances([[0, 0]], [[3, 4]]), [[5.]])
    print('\nRequired boundary checks: PASS')


def main():
    np.set_printoptions(precision=6, suppress=True)
    print('NP-E1: arrays, shapes, slicing and a separate copy')
    A = np.array([[.20, -.50, 1.10, .40], [.80, .30, -.70, .60], [-.40, .90, .10, -.20]])
    print('A.ndim =', A.ndim, 'A.shape =', A.shape, 'A.size =', A.size)
    _show('second row', A[1])
    _show('last two columns', A[:, -2:])
    B = A.copy()
    B[B < 0] = 0
    _show('B', B)
    _show('unchanged A', A)

    print('\nNP-E2: elementwise product, dot product and lengths')
    a, b = np.array([.80, -.60, .70]), np.array([-.20, .90, .40])
    _show('a * b', a * b)
    _show('a @ b', a @ b)
    for name, vector in (('a', a), ('b', b)):
        length = np.sqrt(np.sum(vector * vector))
        print(name, 'length =', length)
        np.testing.assert_allclose(length, np.linalg.norm(vector), atol=1e-6)

    print('\nNP-E3: angle')
    _show('degrees', angle_deg([1., -1.], [1., 0.]))

    print('\nNP-E4: distances')
    p, q = np.array([-.50, 1.20, 2.00]), np.array([.80, -.40, 1.10])
    _show('p - q', p - q)
    _show('Euclidean', np.sqrt(np.sum((p - q) ** 2)))
    _show('Manhattan', np.sum(np.abs(p - q)))

    print('\nNP-E5: unit vector')
    a = np.array([-1.20, .40, .80])
    normalized = unit_vector(a)
    _show('unit vector', normalized)
    print('unit length:', np.allclose(np.linalg.norm(normalized), 1.0))

    print('\nNP-E6: reductions')
    A = np.array([[.20, -.50, 1.10], [.80, .40, -.30]])
    _show('sum(axis=0), one sum per column', A.sum(axis=0))
    _show('mean(axis=1), one mean per row', A.mean(axis=1))
    _show('overall maximum', A.max())

    print('\nNP-E7: scale and bias broadcast across rows')
    X = np.array([[.80, -.60, .70], [-.40, .50, .20]])
    scale, bias = np.array([1.50, .50, -1.00]), np.array([-.10, .20, .30])
    _show('Y = X * scale + bias', X * scale + bias)

    print('\nNP-E8: image batch')
    a = np.array([[[.20, -.40], [.70, .10]], [[.30, .80], [-.20, .50]], [[-.10, .60], [.40, -.30]]])
    b = a + .10
    batch = np.stack((a, b), axis=0)
    print('BCHW:', batch.shape)
    _show('blue channels', batch[:, 2, :, :])
    print('BHWC:', batch.transpose(0, 2, 3, 1).shape)

    print('\nNP-M1: angles to all rows')
    query = np.array([.60, -.80, .00])
    X = np.array([[.30, -.40, .00], [-.60, .80, .00], [.80, .60, .00], [.00, .00, .00]])
    _show('angles', angles_to(query, X))

    print('\nNP-M2: stable top-k cosine retrieval')
    gallery = np.array([[.60, -.80, .00], [.30, -.40, .00], [.80, .60, .00], [-.60, .80, .00], [.00, .00, .00]])
    ids = np.array(['img_a', 'img_b', 'img_c', 'img_d', 'img_zero'])
    indices, similarities = top_k_cosine(query, gallery, 3)
    _show('indices', indices)
    _show('similarities', similarities)
    _show('selected IDs', ids[indices])

    print('\nNP-M3: all pairwise distances')
    A, B = np.array([[.20, -.40], [.80, .30], [-.50, .70]]), np.array([[.10, .20], [-.60, -.20]])
    print('broadcast difference shape:', (A[:, None, :] - B[None, :, :]).shape)
    distances = pairwise_distances(A, B)
    _show('D', distances)
    _show('nearest B indices', distances.argmin(axis=1))

    print('\nNP-M4: linear -> ReLU -> linear (provided parameters)')
    X = np.array([[.80, -.60, .70], [-.40, -.50, .90]])
    W1, b1 = np.array([[-.10, .20, -.80], [.30, -.50, .10]]), np.array([-.15, .05])
    W2, b2 = np.array([[.20, -.60], [-.80, .52]]), np.array([.10, -.20])
    Z = X @ W1.T + b1
    H = np.maximum(Z, 0.0)
    Y = H @ W2.T + b2
    for name, value in (('Z', Z), ('H', H), ('Y', Y)):
        _show(name, value)
    _boundary_checks()


if __name__ == '__main__':
    main()
