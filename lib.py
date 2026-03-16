"""Analysis utilities: supervised projections of residual stream to belief states."""

import numpy as np
from sklearn.linear_model import Ridge
from mess3 import Mess3


def compute_ground_truth_beliefs(tokens, instance_ids, instances):
    """Compute ground-truth predictive vectors for all sequences.

    Returns: (N, seq_len, 3) array of belief states.
    """
    N, seq_len = tokens.shape
    beliefs = np.empty((N, seq_len, 3))
    for i, inst in enumerate(instances):
        mask = instance_ids == i
        beliefs[mask] = inst.predictive_vectors(tokens[mask])
    return beliefs


def fit_supervised_projection(H, beliefs, instance_ids, per_instance=True):
    """Fit linear map from residual stream to ground-truth belief states.

    H: (N, seq_len, d_model)
    beliefs: (N, seq_len, 3)
    per_instance: if True, fit one Ridge per instance. If False, fit a single global one.

    Returns: function proj(H_new, instance_ids_new) -> (N, seq_len, 3) predicted beliefs,
             and the fitted model(s).
    """
    N, seq_len, d_model = H.shape
    if per_instance:
        models = {}
        for i in np.unique(instance_ids):
            mask = instance_ids == i
            X = H[mask].reshape(-1, d_model)
            Y = beliefs[mask].reshape(-1, 3)
            m = Ridge(alpha=1.0).fit(X, Y)
            models[i] = m

        def proj(H_new, ids_new):
            out = np.empty((H_new.shape[0], H_new.shape[1], 3))
            for i in np.unique(ids_new):
                mask = ids_new == i
                out[mask] = models[i].predict(H_new[mask].reshape(-1, d_model)).reshape(-1, H_new.shape[1], 3)
            return out
    else:
        X = H.reshape(-1, d_model)
        Y = beliefs.reshape(-1, 3)
        model = Ridge(alpha=1.0).fit(X, Y)
        models = model

        def proj(H_new, ids_new=None):
            return model.predict(H_new.reshape(-1, d_model)).reshape(H_new.shape[0], H_new.shape[1], 3)

    return proj, models


def project_to_belief_2d(H, beliefs, instance_ids, per_instance=True):
    """Project residual stream to 2D via supervised readout to belief states.

    Since beliefs live on a 2-simplex (3 components summing to 1), the effective
    dimensionality is 2. We return the first 2 components of the predicted belief.

    Returns: (N*seq_len, 2) projected coordinates.
    """
    proj_fn, _ = fit_supervised_projection(H, beliefs, instance_ids, per_instance=per_instance)
    pred = proj_fn(H, instance_ids)  # (N, seq_len, 3)
    # Drop 3rd component (redundant since they sum to ~1)
    return pred[:, :, :2]
