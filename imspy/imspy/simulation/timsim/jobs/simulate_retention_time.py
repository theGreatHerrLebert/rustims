from typing import Optional
import numpy as np
import pandas as pd

# Optional dependency for truncated‐normal; if not installed, we'll fall back to np.clip
try:
    from scipy.stats import truncnorm
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

from imspy.algorithm.rt.predictors import (
    DeepChromatographyApex,
    load_deep_retention_time_predictor,
    predict_retention_time_with_koina,
)
from imspy.algorithm.utility import load_tokenizer_from_resources


def simulate_retention_times(
    peptides: pd.DataFrame,
    *,
    gradient_length: float = 3600.0,
    use_koina_model: Optional[str] = None,
    rt_noise_std: Optional[float] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Simulate retention times, optionally using a Koina model or falling back to DeepChromatographyApex,
    then (if requested) add Gaussian noise with hard or truncated clipping.

    Parameters
    ----------
    peptides
        DataFrame of peptides (must include whatever format your RT predictors expect).
    gradient_length
        Total gradient length in seconds.
    use_koina_model
        Name of a KOINA RT model to use; if None or empty, skips KOINA.
    rt_noise_std
        Standard deviation (in seconds) of Gaussian noise to add; if None, no noise.
    verbose
        Whether to print progress messages.

    Returns
    -------
    DataFrame
        The input DataFrame augmented with a "retention_time" column.
    """

    if verbose:
        print("🔬 Simulating retention times…")

    # 1) Predict with KOINA if requested, else use DeepChromatographyApex
    if use_koina_model:
        if verbose:
            print(f"   ▶ Attempting KOINA model '{use_koina_model}'")
        try:
            rt_df = predict_retention_time_with_koina(
                model_name=use_koina_model,
                data=peptides,
                gradient_length=gradient_length,
            )
        except Exception as e:
            if verbose:
                print(f"   ⚠️  KOINA failed ({e}); falling back to DeepChromatographyApex")
            apex = DeepChromatographyApex(
                model=load_deep_retention_time_predictor(),
                tokenizer=load_tokenizer_from_resources("tokenizer-ptm"),
                verbose=verbose,
            )
            rt_df = apex.simulate_separation_times_pandas(
                data=peptides,
                gradient_length=gradient_length,
            )
    else:
        if verbose:
            print("   ▶ Using DeepChromatographyApex")
        apex = DeepChromatographyApex(
            model=load_deep_retention_time_predictor(),
            tokenizer=load_tokenizer_from_resources("tokenizer-ptm"),
            verbose=verbose,
        )
        rt_df = apex.simulate_separation_times_pandas(
            data=peptides,
            gradient_length=gradient_length,
        )

    # 2) Optionally add noise
    if rt_noise_std is not None:
        # identify the retention time column
        rt_cols = [c for c in rt_df.columns if "retention_time" in c]
        if not rt_cols:
            raise ValueError("No 'retention_time' column found in predictor output.")
        rt_col = rt_cols[0]

        original = rt_df[rt_col].to_numpy()
        lower, upper = 0.0, float(original.max())

        if verbose:
            print(f"   ▶ Adding Gaussian noise (std={rt_noise_std}s)")

        if _HAS_SCIPY:
            # truncated-normal draw ensures no values outside [lower, upper]
            a, b = (lower - original) / rt_noise_std, (upper - original) / rt_noise_std
            noise = truncnorm(a, b, loc=0.0, scale=rt_noise_std).rvs(size=original.shape)
            noisy = original + noise
        else:
            # simple hard clip if scipy is unavailable
            noisy = np.clip(
                original + np.random.normal(0.0, rt_noise_std, size=original.shape),
                lower,
                upper,
            )

        rt_df[rt_col] = noisy

    return rt_df