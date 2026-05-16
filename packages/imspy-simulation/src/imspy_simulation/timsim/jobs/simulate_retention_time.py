from typing import Optional

import pandas as pd

from imspy_predictors.rt.predictors import (
    DeepChromatographyApex,
    load_deep_retention_time_predictor,
    predict_retention_time_with_koina,
)
from imspy_predictors.utility import load_tokenizer_from_resources


def simulate_retention_times(
        peptides: pd.DataFrame,
        verbose: bool = False,
        gradient_length: float = 60 * 60,
        use_koina_model: Optional[str] = None,
        rt_model_kind: str = "local",
        chronologer_base_path: Optional[str] = None,
) -> pd.DataFrame:
    """Simulate retention times.
    Args:
        peptides: Peptides DataFrame.
        verbose: Verbosity.
        gradient_length: Gradient length in seconds.
        use_koina_model: Model name for Koina retention time prediction, if None, use DeepChromatographyApex.
        rt_model_kind: "local" (default — DeepChromatographyApex, optionally
            via Koina) or "chronologer" (opt-in local Chronologer RT
            predictor). The default keeps the published benchmark RT axis
            engine-neutral; "chronologer" is a development / debugging mode
            where the RT axis is matchable by a Chronologer-based search
            engine. NOTE: an FDP measured on a Chronologer-RT sim is not a
            calibrated number — a Chronologer-using engine matches that RT
            axis by construction. Use it for correctness / feature-behaviour
            testing, not FDR claims.
        chronologer_base_path: Path to the upstream Chronologer base .pt;
            required when rt_model_kind == "chronologer".

    Returns:
        pd.DataFrame: Peptides DataFrame with a ``retention_time_gru_predictor``
            column (RT in seconds, spanning [0, gradient_length]).
    """

    if verbose:
        print("Simulating retention times ...")

    if rt_model_kind == "chronologer":
        import numpy as np
        from imspy_predictors.rt.chronologer import Chronologer

        if not chronologer_base_path:
            raise ValueError(
                "rt_model_kind='chronologer' requires chronologer_base_path "
                "(path to the upstream Chronologer base .pt)")
        if verbose:
            print(f"Using local Chronologer RT model "
                  f"(base={chronologer_base_path})")

        chrono = Chronologer.from_base(base_model_path=str(chronologer_base_path))
        out = peptides.copy()
        seqs = out["sequence_modified"].astype(str).tolist()
        rt_min = np.asarray(chrono.simulate_separation_times(seqs), dtype=float)

        finite = np.isfinite(rt_min)
        if int(finite.sum()) < 2:
            raise RuntimeError(
                "Chronologer returned <2 finite retention times; cannot "
                "scale to the gradient.")
        # Linear-map the finite (minute-scale) predictions onto
        # [0, gradient_length] seconds — mirrors the DeepChromatographyApex
        # path so every downstream consumer is untouched.
        lo, hi = float(np.nanmin(rt_min)), float(np.nanmax(rt_min))
        rt_s = (rt_min - lo) / (hi - lo) * float(gradient_length)
        n_unsupported = int((~finite).sum())
        if n_unsupported:
            if verbose:
                print(f"  {n_unsupported} sequence(s) unsupported by "
                      f"Chronologer; assigning uniform-random in-gradient RT")
            rng = np.random.default_rng(42)
            rt_s[~finite] = rng.uniform(0.0, float(gradient_length),
                                        n_unsupported)
        out["retention_time_gru_predictor"] = rt_s
        return out

    if use_koina_model is not None and use_koina_model != "":
        try:
            if verbose:
                print(f"Using Koina model: {use_koina_model}")

            peptide_rt = predict_retention_time_with_koina(
                model_name=use_koina_model,
                data=peptides,
                gradient_length=gradient_length,
            )

        except Exception as e:
            print(f"Failed to predict retention time with KOINA: {e}")
            print("Falling back to DeepChromatographyApex model ...")

            RTColumn = DeepChromatographyApex(
                model=load_deep_retention_time_predictor(),
                tokenizer=load_tokenizer_from_resources(tokenizer_name="tokenizer-ptm"),
                verbose=verbose,
            )

            peptide_rt = RTColumn.simulate_separation_times_pandas(
                data=peptides,
                gradient_length=gradient_length,
            )

    else:
        # create RTColumn instance
        RTColumn = DeepChromatographyApex(
            model=load_deep_retention_time_predictor(),
            tokenizer=load_tokenizer_from_resources(tokenizer_name="tokenizer-ptm"),
            verbose=verbose,
        )

        # predict rts
        peptide_rt = RTColumn.simulate_separation_times_pandas(
            data=peptides,
            gradient_length=gradient_length,
        )

    return peptide_rt