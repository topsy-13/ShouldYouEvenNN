"""
Creates a condensed table of key results per experiment
for reporting and appendix use.
"""

import pandas as pd
from pathlib import Path


def build_master_table(csv_path="./experiments/testing/final_results/summary/all_experiments.csv",
                       out_path="analysis_outputs/master_results_table.csv"):
    df = pd.read_csv(csv_path)

    # --- Choose the most crucial columns ---
    keep_cols = [
        "exp_id", "data_id", "seed",
        "naml_best_validation_metric",
        'naml_test_accuracy',
        "ebe_decision", "ebe_p_above_goal", 
        "ebe_top_forecast_val_acc",
        "ebe_val_acc", "ebe_test_acc",
        "ebe_surpassed_val_baseline", "ebe_surpassed_test_baseline",
        "ebe_time_used_s",
    ]

    # Keep only available ones (safe subset)
    keep_cols = [c for c in keep_cols if c in df.columns]
    master = df[keep_cols].copy()

    # --- Optional rounding for clarity ---
    numeric_cols = master.select_dtypes(include="number").columns
    master[numeric_cols] = master[numeric_cols].round(4)

    # --- Sort by exp_id for readability ---
    master = master.sort_values(by="exp_id").reset_index(drop=True)

    # --- Export both CSV and LaTeX ---
    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    master.to_csv(out_path, index=False)

    latex_path = out_dir / "master_results_table.tex"
    master.to_latex(
        latex_path,
        index=False,
        float_format="%.4f",
        caption="Condensed EBE-NAS Experiment Results per Dataset",
        label="tab:master_results",
    )

    print(f"Master table saved to:\n  CSV:  {out_path}\n  LaTeX: {latex_path}")
    return master


if __name__ == "__main__":
    master_df = build_master_table()
    print(master_df.head())
