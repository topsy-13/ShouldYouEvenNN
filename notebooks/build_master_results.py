"""
Creates a condensed table of key results per experiment
for reporting and appendix use.
"""

import pandas as pd
from pathlib import Path


def build_master_table(
    csv_path="./experiments/testing/final_results/summary/all_experiments.csv",
    out_path="analysis_outputs/master_results_table.csv",
):
    df = pd.read_csv(csv_path)

    # --- Select key columns ---
    keep_cols = [
        'data_id',
        "naml_best_validation_metric", "naml_test_accuracy",
        "ebe_decision",
        "ebe_val_acc", "ebe_test_acc",
        "ebe_surpassed_val_baseline", "ebe_surpassed_test_baseline",
        "ebe_time_used_s",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]

    master = df[keep_cols].copy()

    # --- Round numeric columns for readability ---
    numeric_cols = master.select_dtypes(include="number").columns
    master[numeric_cols] = master[numeric_cols].round(4)

    # --- Sort by exp_id ---
    master = master.sort_values(by="data_id").reset_index(drop=True)

    # --- Prepare output directories ---
    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Export CSV ---
    master.to_csv(out_path, index=False)

    # --- Export LaTeX (escaped, safe for direct \input{}) ---
    latex_path = out_dir / "master_results_table.tex"
    master.to_latex(
        latex_path,
        index=False,
        float_format="%.4f",
        caption="Condensed EBE-NAS Experiment Results per Dataset",
        label="tab:master_results",
        escape=True,  # <-- key fix: escapes underscores & special chars
        longtable=True
    )

    print(f"Master table saved to:\n  CSV:  {out_path}\n  LaTeX: {latex_path}")
    return master


if __name__ == "__main__":
    master_df = build_master_table()
    print(master_df.head())
