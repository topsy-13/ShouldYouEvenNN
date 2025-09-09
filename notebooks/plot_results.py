import pandas as pd
import json 

import matplotlib.pyplot as plt
import seaborn as sns
import ast
import numpy as np


def load_n_plot(exp_id):
    # Standard MLP Metric

    mlp_path = f'./experiments/ebe_vs/standard_models/standard_MLP'

    with open(f"{mlp_path}/{exp_id}_MLP.json",) as json_file:
            mlp_json = json.load(json_file)
    mlp_metric = mlp_json['test_acc']

    # Standard ML Models
    ml_path = f'./experiments/ebe_vs/standard_models'
    with open(f"{ml_path}/{exp_id}_ML.json",) as json_file:
            ml_json = json.load(json_file)
    ml_metric = ml_json['best_accuracy']

    # NaiveAutoML Models
    naml_path = f'./experiments/ebe_vs/naml_baseline-models'
    with open(f"{naml_path}/{exp_id}_NAML.json",) as json_file:
            naml_json = json.load(json_file)
    naml_metric = naml_json['test_accuracy_best_non_nn_model']

    # oracle models
    # oracle_path = f'../experiments/ebe_vs/oracle'
    # with open(f"{oracle_path}/{exp_id}_Oracle-summary.json",) as json_file:
    #         oracle_json = json.load(json_file)

    # oracle_metric = oracle_json['max_test_acc']
    # oracle_df = pd.read_csv(f'{oracle_path}/{exp_id}_Oracle_all.csv')

    # EBE models
    ebe_path = f'./experiments/ebe_vs/ebe-models'
    with open(f"{ebe_path}/{exp_id}_ES_EBE-summary.json",) as json_file:
            ebe_json = json.load(json_file)

    ebe_metric = ebe_json['max_test_acc']
    ebe_df = pd.read_csv(f'{ebe_path}/{exp_id}_ES_EBE.csv')

    # region Plotting
    for col in ['es_train_losses', 'es_val_losses', 'es_train_accs', 'es_val_accs']:
        def safe_eval(x):
            x_str = str(x)  # force to string
            try:
                return ast.literal_eval(x_str)
            except (ValueError, SyntaxError):
                return x_str  # fallback to string if parsing fails
        ebe_df[col] = ebe_df[col].apply(safe_eval)

    fig, ax = plt.subplots(figsize=(12, 7))

    # Extract the maximum length of each list
    max_lengths = ebe_df['es_val_accs'].apply(len)

    # If you want to get the maximum length across all lists
    xmax = max_lengths.max()

    # Horizontal reference lines + text
    ax.axhline(y=ml_metric, color='r', linestyle='--', linewidth=1)
    ax.text(xmax - 50, ml_metric, "ml_acc", color='r', va='center', ha='left')

    ax.axhline(y=mlp_metric, color='g', linestyle='--', linewidth=1)
    ax.text(xmax - 40, mlp_metric, "mlp_acc", color='g', va='center', ha='left')

    ax.axhline(y=naml_metric, color='b', linestyle='--', linewidth=1)
    ax.text(xmax - 30, naml_metric, "naml_acc", color='b', va='center', ha='left')

    # ax.axhline(y=oracle_metric, color='black', linestyle='-', linewidth=1)
    # ax.text(xmax, oracle_metric, "oracle_acc", color='black', va='center', ha='left')

    ax.axhline(y=ebe_metric, color='orange', linestyle='--', linewidth=2)
    ax.text(xmax+30, ebe_metric, "ebe_acc", color='black', va='center', ha='left', fontweight='bold')

    ax.axvline(x=np.max(ebe_df['epochs_trained']))
    # Find which series has the overall maximum value
    val_accs_original = np.array(ebe_df['es_val_accs'].tolist(), dtype=object)  # Keep original
    max_values = []
    for i in range(len(ebe_df)):
        series = ebe_df['es_test_acc'].iloc[i]  # Use iloc for safer indexing
        max_values.append(np.max(series))

    best_series_idx = np.argmax(max_values)

    # Plot individual learning curves
    for i in range(len(ebe_df)):
        series = ebe_df['es_val_accs'].iloc[i]
        
        if i == best_series_idx:
            ax.plot(series, alpha=1.0, linewidth=2, label='Best EBE model', color='blue')
        else:
            ax.plot(series, alpha=0.2, linewidth=1)

    # NaN padding approach for mean curve (using ORIGINAL data, not truncated)
    max_len = max(len(x) for x in val_accs_original)

    padded_val_accs = []
    for acc_seq in val_accs_original:  # Use original data
        padded = np.full(max_len, np.nan)
        padded[:len(acc_seq)] = acc_seq
        padded_val_accs.append(padded)

    val_accs_padded = np.array(padded_val_accs)

    # Calculate mean and std, ignoring NaN values
    mean_curve = np.nanmean(val_accs_padded, axis=0)
    std_curve = np.nanstd(val_accs_padded, axis=0)

    # Plot mean curve with error bands
    epochs = range(len(mean_curve))
    ax.plot(epochs, mean_curve, color='purple', label='EBE Mean after ES', linewidth=2, alpha=0.7)
    ax.fill_between(epochs, mean_curve-std_curve, mean_curve+std_curve, color='purple', alpha=0.3)

    # Rest of your plotting code...
    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("Validation Accuracy", fontsize=14)
    fig.suptitle("Learning Curves by ES vs Benchmarkings", fontsize=14, fontweight='bold')
    ax.set_title(f"for {dataset_name} dataset", fontsize=12, color="gray")
    ax.grid(alpha=0.3)
    ax.set_ylim(0.4, 1)
    plt.tight_layout()
    plt.legend()
    plt.savefig(f'./figures/Experiment Results/{exp_id}.png')
    plt.show()

    # endregion

# load_n_plot('54_13')


if __name__ == '__main__':
    # Datasets to test
    SEED = 14125
    dataset_ids_path = 'experiments/datasets/openml_datasets.json'
    with open(dataset_ids_path) as f:
        dataset_ids = json.load(f)
    for dataset_name, data_id in dataset_ids.items():
        try:
            load_n_plot(exp_id=f'{data_id}_{SEED}')
        except FileNotFoundError:
            print(f"File not found for dataset {dataset_name} (ID: {data_id})")
            # or just: pass
        except Exception as e:
            print(f"Error processing dataset {dataset_name}: {e}")
            # or just: pass
        plt.close()
        
#     print('There is more hope')