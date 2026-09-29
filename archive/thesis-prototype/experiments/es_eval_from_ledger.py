import pandas as pd
import time
import json
import sys
import os

sys.path.append(os.path.abspath("./src"))

import data_preprocessing as dp
from architecture_generator import create_model_from_row
from utils import set_seed


def evaluate_from_ledger(ebe_results: pd.DataFrame,
                         data_id: int,
                         seed: int,
                         top_fraction: float = 0.2,
                         export_path: str = "./experiments/ebe_vs/v2/ebe"):
    """
    Take an EBE ledger (DataFrame) and trains and evaluate the top models on the test set.

    Args:
        ebe_results: DataFrame with architectures & metrics from EBE.
        data_id: OpenML dataset ID.
        seed: Random seed used for data split.
        top_fraction: fraction of top models (by score) to evaluate.
        export_path: where to save results.

    Returns:
        dict with max test accuracy and summary metrics.
    """
    # Load dataset
    X_train, y_train, X_val, y_val, X_test, y_test = dp.get_preprocessed_data(
        dataset_id=data_id,
        scaling=True,
        random_seed=seed,
        return_as="tensor",
        task_type="classification"
    )
    input_size, output_size = dp.get_tensor_sizes(X_train, y_train)
    print('Data loaded')
    # Select top candidates
    n_top = max(1, int(len(ebe_results) * top_fraction))
    top_models = ebe_results.head(n_top)

    es_results = []
    total_time = 0
    best_test_acc = 0.0
    best_model_id = None

    for i, (_, model_row) in enumerate(top_models.iterrows()):
        # Print the n model being trained
        print(f'Training model {i + 1} of {n_top}')
        model_id = int(model_row["id"])
        model_seed = int(model_row.get("arch_seed", seed))
        batch_size = int(model_row.get("batch_size", 32))

        g, seed_worker = set_seed(model_seed)

        # Build loaders
        train_loader = dp.create_dataloader(X=X_train, y=y_train,
                                            batch_size=batch_size,
                                            generator=g, seed_worker=seed_worker)
        val_loader = dp.create_dataloader(X=X_val, y=y_val,
                                          batch_size=batch_size,
                                          generator=g, seed_worker=seed_worker)
        test_loader = dp.create_dataloader(X=X_test, y=y_test,
                                           batch_size=batch_size,
                                           generator=g, seed_worker=seed_worker)

        # Rebuild model from ledger row
        model = create_model_from_row(model_row, input_size, output_size)

        # Train with Early Stopping + Learning curve
        start = time.time()
        es_train_loss, es_train_acc, es_val_loss, es_val_acc, lc = model.es_train(
            train_loader, val_loader,
            es_patience=150, max_epochs=1000,
            verbose=False, return_lc=True
        )
        end = time.time()
        total_time += end - start

        # Test evaluation
        test_loss, test_acc = model.evaluate(test_loader)

        record = {
            "id": model_id,
            "train_acc": es_train_acc,
            "train_loss": es_train_loss,
            "val_acc": es_val_acc,
            "val_loss": es_val_loss,
            "test_acc": test_acc,
            "test_loss": test_loss,
            # Save the full learning curve as lists
            "lc_train_accs": lc.get("es_train_accs", []),
            "lc_val_accs": lc.get("es_val_accs", []),
            "lc_train_losses": lc.get("es_train_losses", []),
            "lc_val_losses": lc.get("es_val_losses", [])
        }
        es_results.append(record)

        if test_acc and test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_id = model_id

    # Convert to DataFrame
    es_results_df = pd.DataFrame(es_results)

    # Summary
    summary = {
        "seed": seed,
        "data_id": data_id,
        "training_time": total_time,
        "max_test_acc": es_results_df["test_acc"].max(),
        "min_test_loss": es_results_df["test_loss"].min(),
        "best_model_id": best_model_id
    }

    # Export
    exp_id = f"{data_id}_{seed}"
    es_results_df.to_csv(f"{export_path}/{exp_id}_ES_EBE.csv", index=False)
    with open(f"{export_path}/{exp_id}_ES_EBE-summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    print("=== Final Test Evaluation ===")
    print(summary)

    return summary

import json
import os
if __name__ == "__main__":
    # Datasets to test
    SEED = 14125
    dataset_ids_path = 'experiments/datasets/openml_datasets.json'
    with open(dataset_ids_path) as f:
        dataset_ids = json.load(f)
    omit_ids = [1111]
    
    crashed = {}

    for dataset_name, data_id in dataset_ids.items():
        print(f"Starting dataset {data_id} - {dataset_name}")
        if data_id not in omit_ids:
            try:
                ebe_results = pd.read_csv(f'./experiments/ebe_vs/v2/ebe/{data_id}_{SEED}_EBE.csv') 
                
                evaluate_from_ledger(ebe_results,
                         data_id=data_id,
                         seed=SEED,
                         top_fraction= 0.2
                         )
            except Exception as e:
                crashed[data_id] = str(e)
        else:
            pass

    # export all crashes into a single JSON
    with open("crashed_datasets.json", "w") as f:
        json.dump(crashed, f, indent=4)

    print('There is hope')
    os.system("shutdown /s /t 30")