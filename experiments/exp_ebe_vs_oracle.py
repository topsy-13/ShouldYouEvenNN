# region Imports
import os
import sys
import json

import time
import torch

sys.path.append(os.path.abspath("./src"))

import generations
import search_space
import data_preprocessing as dp
import baseline_models as bm

from instance_sampling import create_dataloaders
from architecture_generator import create_model_from_row
# endregion


def main(data_id=54, seed=13, naive_models=50, 
         strategy='mean', n_individuals=100, 
         n_max_epochs=100, n_top_models_to_train=10, 
         budget_factor=3, PERCENTILE_DROP=15):

    # region Set the scenario
    DATA_ID = data_id
    SEED = seed
    NAIVE_MODELS = naive_models  # for naiveautoml
    STRATEGY = strategy # Mean as baseline
    BUDGET_FACTOR = budget_factor  # EBE gets MULTIPLIED the time of the baseline
    N_INDIVIDUALS = n_individuals  # Individuals per generation
    N_MAX_EPOCHS = n_max_epochs  # Max epochs for EBE, honestly not that important
    N_TOP_MODELS_TO_TRAIN = n_top_models_to_train # For full training after EBE
    PERCENTILE_DROP = 15 # Starting point to drop 

    X_train, y_train, X_val, y_val, X_test, y_test = dp.get_preprocessed_data(
        dataset_id=DATA_ID,
        scaling=True,
        random_seed=SEED,
        return_as='tensor',
        task_type='classification'
        )
    X_combined = torch.cat([X_train, X_val], dim=0)
    y_combined = torch.cat([y_train, y_val], dim=0)

    # NaiveAutoML baseline
    # ! This is showing reproducibility issues, it appears NaiveAutoML is not fully deterministic even with fixed seed
    print('Getting NaiveAutoML baseline...') 
    baseline_metric, time_budget, naml_scoreboard, best_model = bm.get_models_and_baseline_metric(
        X_train, y_train,
        n_models=NAIVE_MODELS,
        random_state=SEED,
        strategy=STRATEGY
    )  
    naml_scoreboard['pipeline'] = naml_scoreboard['pipeline'].astype(str) # data type for json export

    # Test the best non neural pipeline + model on the test set
    accuracy = best_model.score(X_test, y_test)

    # Get baseline metrics
    strategies = ["best", "median", "mean", "worst"]
    naml_metrics = {
                    strategy: bm.get_baseline_metric(naml_scoreboard, strategy=strategy)
                    for strategy in strategies
                    }
    
    # Store NAML results
    naml_results = {
    "seed": SEED,
    "data_id": DATA_ID,
    "strategy": STRATEGY,
    "n_models": NAIVE_MODELS,
    "time_taken": time_budget,
    "baseline_metric": baseline_metric,
    "metrics": naml_metrics, 
    "test_accuracy_best_non_nn_model": accuracy,
    "scoreboard": naml_scoreboard.to_dict(orient="records"),
    }

    print("Baseline metric:", baseline_metric)
    print("Time for baseline models:", time_budget)


    nn_time_budget = time_budget * BUDGET_FACTOR
    print("Time budget assigned for EBE:", nn_time_budget)

    # Export scoreboard for later analysis
    directory = f'experiments/ebe_vs_oracle/naml_baseline-models'
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, f"{DATA_ID}.json"), 'w') as json_file:
        json.dump(naml_results, json_file, indent=4)
    # endregion


    # region EBE
    print('Starting EBE...')
    ebe_results = {}

    # Same search space for the whole experiment
    input_size, output_size = dp.get_tensor_sizes(X_train, y_train)
    s_space = search_space.SearchSpace(input_size, output_size)
    # The generation is created given the search space and the number of individuals.
    nn_start_time = time.time()
    generation = generations.Generation(s_space, N_INDIVIDUALS, starting_instances=500)
    generation.run_ebe(epochs=N_MAX_EPOCHS,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val, 
                    percentile_drop=PERCENTILE_DROP,
                    baseline_metric=baseline_metric, 
                    time_budget=nn_time_budget,
                    epoch_threshold=3 # for forecasting
                    )
    nn_end_time = time.time()
    nn_taken_time = nn_end_time - nn_start_time
    
    ebe_results_df = generation.history
    ebe_results_df[['activation_fn', 'optimizer_type']] = ebe_results_df[['activation_fn', 'optimizer_type']].astype(str)
    ebe_results_df = ebe_results_df.fillna('None')

    # Store EBE results
    ebe_results = {
        'seed': SEED,
        'data_id': DATA_ID,
        'n_individuals': N_INDIVIDUALS,
        'time_factor VS Naive': BUDGET_FACTOR,
        'n_top_models_to_train': N_TOP_MODELS_TO_TRAIN,
        'time_taken_in_ebe': nn_taken_time,
        'percentile_drop': PERCENTILE_DROP,
        'baseline_metric': baseline_metric
        }

    # Train the best models 
    print('Training the best models by ES...')
    for n_model in range(N_TOP_MODELS_TO_TRAIN):
        print(f'Training model top {n_model+1}/{N_TOP_MODELS_TO_TRAIN}')
        model = create_model_from_row(ebe_results_df.iloc[n_model], 
                                      input_size, output_size)
        
        batch_size = int(ebe_results_df.iloc[n_model]['batch_size'])  # batch size for training the model

        train_loader = create_dataloaders(X=X_train, y=y_train, 
                            batch_size=batch_size)
        val_loader = create_dataloaders(X=X_val, y=y_val, 
                            batch_size=batch_size)
        test_loader = create_dataloaders(X=X_test, y=y_test, 
                            batch_size=batch_size)
        
        starting_time = time.time()
        # Train the model till convergence with early stopping
        final_train_loss, final_train_acc, final_val_loss, final_val_acc, learning_curve = model.es_train(train_loader=train_loader, val_loader=val_loader,
                        es_patience=150, # epochs without improvement
                        max_epochs=1000, # cap for epochs
                        verbose=False, # print training progress
                        return_lc=True # return learning curves
        )
        ending_time = time.time()

        # Evaluate on test set
        test_loss, test_acc = model.evaluate(test_loader)

        # Build new columns
        calculated_metrics = ['training_time_ES', 'final_train_acc', 'final_val_acc', 'final_test_loss', 'final_test_acc', 'learning_curve']
        ebe_results_df.loc[n_model, calculated_metrics] = [
                                                            ending_time - starting_time,
                                                            final_train_acc,
                                                            final_val_acc,
                                                            test_loss,
                                                            test_acc,
                                                            learning_curve
                                                            ]
        ebe_results_df[calculated_metrics] = ebe_results_df[calculated_metrics].fillna(0)
    
    ebe_results['max_test_acc'] = ebe_results_df['final_test_acc'].max()
    ebe_results['scoreboard'] = ebe_results_df.to_dict(orient='records')

    # Export ebe_results for later analysis
    directory = f'experiments/ebe_vs_oracle/ebe-models'
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, f"{DATA_ID}.json"), 'w') as json_file:
        json.dump(ebe_results, json_file, indent=4)
    
    print("EBE results exported.")
    # endregion

    # region Oracle
    print('Starting Oracle...')
    oracle_results = {
        'seed': SEED,
        'data_id': DATA_ID,
        'n_models': N_INDIVIDUALS,
        'time_taken_in_oracle': None,  # to be filled
    }
    # clear GPU memory
    torch.cuda.empty_cache()
    # Same search space that for EBE


    


    # endregion

if __name__ == "__main__":
    main()


#* TODO: Store learning curves of the best models for analysis Done
# TODO: Build the Oracle
# TODO: Compare vs Naive Early Stopping 
# TODO: Omit the Neural models if found by NAML
# TODO: Try to build a json with NAML and EBE results for easy comparison