import os
import sys
import json

import time

# region Imports
sys.path.append(os.path.abspath("./src"))

import generations
import search_space
import data_preprocessing as dp
import baseline_models as bm

from instance_sampling import create_dataloaders
from architecture_generator import create_model_from_row
# endregion


def main():

    # region Set the scenario
    DATA_ID = 54
    SEED = 13
    N_MODELS = 50  # for naiveautoml

    X_train, y_train, X_val, y_val, X_test, y_test = dp.get_preprocessed_data(
        dataset_id=DATA_ID,
        scaling=True,
        random_seed=SEED,
        return_as='tensor',
        task_type='classification'
    )



    # NaiveAutoML baseline
    # ! This is showing reproducibility issues, it appears NaiveAutoML is not fully deterministic even with fixed seed
    print('Getting NaiveAutoML baseline...') 
    naml_results = {}
    baseline_metric, time_budget, naml_scoreboard, best_model = bm.get_models_and_baseline_metric(
        X_train, y_train,
        n_models=N_MODELS,
        random_state=SEED,
        strategy='mean'
    )  # Mean as baseline

    # Test the best non neural pipeline + model on the test set
    accuracy = best_model.score(X_test, y_test)

    max_metric = bm.get_baseline_metric(naml_scoreboard, strategy='best')
    median_metric = bm.get_baseline_metric(naml_scoreboard, strategy='median')
    worst_metric = bm.get_baseline_metric(naml_scoreboard, strategy='worst')

    naml_scoreboard['pipeline'] = naml_scoreboard['pipeline'].astype(str)
    naml_results['seed'] = SEED
    naml_results['data_id'] = DATA_ID
    naml_results['time_taken'] = time_budget
    naml_results['baseline_metric'] = baseline_metric
    naml_results['max_metric'] = max_metric
    naml_results['median_metric'] = median_metric
    naml_results['worst_metric'] = worst_metric
    naml_results['test_accuracy_of_best_nonNN_model'] = accuracy
    naml_results['scoreboard'] = naml_scoreboard.to_dict(orient='records')

    print("Baseline metric:", baseline_metric)
    print("Time for baseline models:", time_budget)

    BUDGET_FACTOR = 3  # EBE gets MULTIPLIED the time of the baseline
    nn_time_budget = time_budget * BUDGET_FACTOR
    print("Time budget for EBE:", nn_time_budget)

    # Export scoreboard for later analysis
    directory = f'experiments/ebe_vs_oracle/naml_baseline-models'
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, f"{DATA_ID}.json"), 'w') as json_file:
        json.dump(naml_results, json_file, indent=4)

    # endregion

    # region EBE
    print('Starting EBE...')
    ebe_results = {}
    N_INDIVIDUALS = 100
    N_MAX_EPOCHS = 100
    percentile_drop = 15

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
                    percentile_drop=percentile_drop,
                    baseline_metric=baseline_metric, 
                    time_budget=nn_time_budget,
                    epoch_threshold=3 # for forecasting
                    )
    nn_end_time = time.time()
    nn_taken_time = nn_end_time - nn_start_time
    
    ebe_results_df = generation.return_df()
    ebe_results_df[['activation_fn', 'optimizer_type']] = ebe_results_df[['activation_fn', 'optimizer_type']].astype(str)
    ebe_results_df = ebe_results_df.fillna('None')

    ebe_results['seed'] = SEED
    ebe_results['data_id'] = DATA_ID
    ebe_results['baseline_metric'] = baseline_metric
    ebe_results['n_individuals'] = N_INDIVIDUALS
    ebe_results['time_taken_in_ebe'] = nn_taken_time
    ebe_results['time_factor VS Naive'] = BUDGET_FACTOR
    
    # Train the best models 
    print('Training the best models by ES...')
    N_MODELS_TO_TRAIN = 10
    for n_model in range(N_MODELS_TO_TRAIN):
        print(f'Training model {n_model+1}/{N_MODELS_TO_TRAIN}')
        starting_time = time.time()
        model = create_model_from_row(ebe_results_df.iloc[n_model], 
                                      input_size, output_size)
        
        batch_size = int(ebe_results_df.iloc[n_model]['batch_size'])  # batch size for training the model

        train_loader = create_dataloaders(X=X_train, y=y_train, 
                            batch_size=batch_size)
        val_loader = create_dataloaders(X=X_val, y=y_val, 
                            batch_size=batch_size)
        test_loader = create_dataloaders(X=X_test, y=y_test, 
                            batch_size=batch_size)
        
        # Train the model
        final_train_loss, final_train_acc, final_val_loss, final_val_acc = model.es_train(train_loader=train_loader, val_loader=val_loader,
                        es_patience=100, # epochs without improvement
                        max_epochs=1000, # cap for epochs
                        verbose=False, # print training progress
        )
        ending_time = time.time()
        ebe_results_df.loc[n_model, 'time_to_train_byES'] = ending_time - starting_time
        # Build dataframe with new column
        ebe_results_df.loc[n_model, 'final_train_acc'] = final_train_acc
        ebe_results_df.loc[n_model, 'final_val_acc'] = final_val_acc

        # Evaluate on test set
        test_loss, test_acc = model.evaluate(test_loader)
        ebe_results_df.loc[n_model, 'final_test_acc'] = test_acc

        ebe_results_df[['final_train_acc', 'final_val_acc', 'final_test_acc', 'time_to_train_byES']] = ebe_results_df[['final_train_acc', 'final_val_acc', 'final_test_acc', 'time_to_train_byES']].fillna(0)
    
    ebe_results['max_test_acc'] = ebe_results_df['final_test_acc'].max()
    ebe_results['scoreboard'] = ebe_results_df.to_dict(orient='records')

    # Export ebe_results for later analysis
    directory = f'experiments/ebe_vs_oracle/ebe-models'
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, f"{DATA_ID}-2.json"), 'w') as json_file:
        json.dump(ebe_results, json_file, indent=4)
    
    print("EBE results exported.")
    # endregion

    # region Oracle
    print('Starting Oracle...')
    # endregion

if __name__ == "__main__":
    main()
