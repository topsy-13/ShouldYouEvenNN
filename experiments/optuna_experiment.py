import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import sys
import data_preprocessing as dp
from instance_sampling import create_dataloaders

# Example training/evaluation loop
def train_and_evaluate(model, optimizer, criterion, train_loader, val_loader, device):
    model.to(device)
    model.train()
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        preds = model(batch_x)
        loss = criterion(preds, batch_y)
        loss.backward()
        optimizer.step()

    # quick validation pass
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            preds = model(batch_x)
            _, predicted = torch.max(preds, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

    return correct / total


# Define objective function for Optuna

def objective(trial):
    global X_train, y_train, X_val, y_val
    
    # Architecture
    n_layers = trial.suggest_int("n_layers", 2, 100)
    n_neurons = trial.suggest_int("n_neurons", 3, 500)
    activation_fn = trial.suggest_categorical("activation_fn", 
                        [nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh, nn.ELU, nn.GELU])
    dropout_rate = trial.suggest_categorical("dropout_rate", [0, 0.1, 0.2, 0.3, 0.4, 0.5])
    layer_norm = trial.suggest_categorical("layer_norm", [True, False])
    skip_conn = trial.suggest_categorical("skip_conn", [True, False])
    initializer = trial.suggest_categorical("initializer", 
                        ["xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal"])

    # Training hyperparams
    lr = trial.suggest_float("learning_rate", 0.0001, 0.1, log=True)
    batch_size = trial.suggest_int("batch_size", 32, 1024, log=True)
    weight_decay = trial.suggest_categorical("weight_decay", [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
    momentum = trial.suggest_categorical("momentum", [0.8, 0.9, 0.95, 0.99])
    lr_scheduler = trial.suggest_categorical("lr_scheduler", ["step", "exponential", "cosine", "none"])

    # Build model
    layers = []
    in_features = input_size
    for _ in range(n_layers):
        layers.append(nn.Linear(in_features, n_neurons))
        
        # init
        if initializer == "xavier_uniform":
            nn.init.xavier_uniform_(layers[-1].weight)
        elif initializer == "xavier_normal":
            nn.init.xavier_normal_(layers[-1].weight)
        elif initializer == "kaiming_uniform":
            nn.init.kaiming_uniform_(layers[-1].weight, nonlinearity="relu")
        elif initializer == "kaiming_normal":
            nn.init.kaiming_normal_(layers[-1].weight, nonlinearity="relu")

        if layer_norm:
            layers.append(nn.LayerNorm(n_neurons))
        layers.append(activation_fn())
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        in_features = n_neurons

    layers.append(nn.Linear(in_features, output_size))
    model = nn.Sequential(*layers)

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    criterion = nn.CrossEntropyLoss()

    # Data (replace with your train/val split)
    train_loader = create_dataloaders(X=X_train, y=y_train, 
                            batch_size=batch_size)
    val_loader = create_dataloaders(X=X_val, y=y_val, 
                                batch_size=batch_size)

    # Training loop
    acc = train_and_evaluate(model, optimizer, criterion, train_loader, val_loader, device="cpu")

    return acc


# Run study
n_trials = 820  # equal to the number of models EBE evaluated

# Load data (yes, again)
X_train, y_train, X_val, y_val, X_test, y_test = dp.get_preprocessed_data(
    dataset_id=54,
    scaling=True,
    random_seed=13,
    return_as='tensor',
    task_type='classification'
    )


input_size, output_size = dp.get_tensor_sizes(X_train, y_train)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=n_trials)

print("Best trial:", study.best_trial.params)
print("Best accuracy:", study.best_trial.value)
