"""
optimize_hyperparams.py
-----------------------
This script uses the Optuna framework to perform an automated hyperparameter search
for the Combined_pH_PINN model defined in pH_PINN.py (raw-data, pH-only setup).

Run from project root:
    python -m src.ph_pinn.optimize_hyperparams --objective val_loss
    or
    python -m src.ph_pinn.optimize_hyperparams --objective h_loss

View results:
    optuna-dashboard "sqlite:///results/PINN Data/optimize_hyperparams.db"

IMPORTANT ABOUT `val_loss`:
    This script creates its **own** random shuffle and train/validation split
    (see `main()` below). Your main training script also creates its **own**
    random shuffle/split. If you optimize for `val_loss` here, but later train
    a model in a different script with a different shuffle/split or slightly
    different batching logic, the validation loss you see there may **not**
    match the value Optuna found here. To get an identical number, you must
    reuse the same permutation/split and batching.
"""
import os
import sys
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import optuna
import argparse

# Ensure the script can find other modules in the project
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# --- Import components from your current training code (raw-data pH-only) ---
from src.hr_model.model import DEFAULT_PARAMS
from src.ph_pinn.pH_PINN import (
    Combined_pH_PINN,
    generate_data,
    train_step,
    evaluate_model,  # wrapper that calls loss_fn_ph
)

# JAX configuration
jax.config.update("jax_enable_x64", True)


def objective(trial, epochs_per_trial, static_data, objective_metric):
    """
    Optuna objective: train for a fixed number of epochs and return best metric.

    Notes on reproducibility vs. your main training script:
        - This objective uses the pre-shuffled, pre-split data stored in
          `static_data` (see main()).
        - It also uses floor division for the number of batches and therefore
          may drop the last partial batch.
        - If a different script later trains with the *same hyperparameters*
          but a *different* shuffle/split/batching policy, its final val loss
          can differ from the number Optuna reports here — especially when
          `objective_metric='val_loss'`.
    """
    # --- 1) Suggest Hyperparameters ---
    master_key = static_data['master_key']
    model_key, epoch_key = jax.random.split(master_key)

    # Network widths/depths (Hamiltonian / R / J); StateNN removed
    h_width = trial.suggest_categorical("h_width", [32, 64, 128, 256, 512, 1024, 2048])
    h_depth = trial.suggest_int("h_depth", 1, 6)
    d_width = trial.suggest_categorical("d_width", [2, 4, 8, 16, 32, 64, 128, 256])
    d_depth = trial.suggest_int("d_depth", 1, 6)
    j_width = trial.suggest_categorical("j_width", [2, 4, 8, 16, 32, 64, 128, 256])
    j_depth = trial.suggest_int("j_depth", 1, 6)

    epsilon = trial.suggest_float("epsilon", 0.001, 5.0)

    # Optimizer
    initial_learning_rate = trial.suggest_float("initial_learning_rate", 1e-5, 1e-2, log=True)
    decay_steps = trial.suggest_int("decay_steps", 1, 1000)

    # Training & Loss (no warmup, no phys_res)
    batch_size = trial.suggest_categorical("batch_size", [1250, 2500, 5000, 10000, 20000, 40000])
    lambda_conservative_max = trial.suggest_float("lambda_conservative_max", 0, 1)
    lambda_dissipative_max  = trial.suggest_float("lambda_dissipative_max",  0, 1)
    lambda_physics_max      = trial.suggest_float("lambda_physics_max",      0, 1)
    lambda_j_structure_max  = trial.suggest_float("lambda_j_structure_max",  0, 1)
    lambda_r_structure_max  = trial.suggest_float("lambda_r_structure_max",  0, 1)

    # --- 2) Build Model & Optimizer ---
    nn_config = {
        "hamiltonian_net": {"hidden_sizes": [h_width] * h_depth, "epsilon": epsilon},
        "dissipation_net": {"hidden_sizes": [d_width] * d_depth},
        "j_net":           {"hidden_sizes": [j_width] * j_depth},
        # Activation mirrors the training code
        "activation": jax.nn.tanh,
    }

    model = Combined_pH_PINN(key=model_key, config=nn_config, state_dim=static_data['e_dim'])

    lr_schedule = optax.linear_schedule(
        init_value=initial_learning_rate, end_value=1e-5, transition_steps=decay_steps
    )
    optimizer = optax.adamw(learning_rate=lr_schedule)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # Fixed λs (no warmup)
    lambdas = {
        "lambda_conservative": lambda_conservative_max,
        "lambda_dissipative":  lambda_dissipative_max,
        "lambda_physics":      lambda_physics_max,
        "lambda_j_structure":  lambda_j_structure_max,
        "lambda_r_structure":  lambda_r_structure_max,
    }

    # --- 3) Train Loop ---
    best_val_loss = jnp.inf
    best_train_h_loss = jnp.inf

    # NOTE: here we use floor division -> the last partial batch (if any) is dropped.
    # Your main training script uses ceil division and trains on all samples.
    num_batches = static_data['e_train'].shape[0] // batch_size
    if num_batches == 0:
        num_batches = 1

    for _epoch in range(epochs_per_trial):
        # Deterministic shuffling per epoch across trials
        epoch_key, shuffle_key = jax.random.split(epoch_key)
        perm = jax.random.permutation(shuffle_key, static_data['e_train'].shape[0])

        e_s   = static_data['e_train'][perm]
        x_s   = static_data['x_train'][perm]
        ed_s  = static_data['e_dot_train'][perm]
        H_s   = static_data['H_train'][perm]

        epoch_train_h_loss = 0.0
        for i in range(num_batches):
            start, end = i * batch_size, (i + 1) * batch_size
            e_b, x_b, ed_b, H_b = e_s[start:end], x_s[start:end], ed_s[start:end], H_s[start:end]

            model, opt_state, _, loss_comps = train_step(
                model, opt_state, optimizer,
                e_b, x_b, ed_b, H_b,
                **lambdas, **static_data['static_params']
            )
            epoch_train_h_loss += loss_comps['hamiltonian']

        avg_epoch_train_h_loss = epoch_train_h_loss / num_batches
        if avg_epoch_train_h_loss < best_train_h_loss:
            best_train_h_loss = avg_epoch_train_h_loss

        # Validation (full set, same split for all trials)
        val_loss, _ = evaluate_model(
            model,
            static_data['e_val'], static_data['x_val'], static_data['e_dot_val'], static_data['H_val'],
            **lambdas, **static_data['static_params']
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss

    # --- 4) Objective ---
    # If we optimize w.r.t. train Hamiltonian loss, return that;
    # otherwise, return the best validation loss on this script's split.
    if objective_metric == 'h_loss':
        return float(best_train_h_loss)
    return float(best_val_loss)


def main():
    """Set up data, build an Optuna study, and launch hyperparameter search."""
    parser = argparse.ArgumentParser(description="Optuna HPO for raw-data pH-PINN.")
    parser.add_argument(
        '--objective',
        type=str,
        default='val_loss',
        choices=['val_loss', 'h_loss'],
        help="Objective metric to minimize. NOTE: 'val_loss' uses THIS script's split."
    )
    args = parser.parse_args()
    print(f"Starting optimization with objective: {args.objective}")

    # --- Load data once (RAW) ---
    print("Loading and preparing data for optimization...")
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'PINN Data', 'error_system_data.pkl')
    t, e, x, e_dot, x_dot, H, _ = generate_data(data_path)
    if t is None:
        sys.exit("Exiting: Data loading failed.")

    # Deterministic split via master seed
    master_seed = 42
    master_key = jax.random.PRNGKey(master_seed)

    validation_split = 0.2
    num_samples = e.shape[0]

    # IMPORTANT:
    # This permutation defines which samples are in train and which are in val
    # for the *entire* Optuna study. If another script uses a different
    # permutation, its 'val_loss' won't match what Optuna found here.
    split_key, _ = jax.random.split(master_key)
    perm = jax.random.permutation(split_key, num_samples)

    # Shuffle RAW (no normalization)
    t_s, e_s, x_s, ed_s, xd_s, H_s = t[perm].reshape(-1, 1), e[perm], x[perm], e_dot[perm], x_dot[perm], H[perm]

    split_idx = int(num_samples * (1 - validation_split))
    e_train, e_val   = jnp.split(e_s,  [split_idx])
    x_train, x_val   = jnp.split(x_s,  [split_idx])
    ed_train, ed_val = jnp.split(ed_s, [split_idx])
    H_train, H_val   = jnp.split(H_s,  [split_idx])

    # --- Package for objective ---
    static_data = {
        'master_key': master_key,  # used to seed each trial deterministically
        'e_dim': e_train.shape[1],

        # RAW splits
        'e_train': e_train, 'x_train': x_train, 'e_dot_train': ed_train, 'H_train': H_train,
        'e_val':   e_val,   'x_val':   x_val,   'e_dot_val':   ed_val,   'H_val':   H_val,

        # Params expected by loss_fn_ph via train_step/evaluate_model
        'static_params': {
            'hr_params': {**DEFAULT_PARAMS, 'ge': 0.62},
            'I_ext': jnp.array([0.8, 0.8]),
            'xi': jnp.array([[0, 1], [1, 0]]),
        },
    }

    # --- Create & run study ---
    print("\nStarting Optuna hyperparameter search...")
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results', 'PINN Data')
    os.makedirs(results_dir, exist_ok=True)
    db_path = os.path.join(results_dir, "optimize_hyperparams.db")
    storage_name = f"sqlite:///{db_path}"
    study_name = f"ph_pinn_optimization_{args.objective}"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="minimize",
        load_if_exists=True,
    )

    # Wrap objective to pass extra args
    objective_with_args = lambda trial: objective(
        trial,
        epochs_per_trial=1000,  # keep same training budget per trial
        static_data=static_data,
        objective_metric=args.objective
    )
    study.optimize(objective_with_args, n_trials=100)

    # --- Report ---
    print("\nOptimization finished.")
    print(f"Study results are saved in: {storage_name}")
    print(f"To view dashboard, run: optuna-dashboard {storage_name}")

    best_trial = study.best_trial
    print("\n" + "=" * 40)
    print("         Best Trial Found")
    print("=" * 40)
    print(f"  Value (Best {args.objective}): {best_trial.value:.6f}")
    print("  Best Hyperparameters: ")
    for key, value in best_trial.params.items():
        print(f"    '{key}': {value},")
    print("=" * 40)


if __name__ == "__main__":
    main()
