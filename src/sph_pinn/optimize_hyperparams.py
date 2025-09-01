"""
optimize_hyperparams.py
-----------------------
This script uses the Optuna framework to perform an automated hyperparameter
search for the Combined_sPHNN_PINN model defined in pH_PINN.py.
It defines an "objective" function that Optuna repeatedly calls with different
hyperparameter combinations. Each call (a "trial") trains the model for a
fixed number of epochs and returns the best training Hamiltonian loss.
Optuna uses these results to intelligently search for the optimal set of hyperparameters.
Results of the study are saved to a SQLite database file (optimize_hyperparams.db)
in the 'results/PINN Data/' directory, allowing the optimization to be paused
and resumed.
To run this script:
1. Make sure you have Optuna and its storage dependencies installed:
   pip install optuna
   pip install "optuna[storages]"
2. Place this file in the `src/sph_pinn/` directory.
3. Run from the root directory of your project: python -m src.sph_pinn.optimize_hyperparams

To view the results dashboard after running (run from project root):
optuna-dashboard "sqlite:///results/PINN Data/optimize_hyperparams.db"
"""
import os
import sys
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import optuna

# Ensure the script can find other modules in the project
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# --- Import necessary components from your existing files ---
from src.hr_model.model import DEFAULT_PARAMS
from src.sph_pinn.pH_PINN import (
    StateNN, HamiltonianNN, DissipationNN, DynamicJ_NN, # Import component networks
    generate_data,
    normalize,
    create_windows, # Import windowing function
    train_step,
    evaluate_model,
)

# JAX configuration
jax.config.update("jax_enable_x64", True)

# ==============================================================================
# 1. LOCAL MODEL DEFINITION FOR HYPERPARAMETER FLEXIBILITY
# ==============================================================================

# We redefine the combined model here to match the new architecture in pH_PINN.py
# and to easily pass hyperparameters from the Optuna trial.
class Combined_sPHNN_PINN(eqx.Module):
    """Main model combining a unified state predictor and sPHNN structure."""
    state_net: StateNN
    hamiltonian_net: HamiltonianNN
    dissipation_net: DissipationNN
    j_net: DynamicJ_NN

    def __init__(self, key, config):
        state_key, h_key, d_key, j_key = jax.random.split(key, 4)

        state_dim = config['state_dim']
        q_dim = config['q_dim']

        # Unpack configs for clarity
        cfg_state = config['state_nn']
        cfg_h = config['hamiltonian_nn']
        cfg_d = config['dissipation_nn']
        cfg_j = config['j_net']

        self.state_net = StateNN(
            key=state_key,
            out_size=state_dim + q_dim,
            width=cfg_state['width'],
            depth=cfg_state['depth'],
            activation=cfg_state['activation'],
            mapping_size=cfg_state['fourier_features']['mapping_size'],
            scale=cfg_state['fourier_features']['scale']
        )
        self.hamiltonian_net = HamiltonianNN(
            h_key, state_dim=state_dim,
            hidden_size=cfg_h['hidden_size'],
            ficnn_width=cfg_h['ficnn']['width'],
            ficnn_depth=cfg_h['ficnn']['depth'],
            ficnn_activation=cfg_h['ficnn']['activation']
        )
        self.dissipation_net = DissipationNN(
            d_key, state_dim=state_dim,
            hidden_size=cfg_d['hidden_size'],
            width=cfg_d['width'],
            depth=cfg_d['depth'],
            activation=cfg_d['activation']
        )
        self.j_net = DynamicJ_NN(
            j_key, state_dim=state_dim,
            hidden_size=cfg_j['hidden_size'],
            width=cfg_j['width'],
            depth=cfg_j['depth'],
            activation=cfg_j['activation']
        )

# ==============================================================================
# 2. OBJECTIVE FUNCTION FOR OPTUNA
# ==============================================================================

def objective(trial, epochs_per_trial, static_data):
    """
    The main objective function that Optuna will minimize.
    Args:
        trial (optuna.Trial): An Optuna trial object used to suggest hyperparameters.
        epochs_per_trial (int): The number of epochs to train for during each trial.
        static_data (dict): A dictionary containing all pre-processed UN-WINDOWED data and stats.
    Returns:
        float: The best training Hamiltonian loss achieved during the trial.
    """
    # --- 1. Suggest Hyperparameters from the Search Space ---
    key = jax.random.PRNGKey(42)  # Use a fixed key for reproducibility across trials
    model_key, _ = jax.random.split(key)

    # Data pre-processing
    window_size = trial.suggest_int("window_size", 32, 128)

    # StateNN Fourier Features & Architecture
    mapping_size = trial.suggest_int("mapping_size", 32, 128)
    scale = trial.suggest_float("scale", 10, 500, log=True)
    state_width = trial.suggest_int("state_width", 8, 512)
    state_depth = trial.suggest_int("state_depth", 2, 6)
    state_activation_name = trial.suggest_categorical("state_activation", ["tanh", "softplus"])
    state_activation = getattr(jax.nn, state_activation_name)

    # HamiltonianNN (LSTM + FICNN)
    h_hidden_size = trial.suggest_int("h_hidden_size", 8, 512)
    h_ficnn_width = trial.suggest_int("h_ficnn_width", 8, 512)
    h_ficnn_depth = trial.suggest_int("h_ficnn_depth", 2, 6)

    # DissipationNN and J_NN (LSTMs + MLPs)
    jr_hidden_size = trial.suggest_int("jr_hidden_size", 16, 64)
    d_width = trial.suggest_int("d_width", 4, 64)
    d_depth = trial.suggest_int("d_depth", 2, 6)
    j_width = trial.suggest_int("j_width", 4, 64)
    j_depth = trial.suggest_int("j_depth", 2, 6)

    # Optimizer
    lr_initial = trial.suggest_float("lr_initial", 1e-4, 1e-2, log=True)
    decay_steps = trial.suggest_int("decay_steps", 500, 4000)

    # Training and Loss
    batch_size = trial.suggest_categorical("batch_size", [1024, 2048])
    lambda_conservative_max = trial.suggest_float("lambda_conservative_max", 0.1, 20, log=True)
    lambda_dissipative_max = trial.suggest_float("lambda_dissipative_max", 0.1, 20, log=True)
    lambda_physics_max = trial.suggest_float("lambda_physics_max", 0.1, 20, log=True)
    lambda_warmup_epochs = trial.suggest_int("lambda_warmup_epochs", 500, 3000)

    # --- 2. Create Windowed Data for this Trial ---
    (t_train_w, s_train_w, q_train_w, s_dot_train_w, H_train_w) = create_windows(
        window_size, static_data['t_train_norm'], static_data['s_train_norm'], static_data['q_train_norm'],
        static_data['s_dot_train_norm'], static_data['H_train_norm']
    )
    (t_val_w, s_val_w, q_val_w, s_dot_val_w, H_val_w) = create_windows(
        window_size, static_data['t_val_norm'], static_data['s_val_norm'], static_data['q_val_norm'],
        static_data['s_dot_val_norm'], static_data['H_val_norm']
    )

    # --- 3. Build Model and Optimizer with Suggested Values ---
    nn_config = {
        "state_dim": static_data['s_dim'],
        "q_dim": static_data['q_dim'],
        "state_nn": {
            "width": state_width, "depth": state_depth, "activation": state_activation,
            "fourier_features": {"mapping_size": mapping_size, "scale": scale}
        },
        "hamiltonian_nn": {
            "hidden_size": h_hidden_size,
            "ficnn": {"width": h_ficnn_width, "depth": h_ficnn_depth, "activation": jax.nn.softplus}
        },
        "dissipation_nn": {
            "hidden_size": jr_hidden_size, "width": d_width, "depth": d_depth, "activation": jax.nn.softplus
        },
        "j_net": {
            "hidden_size": jr_hidden_size, "width": j_width, "depth": j_depth, "activation": jax.nn.softplus
        }
    }
    model = Combined_sPHNN_PINN(key=model_key, config=nn_config)

    lr_schedule = optax.linear_schedule(
        init_value=lr_initial, end_value=1e-5, transition_steps=decay_steps
    )
    optimizer = optax.adamw(learning_rate=lr_schedule)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # --- 4. Run the Training Loop ---
    best_training_hamiltonian_loss = jnp.inf
    num_batches = t_train_w.shape[0] // batch_size
    if num_batches == 0:
        num_batches = 1

    for epoch in range(epochs_per_trial):
        warmup_factor = jnp.minimum(1.0, (epoch + 1) / lambda_warmup_epochs)
        current_lambda_conservative = lambda_conservative_max * warmup_factor
        current_lambda_dissipative = lambda_dissipative_max * warmup_factor
        current_lambda_physics = lambda_physics_max * warmup_factor

        key, shuffle_key = jax.random.split(key)
        perm = jax.random.permutation(shuffle_key, t_train_w.shape[0])
        t_shuffled, s_shuffled, q_shuffled, s_dot_shuffled, H_shuffled = (
            t_train_w[perm], s_train_w[perm], q_train_w[perm], s_dot_train_w[perm], H_train_w[perm]
        )

        epoch_hamiltonian_loss = 0.0
        for i in range(num_batches):
            start, end = i * batch_size, (i + 1) * batch_size
            t_b, s_b, q_b, s_dot_b, H_b = (
                t_shuffled[start:end], s_shuffled[start:end], q_shuffled[start:end],
                s_dot_shuffled[start:end], H_shuffled[start:end]
            )
            model, opt_state, _, loss_components = train_step(
                model, opt_state, optimizer, t_b, s_b, q_b, s_dot_b, H_b,
                current_lambda_conservative, current_lambda_dissipative, current_lambda_physics,
                static_data['hr_params'], static_data['t_mean'], static_data['t_std'],
                static_data['s_mean'], static_data['s_std'], static_data['q_mean'],
                static_data['q_std'], static_data['s_dot_mean'], static_data['s_dot_std'],
                static_data['H_mean'], static_data['H_std']
            )
            epoch_hamiltonian_loss += loss_components['hamiltonian']

        avg_epoch_hamiltonian_loss = epoch_hamiltonian_loss / num_batches

        if avg_epoch_hamiltonian_loss < best_training_hamiltonian_loss:
            best_training_hamiltonian_loss = avg_epoch_hamiltonian_loss

    # --- 5. Return the Final Metric to Optuna ---
    return best_training_hamiltonian_loss


# ==============================================================================
# 3. MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == "__main__":
    # --- 1. Load and Prepare Data (Done Once) ---
    print("Loading and preparing data for optimization...")
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'PINN Data', 'error_system_data.pkl')
    t, s, q, s_dot_true, H_analytical = generate_data(data_path)
    if t is None:
        sys.exit("Exiting: Data loading failed. Make sure 'error_system_data.pkl' exists.")

    validation_split = 0.2
    num_samples = s.shape[0]
    key = jax.random.PRNGKey(123)
    perm = jax.random.permutation(key, num_samples)
    t_shuffled, s_shuffled, q_shuffled, s_dot_shuffled, H_shuffled = \
        t[perm], s[perm], q[perm], s_dot_true[perm], H_analytical[perm]
    t_shuffled = t_shuffled.reshape(-1, 1)

    split_idx = int(num_samples * (1 - validation_split))
    t_train, t_val = jnp.split(t_shuffled, [split_idx])
    s_train, s_val = jnp.split(s_shuffled, [split_idx])
    q_train, q_val = jnp.split(q_shuffled, [split_idx])
    s_dot_train, s_dot_val = jnp.split(s_dot_shuffled, [split_idx])
    H_train, H_val = jnp.split(H_shuffled, [split_idx])

    # --- Normalize Data ---
    t_mean, t_std = jnp.mean(t_train), jnp.std(t_train)
    s_mean, s_std = jnp.mean(s_train, axis=0), jnp.std(s_train, axis=0)
    q_mean, q_std = jnp.mean(q_train, axis=0), jnp.std(q_train, axis=0)
    s_dot_mean, s_dot_std = jnp.mean(s_dot_train, axis=0), jnp.std(s_dot_train, axis=0)
    H_mean, H_std = jnp.mean(H_train), jnp.std(H_train)

    static_data = {
        's_dim': s_train.shape[1], 'q_dim': q_train.shape[1],
        'hr_params': DEFAULT_PARAMS.copy(),
        't_train_norm': normalize(t_train, t_mean, t_std),
        's_train_norm': normalize(s_train, s_mean, s_std),
        'q_train_norm': normalize(q_train, q_mean, q_std),
        's_dot_train_norm': normalize(s_dot_train, s_dot_mean, s_dot_std),
        'H_train_norm': normalize(H_train, H_mean, H_std),
        't_val_norm': normalize(t_val, t_mean, t_std),
        's_val_norm': normalize(s_val, s_mean, s_std),
        'q_val_norm': normalize(q_val, q_mean, q_std),
        's_dot_val_norm': normalize(s_dot_val, s_dot_mean, s_dot_std),
        'H_val_norm': normalize(H_val, H_mean, H_std),
        't_mean': t_mean, 't_std': t_std, 's_mean': s_mean, 's_std': s_std,
        'q_mean': q_mean, 'q_std': q_std, 's_dot_mean': s_dot_mean, 's_dot_std': s_dot_std,
        'H_mean': H_mean, 'H_std': H_std,
    }

    # --- 2. Create and Run the Optuna Study ---
    print("\nStarting Optuna hyperparameter search...")

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results', 'PINN Data')
    os.makedirs(results_dir, exist_ok=True)

    db_name = os.path.basename(__file__).replace('.py', '.db')
    db_path = os.path.join(results_dir, db_name)

    storage_name = f"sqlite:///{db_path}"
    study_name = "sphnn_pinn_optimization_study_v2" # New study name for new architecture

    # Use a lambda to pass static arguments to the objective function
    objective_with_args = lambda trial: objective(trial, epochs_per_trial=500, static_data=static_data)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="minimize",
        load_if_exists=True
    )

    study.optimize(objective_with_args, n_trials=50) # Increased trials for larger search space

    # --- 3. Print and Save the Results ---
    print("\nOptimization finished.")
    print(f"Study results are saved in: {storage_name}")
    print("Number of finished trials: ", len(study.trials))

    best_trial = study.best_trial

    print("Best trial:")
    print(f"  Value (Best Training Hamiltonian Loss): {best_trial.value:.6f}")
    print("  Best Hyperparameters: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    output_txt_path = os.path.join(results_dir, "best_hyperparams_v2.txt")
    with open(output_txt_path, 'w') as f:
        f.write("Best Hyperparameter Optimization Results (v2 Architecture)\n")
        f.write("========================================================\n\n")
        f.write(f"Best Value (Training Hamiltonian Loss): {best_trial.value}\n\n")
        f.write("Best Hyperparameters:\n")
        for key, value in best_trial.params.items():
            f.write(f"    {key}: {value}\n")

    print(f"\n✅ Best hyperparameters saved to: {output_txt_path}")