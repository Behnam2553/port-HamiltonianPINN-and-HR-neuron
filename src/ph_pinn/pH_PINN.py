import jax, jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
import matplotlib.pyplot as plt
import sys
from src.hr_model.model import DEFAULT_PARAMS
import os
from pathlib import Path
import pickle

# JAX configuration to use 64-bit precision.
# This is important for physics-y problems where numerical stability/precision matters.
jax.config.update("jax_enable_x64", True)


# ==============================================================================
# 1. NEURAL NETWORK DEFINITIONS
# ==============================================================================
# These networks implement the building blocks of the port-Hamiltonian PINN:
#   - HamiltonianNN: learns a scalar energy/Hamiltonian H(e, ctx)
#   - DissipationNN: learns a diagonal dissipation matrix R(e, ctx)
#   - DynamicJ_NN:   learns a sparse/skew-symmetric interconnection matrix J(e, ctx)
# All of them take as input the error state e (dimension = state_dim)
# plus 4 context variables (x1, u1, u2, phi1) extracted elsewhere.
# ==============================================================================


class HamiltonianNN(eqx.Module):
    """
    MLP that approximates the Hamiltonian H(e, ctx), i.e. a scalar energy function.

    Design notes:
    - Input is the concatenation of the error state e (size = e_dim) and 4 context
      features (x1, u1, u2, phi1) -> total input size = in_size_with_ctx.
    - The raw MLP output is "anchored" at a reference point e = x0 (with the same ctx)
      so that:
        * H(x0, ctx) = 0
        * ∂H/∂e (x0, ctx) = 0
      This is done by subtracting a first-order Taylor expansion around x0_ext.
    - A small quadratic regularization in e is added to enforce a strict minimum
      at the equilibrium (helps with identifiability).
    """
    layers: list
    x0: jax.Array              # reference/anchor point in e-space
    epsilon: float = eqx.field(static=True)   # weight for quadratic anchoring
    input_dim: int = eqx.field(static=True)   # total input dim = e_dim + 4
    activation: callable = eqx.field(static=True)

    def __init__(self, key, in_size_with_ctx, e_dim, hidden_sizes, x0, epsilon, activation):
        """
        Args:
            key: JAX PRNG key for parameter init.
            in_size_with_ctx: total input size (e_dim + 4 context features).
            e_dim: dimension of the error/state variable (kept for clarity).
            hidden_sizes: list of hidden layer sizes for the MLP.
            x0: reference point in e-space (typically zeros(state_dim)).
            epsilon: weight for the quadratic anchor term.
            activation: activation function (e.g. jax.nn.tanh).
        """
        self.activation = activation
        self.x0 = x0
        self.epsilon = epsilon
        self.input_dim = in_size_with_ctx

        # Build a simple MLP ending in a single scalar
        # We create len(hidden_sizes) hidden layers + 1 output layer.
        keys = jax.random.split(key, len(hidden_sizes) + 1)
        layers = []
        prev = in_size_with_ctx
        for i, h in enumerate(hidden_sizes):
            layers.append(eqx.nn.Linear(prev, h, key=keys[i]))
            prev = h
        # Final layer outputs 1 scalar (Hamiltonian value)
        layers.append(eqx.nn.Linear(prev, 1, key=keys[-1]))
        self.layers = layers

    def _forward(self, x):
        """Plain MLP forward pass that returns a scalar (shape ())."""
        z = x
        for layer in self.layers[:-1]:
            z = self.activation(layer(z))
        # eqx Linear returns shape (1,), so we take [0] to get a scalar
        return self.layers[-1](z)[0]

    def __call__(self, e_in, ctx_in):
        """
        Evaluate the anchored Hamiltonian H(e_in, ctx_in).

        Steps:
        1. Form the full input x_in = [e_in, ctx_in].
        2. Form the anchored input x0_ext = [x0, ctx_in] (same context, reference e).
        3. Compute raw MLP outputs at both locations.
        4. Subtract the first-order Taylor expansion around x0_ext to enforce
           H(x0, ctx) = 0 and ∂H/∂e(x0, ctx) = 0.
        5. Add a small quadratic term in (e - x0) to make the minimum strict.
        """
        # Concatenate e and context for the MLP input
        x_in = jnp.concatenate([e_in, ctx_in])       # shape: [e_dim + 4]
        x0_ext = jnp.concatenate([self.x0, ctx_in])  # same ctx, anchor at e = x0

        # Raw MLP outputs
        f_x = self._forward(x_in)
        f_x0 = self._forward(x0_ext)

        # Gradient of MLP at the anchor point (wrt full input [e, ctx])
        grad_f_x0 = jax.grad(self._forward)(x0_ext)

        # Normalize so H(e0,ctx)=0 and ∂H/∂e(e0,ctx)=0 (ctx part cancels in the dot)
        # This is f(x) - [f(x0) + ∇f(x0)ᵀ (x - x0)]
        f_norm = f_x0 + jnp.dot(grad_f_x0, x_in - x0_ext)

        # Quadratic anchor for a strict minimum in e-space
        f_reg = self.epsilon * jnp.sum((e_in - self.x0) ** 2)

        # Final Hamiltonian value
        return f_x - f_norm + f_reg


class DissipationNN(eqx.Module):
    """
    Neural network for the dissipation matrix R(e, ctx), **restricted to be diagonal**.

    R(e, ctx) = diag(r1, r2, r3, r4, r5)

    - Input: concatenation of e (size = 5) and 4 context features -> input_dim
    - Output: 5 scalars placed on the diagonal of a 5x5 matrix.
    - All off-diagonal elements are forced to zero to keep the structure simple
      and compatible with the physics-based losses.
    """
    layers: list
    activation: callable = eqx.field(static=True)
    state_dim: int = eqx.field(static=True)
    input_dim: int  # e_dim + 4

    def __init__(self, key, state_dim, input_dim, hidden_sizes, activation):
        """
        Args:
            key: PRNG key.
            state_dim: expected state dimension (must be 5 in this setup).
            input_dim: dimension of [e, ctx].
            hidden_sizes: list with hidden layer sizes (can be empty).
            activation: activation function used in all hidden layers.
        """
        # This implementation is specialized for 5D error states.
        assert state_dim == 5, "Diagonal DissipationNN assumes state_dim == 5."
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.activation = activation

        out_dim = 5  # only the 5 diagonal entries

        if len(hidden_sizes) == 0:
            # No hidden layers: directly map input -> 5 diagonal values
            self.layers = [eqx.nn.Linear(self.input_dim, out_dim, key=key)]
        else:
            # Build MLP with given hidden sizes
            keys = jax.random.split(key, len(hidden_sizes) + 1)
            self.layers = [eqx.nn.Linear(self.input_dim, hidden_sizes[0], key=keys[0])]
            for i in range(1, len(hidden_sizes)):
                self.layers.append(eqx.nn.Linear(hidden_sizes[i - 1], hidden_sizes[i], key=keys[i]))
            # Final layer outputs the 5 diagonal entries
            self.layers.append(eqx.nn.Linear(hidden_sizes[-1], out_dim, key=keys[-1]))

    def __call__(self, e, ctx):
        """
        Forward pass that returns a 5x5 diagonal matrix R.

        Args:
            e: error/state vector, shape (5,)
            ctx: context vector, shape (4,)

        Returns:
            R: (5, 5) array, diagonal filled with NN outputs.
        """
        # Concatenate state and context -> (5 + 4,)
        z = jnp.concatenate([e, ctx])

        # Hidden layers
        for layer in self.layers[:-1]:
            z = self.activation(layer(z))

        # Head: produce 5 diagonal values
        diag_vals = self.layers[-1](z)  # shape (5,)

        # Build diagonal R
        R = jnp.zeros((5, 5), dtype=diag_vals.dtype)
        idx = jnp.arange(5)
        R = R.at[idx, idx].set(diag_vals)

        return R


class DynamicJ_NN(eqx.Module):
    """
    Neural network for a **very sparse** and **skew-symmetric** interconnection matrix J(e, ctx).

    Assumptions:
    - state_dim == 5
    - We only learn the first row entries J[0, 1:], i.e. 4 values:
          [j12, j13, j14, j15]
    - Skew-symmetry is enforced by setting:
          J[i, 0] = -J[0, i] for i = 1..4
    - All remaining entries stay 0.
    - Diagonal is 0 by construction.

    This keeps the model small and JAX-friendly while still allowing
    nontrivial power-conserving interconnections.
    """
    layers: list
    state_dim: int = eqx.field(static=True)
    input_dim: int  # e_dim + 4
    activation: callable = eqx.field(static=True)

    def __init__(self, key, state_dim, input_dim, hidden_sizes, activation):
        """
        Args:
            key: PRNG key.
            state_dim: expected state dimension (must be 5 here).
            input_dim: dimension of [e, ctx].
            hidden_sizes: list with hidden layer sizes (can be empty).
            activation: activation function used in hidden layers.
        """
        assert state_dim == 5, "DynamicJ_NN (sparse) assumes state_dim == 5."
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.activation = activation

        # We predict 4 numbers: [j12, j13, j14, j15]
        out_dim = 4
        if len(hidden_sizes) == 0:
            # Direct mapping input -> 4
            self.layers = [eqx.nn.Linear(self.input_dim, out_dim, key=key)]
        else:
            # MLP with the provided hidden sizes
            keys = jax.random.split(key, len(hidden_sizes) + 1)
            self.layers = [eqx.nn.Linear(self.input_dim, hidden_sizes[0], key=keys[0])]
            for i in range(1, len(hidden_sizes)):
                self.layers.append(eqx.nn.Linear(hidden_sizes[i - 1], hidden_sizes[i], key=keys[i]))
            self.layers.append(eqx.nn.Linear(hidden_sizes[-1], out_dim, key=keys[-1]))

    def __call__(self, e, ctx):
        """
        Forward pass that returns a 5x5 sparse, skew-symmetric J matrix.

        Args:
            e: error/state vector, shape (5,)
            ctx: context vector, shape (4,)

        Returns:
            J: (5, 5) array, with only the first row/column populated.
        """
        # Concatenate inputs, same convention as other nets
        z = jnp.concatenate([e, ctx])

        # Forward through hidden layers
        for layer in self.layers[:-1]:
            z = self.activation(layer(z))

        # Last layer predicts the tail of the first row: [j12, j13, j14, j15]
        row_tail = self.layers[-1](z)  # shape (4,)

        # Build J with two scatter writes (efficient in JAX)
        J = jnp.zeros((5, 5), dtype=row_tail.dtype)
        J = J.at[0, 1:].set(row_tail)   # first row, columns 1..4 (i.e. indices 1: )
        J = J.at[1:, 0].set(-row_tail)  # enforce skew-symmetry for first column

        # Diagonal is already zero; other entries stay zero.
        return J


# --- The Combined Model (StateNN removed) ---

class Combined_pH_PINN(eqx.Module):
    """
    Container module that holds all pH-PINN sub-networks.

    This class does **not** implement the pH dynamics itself; it just bundles:
        - HamiltonianNN  -> H(e, ctx)
        - DissipationNN  -> R(e, ctx)
        - DynamicJ_NN    -> J(e, ctx)

    The training loop / loss function decides how to call these.
    """
    hamiltonian_net: HamiltonianNN
    dissipation_net: DissipationNN
    j_net: DynamicJ_NN

    def __init__(self, key, config: dict, state_dim: int):
        """
        Args:
            key: PRNG key to initialize all sub-networks.
            config: dictionary with sub-configs, expected keys:
                {
                    'hamiltonian_net': {'hidden_sizes': [...], 'epsilon': ...},
                    'dissipation_net': {'hidden_sizes': [...]},
                    'j_net':           {'hidden_sizes': [...]},
                    'activation':      callable
                }
            state_dim: dimension of the error/state e (should be 5 for this setup).
        """
        # Split master key for all three networks
        h_key, d_key, j_key = jax.random.split(key, 3)

        # Extract sub-configs
        h_net_config = config['hamiltonian_net']
        d_net_config = config['dissipation_net']
        j_net_config = config['j_net']
        activation_fn = config['activation']

        # The equilibrium point for the normalized error system is the origin.
        # We use this as the anchor x0 for the Hamiltonian.
        x0_norm = jnp.zeros(state_dim)

        # Input dimensions (raw inputs): e_dim + 4 context features (x1,u1,u2,phi1)
        input_dim_with_ctx = state_dim + 4

        # Hamiltonian network
        self.hamiltonian_net = HamiltonianNN(
            h_key,
            in_size_with_ctx=input_dim_with_ctx,
            e_dim=state_dim,
            hidden_sizes=h_net_config['hidden_sizes'],
            x0=x0_norm,
            epsilon=h_net_config['epsilon'],
            activation=activation_fn,
        )

        # Dissipation network (diagonal R)
        self.dissipation_net = DissipationNN(
            d_key,
            state_dim=state_dim,
            input_dim=input_dim_with_ctx,
            hidden_sizes=d_net_config['hidden_sizes'],
            activation=activation_fn,
        )

        # Skew/sparse J network
        self.j_net = DynamicJ_NN(
            j_key,
            state_dim=state_dim,
            input_dim=input_dim_with_ctx,
            hidden_sizes=j_net_config['hidden_sizes'],
            activation=activation_fn,
        )

# ==============================================================================
# 2. DATA HANDLING
# ==============================================================================

def generate_data(file_path: str):
    """
    Load and aggregate training/validation data from a pickle file.

    The pickle is expected to contain **a list of simulation runs**. Each element
    in that list is a dictionary with keys like:
        't', 'e_x', 'e_y', 'e_z', 'e_u', 'e_phi',
        'x1', 'y1', 'z1', 'u1', 'phi1',
        'x2', 'y2', 'z2', 'u2', 'phi2',
        'd_e_x', ..., 'd_e_phi',
        'd_x1', ..., 'd_phi2',
        'Hamiltonian',
        'dHdt'

    This function:
      1. loads all runs,
      2. converts each run to JAX arrays,
      3. stacks per-run arrays across the time dimension,
      4. concatenates all runs into one big dataset.

    Returns:
        final_t      : (N,)         concatenated time vector
        final_e      : (N, 5)       error states  [e_x, e_y, e_z, e_u, e_phi]
        final_x      : (N, 10)      full states   [x1..phi2]
        final_e_dot  : (N, 5)       true derivatives of error states
        final_x_dot  : (N, 10)      true derivatives of full states
        final_H      : (N,)         analytical Hamiltonian values
        final_dHdt   : (N,)         analytical dH/dt values

    On missing file:
        prints a message and returns a tuple of Nones with the same arity.
    """
    print(f"Loading simulation data from {file_path}...")
    try:
        with open(file_path, 'rb') as f:
            # The file contains a list of result dictionaries
            all_runs_results = pickle.load(f)
    except FileNotFoundError:
        # Keep original behaviour: print + return Nones, so caller can exit cleanly.
        print(f"Error: Data file not found at {file_path}")
        print("Please run 'generate_data_for_pH_PINN.py' to create the data file.")
        return None, None, None, None, None, None

    # Initialize lists to hold data from all runs
    # We collect each run separately and concatenate at the end.
    all_t, all_e, all_x, all_e_dot, all_x_dot, all_H, all_dHdt  = [], [], [], [], [], [], []

    # Process each simulation run
    for i, results in enumerate(all_runs_results):
        print(f"  ... processing run {i + 1}/{len(all_runs_results)}")

        # Extract data for the current run
        # time vector
        t = jnp.asarray(results['t'])

        # error states -> shape (T, 5)
        e = jnp.vstack([
            results['e_x'], results['e_y'], results['e_z'],
            results['e_u'], results['e_phi']
        ]).T

        # full states -> shape (T, 10)
        # ordering is important; later code depends on indices (x1 = 0, u1 = 3, phi1 = 4, u2 = 8, ...)
        x = jnp.vstack([
            results['x1'], results['y1'], results['z1'], results['u1'], results['phi1'],
            results['x2'], results['y2'], results['z2'], results['u2'], results['phi2']
        ]).T

        # true derivatives of error states -> shape (T, 5)
        e_dot_true = jnp.vstack([
            results['d_e_x'], results['d_e_y'], results['d_e_z'],
            results['d_e_u'], results['d_e_phi']
        ]).T

        # analytical Hamiltonian & its time derivative
        H_analytical = jnp.asarray(results['Hamiltonian'])
        dHdt = jnp.asarray(results['dHdt'])

        # Stack the true derivatives for the x states -> shape (T, 10)
        x_dot_true = jnp.vstack([
            results['d_x1'], results['d_y1'], results['d_z1'], results['d_u1'], results['d_phi1'],
            results['d_x2'], results['d_y2'], results['d_z2'], results['d_u2'], results['d_phi2']
        ]).T

        # Append to the main lists for later concatenation
        all_t.append(t)
        all_e.append(e)
        all_x.append(x)
        all_e_dot.append(e_dot_true)
        all_H.append(H_analytical)
        all_dHdt.append(dHdt)
        all_x_dot.append(x_dot_true)

    # Concatenate all runs into single arrays along time axis
    final_t = jnp.concatenate(all_t)
    final_e = jnp.concatenate(all_e)
    final_x = jnp.concatenate(all_x)
    final_e_dot = jnp.concatenate(all_e_dot)
    final_H = jnp.concatenate(all_H)
    final_dHdt = jnp.concatenate(all_dHdt)
    final_x_dot = jnp.concatenate(all_x_dot)

    print("Data loading and aggregation complete.")
    return final_t, final_e, final_x, final_e_dot, final_x_dot, final_H, final_dHdt


def normalize(data, mean, std):
    """
    Normalize data using pre-computed statistics.

    This helper is generic; in the current script you operate mostly on raw data,
    but keeping this makes it easy to switch to normalized training.
    """
    return (data - mean) / (std + 1e-8)


def denormalize(data, mean, std):
    """
    Inverse of `normalize`: reconstruct original scale from normalized data.
    """
    return data * std + mean


def load_best_config_from_study(default_config, objective='h_loss'):
    """
    Load best hyperparameters from an Optuna study (if present) and merge them
    into an existing training configuration.

    This is meant to be a *non-breaking* upgrade step:
    - if the study/database is missing, we just return the original config;
    - if some parameters were not part of the study, we keep defaults.

    Args:
        default_config: the base TRAIN_CONFIG dict to start from.
        objective: part of the study name, e.g. 'h_loss'.

    Returns:
        new_config: updated config with the best trial's parameters injected.
    """
    import copy
    import os
    import optuna
    import jax

    # Path to the Optuna DB (unchanged)
    # Note: this assumes your repo structure has "results/PINN Data/..." two levels up.
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results', 'PINN Data')
    db_path = os.path.join(results_dir, "optimize_hyperparams.db")
    storage_name = f"sqlite:///{db_path}"
    study_name = f"ph_pinn_optimization_{objective}"

    print("=" * 60)
    print(f"Attempting to load best hyperparameters for study '{study_name}'...")
    print(f"Database: {db_path}")
    print("=" * 60)

    try:
        # Load existing study
        study = optuna.load_study(study_name=study_name, storage=storage_name)
        best_params = study.best_trial.params
        print("✅ Successfully loaded best hyperparameters from study.")
    except Exception as e:
        # Keep original behaviour: warn and return untouched config
        print(f"❌ WARNING: Could not load Optuna study. Reason: {e}")
        print("         Falling back to the default TRAIN_CONFIG.")
        return default_config

    # Copy to avoid mutating the original config
    new_config = copy.deepcopy(default_config)

    # --- Top-level scalars we still use in the new setup ---
    # These are directly present at the root of TRAIN_CONFIG.
    direct_mapping_keys = [
        'batch_size',
        'initial_learning_rate',
        'decay_steps',
        'lambda_conservative_max',
        'lambda_dissipative_max',
        'lambda_physics_max',
        'lambda_j_structure_max',
        'lambda_r_structure_max',
    ]
    for key in direct_mapping_keys:
        if key in best_params:
            new_config[key] = best_params[key]

    # --- Network hyperparameters from the study (if present) ---
    # Hamiltonian
    if 'epsilon' in best_params:
        new_config['nn']['hamiltonian_net']['epsilon'] = best_params['epsilon']
    if 'h_width' in best_params and 'h_depth' in best_params:
        # convert width/depth into a list of equal-sized layers
        new_config['nn']['hamiltonian_net']['hidden_sizes'] = [best_params['h_width']] * int(best_params['h_depth'])

    # Dissipation R
    if 'd_width' in best_params and 'd_depth' in best_params:
        new_config['nn']['dissipation_net']['hidden_sizes'] = [best_params['d_width']] * int(best_params['d_depth'])

    # Skew J
    if 'j_width' in best_params and 'j_depth' in best_params:
        new_config['nn']['j_net']['hidden_sizes'] = [best_params['j_width']] * int(best_params['j_depth'])

    # Optional: activation from study (kept softplus by default for convexity)
    if 'activation' in best_params:
        act_name = str(best_params['activation']).lower()
        act_map = {
            'softplus': jax.nn.softplus,
            'relu': jax.nn.relu,
            # add more if you decide to search them
        }
        if act_name in act_map:
            new_config['nn']['activation'] = act_map[act_name]
        else:
            # Keep the original activation if this one is unknown
            print(f"⚠️ Unknown activation '{best_params['activation']}', keeping default.")

    print("✅ TRAIN_CONFIG has been updated with optimized hyperparameters.")
    return new_config
# ==============================================================================
# 3. TRAINING LOGIC
# ==============================================================================

# --- Helper functions for the physics-based loss terms ---


def _alpha(u1, u2, m):
    """
    Piecewise helper for the dissipative field f_d.

    This encodes a set of region-dependent coefficients based on the two inputs
    u1 and u2 (which come from the system state) and a slope/scale parameter m.
    The pattern of conditions matches the original model’s switching logic.

    Args:
        u1: first input variable (scalar or array).
        u2: second input variable (scalar or array).
        m:  model parameter from hr_params['m'].

    Returns:
        alpha(u1, u2): value selected from the piecewise definition.
    """
    conds = [
        jnp.logical_and(u1 >= 1, jnp.logical_and(u2 > -1, u2 < 1)),
        jnp.logical_and(u1 >= 1, u2 <= -1),
        jnp.logical_and(jnp.logical_and(u1 > -1, u1 < 1), u2 >= 1),
        jnp.logical_and(jnp.logical_and(u1 > -1, u1 < 1), jnp.logical_and(u2 > -1, u2 < 1)),
        jnp.logical_and(jnp.logical_and(u1 > -1, u1 < 1), u2 <= -1),
        jnp.logical_and(u1 <= -1, u2 >= 1),
        jnp.logical_and(u1 <= -1, jnp.logical_and(u2 > -1, u2 < 1)),
    ]
    # Each condition maps to a constant expression in terms of m
    choices = [2 * m - 1., -1., -1., 2 * m - 1., -1., -1., 2 * m - 1.]
    return jnp.select(conds, choices, default=-1.)


def _beta(u1, u2, m):
    """
    Piecewise helper for the dissipative field f_d.

    Similar to _alpha, but returns an affine function of u1 (and constants)
    in each region. This is what creates the nonlinearity in the dissipative
    part of the model.

    Args:
        u1: first input variable.
        u2: second input variable.
        m:  model parameter from hr_params['m'].

    Returns:
        beta(u1, u2): value selected from the piecewise definition.
    """
    conds = [
        jnp.logical_and(u1 >= 1, jnp.logical_and(u2 > -1, u2 < 1)),
        jnp.logical_and(u1 >= 1, u2 <= -1),
        jnp.logical_and(jnp.logical_and(u1 > -1, u1 < 1), u2 >= 1),
        jnp.logical_and(jnp.logical_and(u1 > -1, u1 < 1), jnp.logical_and(u2 > -1, u2 < 1)),
        jnp.logical_and(jnp.logical_and(u1 > -1, u1 < 1), u2 <= -1),
        jnp.logical_and(u1 <= -1, u2 >= 1),
        jnp.logical_and(u1 <= -1, jnp.logical_and(u2 > -1, u2 < 1)),
    ]
    choices = [
        2 * m * (u1 - 1),  # region 1
        -4 * m,            # region 2
        -2 * m * (u1 - 1), # region 3
        0.,                # region 4
        -2 * m * (u1 + 1), # region 5
        4 * m,             # region 6
        2 * m * (u1 + 1),  # region 7
    ]
    return jnp.select(conds, choices, default=0.)


def f_c_fn(e, x, hr_params):
    """
    Compute the **conservative** vector field f_c(e, x) for the error system.

    This part of the dynamics is supposed to match J ∇H in the pH formulation.
    It uses the analytical expressions derived from the HR-like model parameters.

    Args:
        e:  error state vector, shape (5,)
            ordered as [e_x, e_y, e_z, e_u, e_phi]
        x:  full state vector, shape (10,)
            ordered as in data loading: x1,y1,z1,u1,phi1,x2,y2,z2,u2,phi2
        hr_params: dict with physical/model parameters
                   (expects keys: 'k','f','rho','d','r','s')

    Returns:
        jnp.array of shape (5,) representing f_c(e, x).
    """
    # pick the components we need
    e_x, e_y, e_u, e_phi = e[0], e[1], e[3], e[4]
    x1, u1 = x[0], x[3]

    # unpack parameters
    k, f, rho, d, r, s = \
        hr_params['k'], hr_params['f'], hr_params['rho'], hr_params['d'], hr_params['r'], hr_params['s']

    # analytical expression
    return jnp.array([
        # ė_x (conservative part)
        e_y + 2 * k * f * u1 * x1 * e_u + rho * x1 * e_phi,
        # ė_y
        -2 * d * x1 * e_x,
        # ė_z
        r * s * e_x,
        # ė_u
        e_x,
        # ė_phi
        e_x
    ])


def f_d_fn(e, x, hr_params):
    """
    Compute the **dissipative** vector field f_d(e, x) for the error system.

    This part of the dynamics is supposed to match -R ∇H in the pH formulation.
    It contains nonlinear and piecewise terms based on the state (x1, u1, phi1, u2)
    and the helper functions _alpha and _beta.

    Args:
        e:  error state vector, shape (5,)
        x:  full state vector, shape (10,)
        hr_params: dict with physical/model parameters
                   (expects keys: 'a','b','k','h','f','rho','ge','r','q','m')

    Returns:
        jnp.array of shape (5,) representing f_d(e, x).
    """
    # unpack error components
    e_x, e_y, e_z, e_u, e_phi = e[0], e[1], e[2], e[3], e[4]
    # pick needed state components
    x1, u1, phi1, u2 = x[0], x[3], x[4], x[8]

    # unpack parameters
    a, b, k, h, f, rho, g_e, r, q_param, m = \
        hr_params['a'], hr_params['b'], hr_params['k'], hr_params['h'], \
            hr_params['f'], hr_params['rho'], hr_params['ge'], hr_params['r'], \
            hr_params['q'], hr_params['m']

    # nonlinear coefficient multiplying e_x
    N_val = -3 * a * x1 ** 2 + 2 * b * x1 + k * h + k * f * u1 ** 2 + rho * phi1 - 2 * g_e

    # piecewise terms for the u-component of the dissipative field
    alpha_val = _alpha(u1, u2, m)
    beta_val = _beta(u1, u2, m)

    return jnp.array([
        N_val * e_x,            # dissipative term for e_x
        -e_y,                   # damping for e_y
        -r * e_z,               # damping for e_z
        alpha_val * e_u + beta_val,  # nonlinear term for e_u
        -q_param * e_phi        # damping for e_phi
    ])


def align_to_reference(ref: jax.Array, pred: jax.Array):
    """
    Align a predicted signal to a reference by optionally flipping its sign.

    This is useful when the learned Hamiltonian (or dH/dt) is only determined
    up to a sign due to the training dynamics. We:
      1. center both signals,
      2. pick s = sign(<ref, pred>),
      3. apply s to pred,
      4. restore original mean.

    This stays JIT-safe and avoids divisions.

    Args:
        ref:  reference array
        pred: predicted array to be aligned

    Returns:
        pred_aligned: pred after sign alignment and re-centering
        s:            scalar sign (+1 or -1) that was used
    """
    ref_c  = ref  - jnp.mean(ref)
    pred_c = pred - jnp.mean(pred)
    # choose sign based on correlation
    s = jnp.sign(jnp.vdot(ref_c, pred_c))
    # avoid 0 sign
    s = jnp.where(s == 0.0, 1.0, s)
    pred_aligned = s * pred_c + jnp.mean(ref)
    return pred_aligned, s


# --- New loss function: raw-data pH losses only (no StateNN, no normalization) ---


@eqx.filter_jit
def loss_fn_ph(
    model: Combined_pH_PINN,
    e_true_batch, x_true_batch, e_dot_true_batch, H_true_batch,
    lambda_conservative: float, lambda_dissipative: float, lambda_physics: float,
    lambda_j_structure: float, lambda_r_structure: float,
    hr_params: dict, I_ext: jax.Array, xi: jax.Array  # I_ext/xi kept in signature for minimal diff
):
    """
    Composite physics-informed loss for the port-Hamiltonian PINN.

    This version works directly on **raw** data (no extra normalization, no StateNN)
    and enforces several structure-related terms:

    - physics loss (structure):  (ė_true - (J - R) ∇H)²
    - conservative structure:    (f_c(e, x) - J ∇H)²
    - dissipative structure:     (f_d(e, x) + R ∇H)²
    - conservative Lie derivative: (∇H · f_c)²  ≈ 0
    - dissipative dH/dt:         (∇H · ė_true - ∇H · f_d)²
    - Hamiltonian monitor:        (H_pred_aligned - H_true)²  (not weighted in total_loss)

    The lambda_* arguments control the relative weight of those terms in the total loss.
    """
    # Build raw context inputs (x1,u1,u2,phi1) from raw x
    # These indices must match the structure created in generate_data(...)
    x1 = x_true_batch[:, 0]
    u1 = x_true_batch[:, 3]
    u2 = x_true_batch[:, 8]
    phi1 = x_true_batch[:, 4]
    ctx_batch = jnp.stack([x1, u1, u2, phi1], axis=1)  # shape (B,4)

    # grad only w.r.t. e (ctx treated constant)
    def grad_H_single(e_raw, ctx_raw):
        # differentiate model.hamiltonian_net(e, ctx) w.r.t. e only
        return jax.grad(lambda ee: model.hamiltonian_net(ee, ctx_raw))(e_raw)

    # ∇H(e, ctx) for the whole batch
    grad_H = jax.vmap(grad_H_single)(e_true_batch, ctx_batch)
    # learned J(e, ctx) and R(e, ctx)
    J = jax.vmap(model.j_net)(e_true_batch, ctx_batch)
    R = jax.vmap(model.dissipation_net)(e_true_batch, ctx_batch)

    # Physical structure flow: ė = (J - R) ∇H
    e_dot_from_structure = jax.vmap(lambda j, r, g: (j - r) @ g)(J, R, grad_H)

    # Analytical fields (raw data) computed from the known model
    f_c_batch = jax.vmap(f_c_fn, in_axes=(0, 0, None))(e_true_batch, x_true_batch, hr_params)
    f_d_batch = jax.vmap(f_d_fn, in_axes=(0, 0, None))(e_true_batch, x_true_batch, hr_params)

    # --- Individual loss terms ---

    # 1) main physics loss: learned structure vs true derivative
    loss_phys = jnp.mean((e_dot_true_batch - e_dot_from_structure) ** 2)

    # 2) J-structure loss: J ∇H should match analytical conservative field f_c
    j_grad_h = jax.vmap(lambda j, g: j @ g)(J, grad_H)
    loss_j_structure = jnp.mean((f_c_batch - j_grad_h) ** 2)

    # 3) R-structure loss: -R ∇H should match analytical dissipative field f_d
    r_grad_h = jax.vmap(lambda r, g: -r @ g)(R, grad_H)
    loss_r_structure = jnp.mean((f_d_batch - r_grad_h) ** 2)

    # 4) conservative (Lie derivative) loss: ∇H · f_c should be zero
    lie_derivative = jax.vmap(jnp.dot)(grad_H, f_c_batch)
    loss_conservative = jnp.mean(lie_derivative ** 2)

    # 5) dissipative dH/dt loss: true dH/dt = ∇H · ė_true should match ∇H · f_d
    dHdt_true = jax.vmap(jnp.dot)(grad_H, e_dot_true_batch)
    dHdt_eq   = jax.vmap(jnp.dot)(grad_H, f_d_batch)
    loss_dissipative = jnp.mean((dHdt_true - dHdt_eq) ** 2)

    # --- Hamiltonian monitoring (not included in total loss) ---
    H_pred = jax.vmap(lambda e, ctx: model.hamiltonian_net(e, ctx))(e_true_batch, ctx_batch)
    H_pred_aligned, _ = align_to_reference(H_true_batch, H_pred)
    loss_hamiltonian = jnp.mean((H_pred_aligned - H_true_batch) ** 2)

    # --- Weighted sum of the structure/physics losses ---
    total_loss = (
        (lambda_conservative * loss_conservative)
        + (lambda_dissipative * loss_dissipative)
        + (lambda_physics * loss_phys)
        + (lambda_j_structure * loss_j_structure)
        + (lambda_r_structure * loss_r_structure)
    )

    # Collect all terms (training loop expects these keys)
    loss_components = {
        "total": total_loss,
        "phys": loss_phys,
        "conservative": loss_conservative,
        "dissipative": loss_dissipative,
        "j_structure": loss_j_structure,
        "r_structure": loss_r_structure,
        "hamiltonian": loss_hamiltonian,
    }
    return total_loss, loss_components


@eqx.filter_jit
def train_step(
    model, opt_state, optimizer,
    e_batch, x_batch, e_dot_batch, H_batch,
    lambda_conservative, lambda_dissipative, lambda_physics,
    lambda_j_structure, lambda_r_structure,
    hr_params, I_ext, xi
):
    """
    Perform one optimization step on a single batch.

    - runs the physics-informed loss
    - computes gradients (filtered for eqx)
    - applies Optax updates
    - returns updated model + optimizer state + loss values
    """
    (loss_val, loss_components), grads = eqx.filter_value_and_grad(loss_fn_ph, has_aux=True)(
        model,
        e_batch, x_batch, e_dot_batch, H_batch,
        lambda_conservative, lambda_dissipative, lambda_physics,
        lambda_j_structure, lambda_r_structure,
        hr_params, I_ext, xi
    )
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_val, loss_components


@eqx.filter_jit
def evaluate_model(
    model,
    e_batch, x_batch, e_dot_batch, H_batch,
    lambda_conservative, lambda_dissipative, lambda_physics,
    lambda_j_structure, lambda_r_structure,
    hr_params, I_ext, xi
):
    """
    Evaluate the model on a (validation) batch without doing an update.

    Returns:
        loss_val: scalar total loss
        loss_components: dict with the same keys as in training
    """
    loss_val, loss_components = loss_fn_ph(
        model,
        e_batch, x_batch, e_dot_batch, H_batch,
        lambda_conservative, lambda_dissipative, lambda_physics,
        lambda_j_structure, lambda_r_structure,
        hr_params, I_ext, xi
    )
    return loss_val, loss_components


# ==============================================================================
# 4. MAIN EXECUTION LOGIC
# ==============================================================================
def main():
    """Main function to run training, evaluation, visualization, and data export."""

    # ==========================================================================
    # --- Centralized Training and Model Configuration ---
    # ==========================================================================
    # All hyperparameters, paths, and model settings are collected in one dict
    # so they can be logged or overridden (e.g. by Optuna) in one place.
    TRAIN_CONFIG = {
        # --- General Setup ---
        "seed": 42,
        # Path to the pre-generated dataset (see generate_data_for_pH_PINN.py)
        "data_file_path": os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', '..', 'results', 'PINN Data/', 'error_system_data.pkl'
        ),
        # Directory to write plots and temporary results
        "output_dir": os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', '..', 'results', 'temp/'
        ),

        # --- Training Hyperparameters ---
        # Note: dataset is large, so batch_size is also large
        "batch_size": 160000,
        "validation_split": 0.2,   # 80% train, 20% val
        "epochs": 3000,

        # --- Optimizer and Learning Rate Schedule ---
        # Linear decay from 1e-3 to 1e-5 over 2000 steps
        "initial_learning_rate": 1e-3,
        "end_learning_rate": 1e-5,
        "decay_steps": 2000,

        # --- Physics Loss Hyperparameters ---
        # These are the weights used in loss_fn_ph(...)
        "lambda_conservative_max": 0.1,
        "lambda_dissipative_max": 0.1,
        "lambda_physics_max": 0.5,
        "lambda_j_structure_max": 1,
        "lambda_r_structure_max": 1,

        # --- System Parameters ---
        # Start from DEFAULT_PARAMS and override 'ge' as in the original script
        "hr_params": {
            **DEFAULT_PARAMS,
            'ge': 0.62,
        },
        # Kept in signature for compatibility, not used in main loss above
        "I_ext": jnp.array([0.8, 0.8]),
        "xi": jnp.array([[0, 1], [1, 0]]),

        # --- Visualization ---
        # Which run from the pickle to visualize at the end
        "run_to_visualize_idx": 0,
        # Which portion of the time series to plot (fractions)
        "vis_start_ratio": 0,
        "vis_end_ratio": 1,

        # --- Neural Network Architectures ---
        # Passed directly to Combined_pH_PINN
        "nn": {
            "hamiltonian_net": {
                "hidden_sizes": [2048],
                "epsilon": 0.001
            },
            "dissipation_net": {
                "hidden_sizes": [2, 2],
            },
            "j_net": {
                "hidden_sizes": [32, 32, 32, 32],
            },
            "activation": jax.nn.tanh,
        }
    }

    # --- (Optional) Load best params from Optuna study ---
    # Left commented to preserve original behaviour. If you want to enable it:
    # read_config_from_study = True
    # if read_config_from_study:
    #     TRAIN_CONFIG = load_best_config_from_study(TRAIN_CONFIG, objective='h_loss')
    # TRAIN_CONFIG["epochs"] = 1000
    # print(TRAIN_CONFIG)
    # ==========================================================================

    # --- Setup ---
    # create PRNGs
    key = jax.random.PRNGKey(TRAIN_CONFIG["seed"])
    model_key, data_key = jax.random.split(key)
    # system / physics parameters
    hr_params = TRAIN_CONFIG["hr_params"]
    I_ext = TRAIN_CONFIG["I_ext"]
    xi = TRAIN_CONFIG["xi"]

    # --- Generate and Prepare Data (RAW) ---
    # Load concatenated data from disk
    t, e, x, e_dot_true, x_dot_true, H_analytical, dHdt_analytical = generate_data(
        TRAIN_CONFIG["data_file_path"]
    )
    if t is None:
        # generate_data() already printed the message, so just exit
        sys.exit("Exiting: Data loading failed.")

    # Shuffle the whole dataset once before splitting
    num_samples = e.shape[0]
    perm = jax.random.permutation(data_key, num_samples)
    # Shuffle RAW (no normalization here)
    t_shuffled, e_shuffled, x_shuffled, e_dot_shuffled, x_dot_shuffled, H_shuffled = (
        t[perm],
        e[perm],
        x[perm],
        e_dot_true[perm],
        x_dot_true[perm],
        H_analytical[perm],
    )

    # Train/val split
    split_idx = int(num_samples * (1 - TRAIN_CONFIG["validation_split"]))
    e_train, e_val = jnp.split(e_shuffled, [split_idx])
    x_train, x_val = jnp.split(x_shuffled, [split_idx])
    e_dot_train, e_dot_val = jnp.split(e_dot_shuffled, [split_idx])
    H_train, H_val = jnp.split(H_shuffled, [split_idx])

    # --- Initialize Model ---
    # State dimension is inferred from the data (should be 5)
    e_dim = e_train.shape[1]

    # Keep nn_config intact for minimal diff; model only uses H/J/R blocks.
    nn_config = TRAIN_CONFIG["nn"]

    # Build the combined pH-PINN model
    model = Combined_pH_PINN(key=model_key, config=nn_config, state_dim=e_dim)

    # --- Training Loop configuration ---
    batch_size = TRAIN_CONFIG["batch_size"]
    epochs = TRAIN_CONFIG["epochs"]
    lambda_conservative = TRAIN_CONFIG["lambda_conservative_max"]
    lambda_dissipative = TRAIN_CONFIG["lambda_dissipative_max"]
    lambda_physics = TRAIN_CONFIG["lambda_physics_max"]
    lambda_j_structure = TRAIN_CONFIG["lambda_j_structure_max"]
    lambda_r_structure = TRAIN_CONFIG["lambda_r_structure_max"]

    # Number of training samples
    N_train = e_train.shape[0]
    # ceil division to get number of batches
    num_batches = (N_train + batch_size - 1) // batch_size

    if num_batches == 0 and e_train.shape[0] > 0:
        # If batch_size is larger than the dataset, do a single batch
        print(f"Warning: batch_size ({batch_size}) > num samples. Setting num_batches to 1.")
        num_batches = 1

    # Learning-rate schedule (linear decay)
    lr_schedule = optax.linear_schedule(
        init_value=TRAIN_CONFIG["initial_learning_rate"],
        end_value=TRAIN_CONFIG["end_learning_rate"],
        transition_steps=TRAIN_CONFIG["decay_steps"]
    )
    # AdamW optimizer (weight decay variant of Adam)
    optimizer = optax.adamw(learning_rate=lr_schedule)
    # Optax needs a filtered version of the model parameters
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # For logging / plotting
    train_losses, val_losses = [], []
    phys_losses, conservative_losses, dissipative_losses, hamiltonian_losses = [], [], [], []
    j_structure_losses, r_structure_losses = [], []

    # Keep track of the best model according to *Hamiltonian* monitoring loss
    best_model, best_val_loss, best_h_loss = model, jnp.inf, jnp.inf

    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        # Shuffle training data every epoch
        key, shuffle_key = jax.random.split(key)
        perm = jax.random.permutation(shuffle_key, e_train.shape[0])
        e_shuf, x_shuf, e_dot_shuf, H_shuf = (
            e_train[perm],
            x_train[perm],
            e_dot_train[perm],
            H_train[perm],
        )

        # Initialize epoch loss accumulators (sample-weighted)
        epoch_losses = {
            k: 0.0
            for k in [
                "total", "phys", "conservative", "dissipative",
                "j_structure", "r_structure", "hamiltonian"
            ]
        }
        samples_accum = 0

        # --- iterate over all mini-batches ---
        for i in range(num_batches):
            start = i * batch_size
            end = min(start + batch_size, N_train)
            e_b, x_b = e_shuf[start:end], x_shuf[start:end]
            e_dot_b, H_b = e_dot_shuf[start:end], H_shuf[start:end]

            # One training step
            model, opt_state, train_loss_val, loss_comps = train_step(
                model, opt_state, optimizer,
                e_b, x_b, e_dot_b, H_b,
                lambda_conservative, lambda_dissipative, lambda_physics,
                lambda_j_structure, lambda_r_structure,
                hr_params, I_ext, xi
            )

            # Accumulate (loss_comps are *means*, so weight by batch size)
            bsz_eff = e_b.shape[0]
            for k, v in loss_comps.items():
                epoch_losses[k] += float(v) * int(bsz_eff)
            samples_accum += int(bsz_eff)

        # Sample-weighted epoch averages
        avg_losses = {k: epoch_losses[k] / samples_accum for k in epoch_losses}

        # Validation over full validation set
        val_loss, val_loss_comps = evaluate_model(
            model,
            e_val, x_val, e_dot_val, H_val,
            lambda_conservative, lambda_dissipative, lambda_physics,
            lambda_j_structure, lambda_r_structure,
            hr_params, I_ext, xi
        )

        # Save for plots
        train_losses.append(avg_losses["total"])
        val_losses.append(val_loss)
        phys_losses.append(avg_losses["phys"])
        conservative_losses.append(avg_losses["conservative"])
        dissipative_losses.append(avg_losses["dissipative"])
        hamiltonian_losses.append(avg_losses["hamiltonian"])
        j_structure_losses.append(avg_losses["j_structure"])
        r_structure_losses.append(avg_losses["r_structure"])

        # Best model tracking (by Hamiltonian monitor on train, as before)
        train_h_loss = avg_losses['hamiltonian']
        if val_loss < best_val_loss:
            best_h_loss = train_h_loss
            best_val_loss = val_loss
            best_model = model

        # (Alternative: track by train_h_loss, left commented to preserve original code)
        # if train_h_loss < best_h_loss:
        #     best_h_loss = train_h_loss
        #     best_val_loss = val_loss
        #     best_model = model

        # Per-epoch logging
        if (epoch + 1) % 1 == 0 or epoch == 0:
            log_str = (
                f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_losses['total']:.4f} | Val Loss: {val_loss:.4f} | "
                f"H_Loss: {avg_losses['hamiltonian']:.4f} | "
                f"PH: {avg_losses['phys']:.4f} | J_Struct: {avg_losses['j_structure']:.4f} | "
                f"R_Struct: {avg_losses['r_structure']:.4f} | "
                f"Cons: {avg_losses['conservative']:.4f} | Diss: {avg_losses['dissipative']:.4f}"
            )
            print(log_str)

    print("Training finished.")
    print(f"Best validation loss achieved: {best_val_loss:.6f}")
    print(f"Best hamiltonian loss achieved: {best_h_loss:.6f}")

    # ==============================================================================
    # 5. VISUALIZATION AND ANALYSIS
    # ==============================================================================
    output_dir = TRAIN_CONFIG["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    run_to_visualize_idx = TRAIN_CONFIG["run_to_visualize_idx"]
    print(f"\nGenerating visualization plots for simulation run #{run_to_visualize_idx + 1}...")

    # Reload the original multi-run file to get one clean, unshuffled run
    with open(TRAIN_CONFIG["data_file_path"], 'rb') as f:
        all_runs = pickle.load(f)

    # Safety check: if user asked for an out-of-bounds run, fall back to 0
    if run_to_visualize_idx >= len(all_runs):
        print(
            f"Error: 'run_to_visualize_idx' ({run_to_visualize_idx}) is out of bounds. "
            f"Max is {len(all_runs) - 1}. Setting to 0."
        )
        run_to_visualize_idx = 0

    vis_results = all_runs[run_to_visualize_idx]

    # Use the selected run's data (same structure as in generate_data)
    t_test = jnp.asarray(vis_results['t']).reshape(-1, 1)
    e_test = jnp.vstack([
        vis_results['e_x'], vis_results['e_y'], vis_results['e_z'],
        vis_results['e_u'], vis_results['e_phi']
    ]).T
    x_test = jnp.vstack([
        vis_results['x1'], vis_results['y1'], vis_results['z1'], vis_results['u1'], vis_results['phi1'],
        vis_results['x2'], vis_results['y2'], vis_results['z2'], vis_results['u2'], vis_results['phi2']
    ]).T
    e_dot_test = jnp.vstack([
        vis_results['d_e_x'], vis_results['d_e_y'], vis_results['d_e_z'],
        vis_results['d_e_u'], vis_results['d_e_phi']
    ]).T
    H_analytical_vis = jnp.asarray(vis_results['Hamiltonian'])

    # Build raw ctx for viz (same 4 features)
    x1v  = x_test[:, 0]
    u1v  = x_test[:, 3]
    u2v  = x_test[:, 8]
    phi1v = x_test[:, 4]
    ctx_v = jnp.stack([x1v, u1v, u2v, phi1v], axis=1)

    # Analytical derivatives from raw states (f_c + f_d)
    f_c_batch_vis = jax.vmap(f_c_fn, in_axes=(0, 0, None))(e_test, x_test, hr_params)
    f_d_batch_vis = jax.vmap(f_d_fn, in_axes=(0, 0, None))(e_test, x_test, hr_params)
    e_dot_from_equations = f_c_batch_vis + f_d_batch_vis

    # PH structure derivative (raw) using the best_model we tracked
    def grad_H_single_v(e_raw, ctx_raw):
        return jax.grad(lambda ee: best_model.hamiltonian_net(ee, ctx_raw))(e_raw)

    grad_H_v = jax.vmap(grad_H_single_v)(e_test, ctx_v)  # ∇H(e,ctx)
    J_v = jax.vmap(best_model.j_net)(e_test, ctx_v)
    R_v = jax.vmap(best_model.dissipation_net)(e_test, ctx_v)
    e_dot_from_structure = jax.vmap(lambda j, r, g: (j - r) @ g)(J_v, R_v, grad_H_v)

    # --- Plot 1: Learned vs Analytical Hamiltonian (raw) ---
    print("Comparing learned Hamiltonian with analytical solution...")
    H_learned = jax.vmap(lambda e, ctx: best_model.hamiltonian_net(e, ctx))(e_test, ctx_v)
    H_learned_aligned, sign_H = align_to_reference(H_analytical_vis, H_learned)

    # If user wants to plot only a sub-range of the time series
    split_start = int(len(t_test) * TRAIN_CONFIG["vis_start_ratio"])
    split_end = int(len(t_test) * TRAIN_CONFIG["vis_end_ratio"])

    plt.figure(figsize=(12, 7))
    plt.plot(
        t_test[split_start:split_end],
        H_analytical_vis[split_start:split_end],
        label='Analytical Hamiltonian',
        color='blue'
    )
    plt.plot(
        t_test[split_start:split_end],
        H_learned_aligned[split_start:split_end],
        label='Learned Hamiltonian (Aligned)',
        color='red'
    )
    plt.title("Time Evolution of Hamiltonians", fontsize=16)
    plt.xlabel("Time", fontsize=14)
    plt.ylabel("Hamiltonian Value", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'hamiltonian_comparison.png'), dpi=300)
    plt.tight_layout()

    # --- Plot 2: Training, Validation, and Physics Losses ---
    plt.figure(figsize=(14, 9))
    plt.plot(train_losses, label='Total Training Loss', color='black', linewidth=2.5)
    plt.plot(val_losses, label='Total Validation Loss', color='firebrick', linewidth=2.5)

    # PH Structure Losses
    plt.plot(phys_losses, label='PH Structure Loss (phys)', color='purple', alpha=0.9)
    plt.plot(j_structure_losses, label='J Structure Loss', color='brown', alpha=0.7)
    plt.plot(r_structure_losses, label='R Structure Loss', color='magenta', alpha=0.7)

    # Core Physics Losses
    plt.plot(conservative_losses, label='Conservative Loss', color='green', alpha=0.9)
    plt.plot(dissipative_losses, label='Dissipative Loss', color='darkcyan', alpha=0.9)

    # Monitoring Loss
    plt.plot(hamiltonian_losses, label='Hamiltonian Loss (Monitor)', color='red')

    plt.yscale('log')
    plt.title('Training, Validation, and Physics Losses Over Epochs', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (Log Scale)', fontsize=12)
    plt.legend(fontsize=10, loc='lower left', ncol=2)
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.savefig(os.path.join(output_dir, 'training_losses_detailed.png'), dpi=300)
    plt.tight_layout()

    # --- Plot 3: Derivative Comparison (Physics Fidelity) for Error States ---
    fig, axes = plt.subplots(e_test.shape[1], 1, figsize=(12, 12), sharex=True)
    state_labels_e_dot = [r'$\dot{e}_x$', r'$\dot{e}_y$', r'$\dot{e}_z$', r'$\dot{e}_u$', r'$\dot{e}_\phi$']
    fig.suptitle("Error Derivative Fidelity Comparison (e_dot)", fontsize=18, y=0.99)

    for i in range(e_test.shape[1]):
        axes[i].plot(
            t_test[split_start:split_end],
            e_dot_test[split_start:split_end, i],
            label='True Derivative',
            color='blue',
            linewidth=2,
            alpha=0.8
        )
        axes[i].plot(
            t_test[split_start:split_end],
            e_dot_from_structure[split_start:split_end, i],
            label='pH-PINN',
            color='red'
        )

        axes[i].set_ylabel(state_labels_e_dot[i], fontsize=14)
        axes[i].grid(True)
        axes[i].legend(loc='upper right')

    axes[-1].set_xlabel("Time", fontsize=14)
    fig.savefig(os.path.join(output_dir, 'error_derivative_fidelity.png'), dpi=300)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # === Plot 4: dH/dt (analytical vs predictions without autodiff) ============
    dHdt_analytical_vis = jnp.asarray(vis_results['dHdt'])  # shape [T]

    # ∇H·ė_true
    dHdt_pred_true = jnp.einsum('ti,ti->t', grad_H_v, e_dot_test)
    # -∇Hᵀ R ∇H  (dissipative form)
    dHdt_pred_pH = -jnp.einsum('ti,tij,tj->t', grad_H_v, R_v, grad_H_v)

    # Align for nicer visualization (optional)
    dHdt_true_aligned, _ = align_to_reference(dHdt_analytical_vis, dHdt_pred_true)
    dHdt_pH_aligned, _ = align_to_reference(dHdt_analytical_vis, dHdt_pred_pH)

    plt.figure(figsize=(12, 6))
    plt.plot(
        t_test[split_start:split_end],
        np.asarray(dHdt_analytical_vis[split_start:split_end]),
        label=r'$\dot H$ (analytical)'
    )
    # If desired, the true-aligned curve can also be shown:
    # plt.plot(t_test[split_start:split_end], np.asarray(dHdt_true_aligned[split_start:split_end]), ...)
    plt.plot(
        t_test[split_start:split_end],
        np.asarray(dHdt_pH_aligned[split_start:split_end]),
        label=r'$-(\nabla H)^\top R\,\nabla H$'
    )
    plt.xlabel('Time')
    plt.ylabel(r'$\dot H$')
    plt.title(r'$\dot H$: analytical vs. predictions')
    plt.legend(loc='best', frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dHdt_comparison.png'), dpi=300)

    # --- Mean J and R matrices over the whole dataset (use raw e/x) ----------
    print("Computing mean J and R matrices over the whole dataset...")

    # Concatenate train and val to get all samples
    x_all = jnp.concatenate([x_train, x_val], axis=0)
    e_all = jnp.concatenate([e_train, e_val], axis=0)
    # Build context for all
    x1_all  = x_all[:, 0]
    u1_all  = x_all[:, 3]
    u2_all  = x_all[:, 8]
    phi1_all = x_all[:, 4]
    ctx_all = jnp.stack([x1_all, u1_all, u2_all, phi1_all], axis=1)

    # Evaluate J and R for all samples
    J_stack = jax.vmap(best_model.j_net)(e_all, ctx_all)           # [N, 5, 5]
    R_stack = jax.vmap(best_model.dissipation_net)(e_all, ctx_all) # [N, 5, 5]

    # Mean across the dataset -> gives an idea of the learned structure
    J_mean = jnp.mean(J_stack, axis=0)  # [5,5]
    R_mean = jnp.mean(R_stack, axis=0)  # [5,5]

    # Plot helpers
    var_labels = [r'$e_x$', r'$e_y$', r'$e_z$', r'$e_u$', r'$e_\phi$']

    def plot_matrix_with_numbers(mat, title, filename):
        """
        Utility to plot a 5x5 matrix with text annotations and save to disk.
        """
        plt.figure(figsize=(6.2, 5.4))
        im = plt.imshow(np.asarray(mat), interpolation='nearest', aspect='equal')  # default colormap
        plt.title(title, fontsize=14)
        plt.xticks(np.arange(5), var_labels, rotation=0)
        plt.yticks(np.arange(5), var_labels)
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel('Mean value', rotation=90, va='center')

        mat_np = np.asarray(mat)
        vmin, vmax = float(np.min(mat_np)), float(np.max(mat_np))
        mid = (vmin + vmax) / 2.0
        # Write numeric values in each cell
        for i in range(mat_np.shape[0]):
            for j in range(mat_np.shape[1]):
                val = mat_np[i, j]
                txt_color = 'white' if val > mid else 'black'
                plt.text(j, i, f"{val:.3g}", ha='center', va='center', color=txt_color, fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=300)

    # Save mean J and R visualizations
    plot_matrix_with_numbers(J_mean, "Mean J matrix (dataset-wide)", "J_mean_matrix.png")
    plot_matrix_with_numbers(R_mean, "Mean R matrix (dataset-wide)", "R_mean_matrix.png")

    # -------------------------------------------------------------------------
    # Close all figures to free resources
    plt.close('all')
    print(f"All plots saved to {output_dir}")

    # ===== SAVE SPECIFIC PLOTTING DATA =====
    # This block exports a pickle with *exactly* the arrays needed to regenerate
    # plots elsewhere (e.g. in a notebook or separate script).

    def _to_numpy_safe(x):
        """Convert JAX/NumPy array-likes to plain NumPy (no copy if not needed)."""
        import jax.numpy as jnp
        import numpy as np
        if isinstance(x, jnp.ndarray):
            return np.asarray(x)
        return np.asarray(x)

    # Save alongside other PINN Data so plotting scripts can find it
    OUT_PATH = os.path.join(
        Path(__file__).resolve().parents[2] / "results" / "PINN Data",
        "pinn_plot_data.pkl"
    )

    print(f"\nSaving specific data required for plots to {OUT_PATH}...")

    payload = {
        # --- Time Vector & Plotting Range ---
        't_test': _to_numpy_safe(t_test),
        'split_start': split_start,
        'split_end': split_end,

        # --- Loss Histories ---
        'train_losses': _to_numpy_safe(train_losses),
        'val_losses': _to_numpy_safe(val_losses),
        'hamiltonian_losses': _to_numpy_safe(hamiltonian_losses),
        'phys_losses': _to_numpy_safe(phys_losses),
        'conservative_losses': _to_numpy_safe(conservative_losses),
        'dissipative_losses': _to_numpy_safe(dissipative_losses),
        'j_structure_losses': _to_numpy_safe(j_structure_losses),
        'r_structure_losses': _to_numpy_safe(r_structure_losses),

        # --- Hamiltonian Plot Data ---
        'H_analytical_vis': _to_numpy_safe(H_analytical_vis),
        'H_learned_aligned': _to_numpy_safe(H_learned_aligned),

        # --- Derivative Fidelity Plot Data (Error system only) ---
        'e_dot_test': _to_numpy_safe(e_dot_test),
        'e_dot_from_equations': _to_numpy_safe(e_dot_from_equations),
        'e_dot_from_structure': _to_numpy_safe(e_dot_from_structure),

        # --- dH/dt Comparison Plot Data ---
        'dHdt_analytical_vis': _to_numpy_safe(dHdt_analytical_vis),
        'dHdt_pred_true': _to_numpy_safe(dHdt_pred_true),
        'dHdt_pred_pH': _to_numpy_safe(dHdt_pred_pH),

        # --- Mean J and R Matrices ---
        'J_mean': _to_numpy_safe(J_mean),
        'R_mean': _to_numpy_safe(R_mean),
    }

    # Write everything to disk
    with open(OUT_PATH, "wb") as f:
        pickle.dump(payload, f)

    print(f"✅ Saved plotting data -> {OUT_PATH}")
    # ===== END SAVE BLOCK =====


if __name__ == "__main__":
    # Allow running this file directly: trains the model and writes outputs.
    main()
