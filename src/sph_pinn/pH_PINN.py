import jax, jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib.pyplot as plt
import pickle
import sys
from src.hr_model.model import DEFAULT_PARAMS
import os

# JAX configuration to use 64-bit precision.
jax.config.update("jax_enable_x64", True)


# ==============================================================================
# 1. NEURAL NETWORK DEFINITIONS
# ==============================================================================

class FourierFeatures(eqx.Module):
    """Encodes a 1D input into a higher-dimensional space using Fourier features."""
    b_matrix: jax.Array
    output_size: int = eqx.field(static=True)

    def __init__(self, key, in_size=1, mapping_size=32, scale=1):
        n_pairs = mapping_size // 2
        self.b_matrix = jax.random.normal(key, (n_pairs, in_size)) * scale
        self.output_size = n_pairs * 2

    def __call__(self, t):
        # CORRECTED: Handle scalar inputs (ndim=0) which occur during jacobian calculation.
        # This ensures `t` is always a 2D array before the matrix multiplication.
        if t.ndim == 0:
            t = t.reshape(1, 1)
        elif t.ndim == 1:
            t = t[None, :]

        t_proj = t @ self.b_matrix.T
        return jnp.concatenate([jnp.sin(t_proj), jnp.cos(t_proj)], axis=-1).squeeze()


class StateNN(eqx.Module):
    """An MLP with Fourier Features to approximate the combined state [q(t), s(t)]."""
    layers: list
    activation: callable

    def __init__(self, key, out_size, width, depth, activation, mapping_size, scale):
        fourier_key, *layer_keys = jax.random.split(key, depth + 1)
        self.activation = activation

        fourier_layer = FourierFeatures(fourier_key, in_size=1, mapping_size=mapping_size, scale=scale)

        self.layers = [
            fourier_layer,
            eqx.nn.Linear(fourier_layer.output_size, width, key=layer_keys[0]),
            *[eqx.nn.Linear(width, width, key=key) for i in range(1, depth - 1)],
            eqx.nn.Linear(width, out_size, key=layer_keys[-1])
        ]

    def __call__(self, t):
        x = self.layers[0](t)
        for layer in self.layers[1:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)


# --- sPHNN Component Networks (Adapted for Windowed Data with LSTMs) ---

class _FICNN(eqx.Module):
    """Internal helper class for a Fully Input Convex Neural Network."""
    w_layers: list
    u_layers: list
    final_layer: eqx.nn.Linear
    activation: callable = eqx.field(static=True)

    def __init__(self, key, in_size: int, out_size: int, width: int, depth: int, activation: callable):
        self.activation = activation
        keys = jax.random.split(key, depth)
        self.w_layers = [eqx.nn.Linear(in_size, width, key=keys[0])]
        self.w_layers.extend([eqx.nn.Linear(in_size, width, key=key) for key in keys[1:-1]])
        self.u_layers = [eqx.nn.Linear(width, width, use_bias=False, key=key) for key in keys[1:-1]]
        self.final_layer = eqx.nn.Linear(width, out_size, use_bias=False, key=keys[-1])

    def __call__(self, s):
        z = self.activation(self.w_layers[0](s))
        for i in range(len(self.u_layers)):
            # Enforce non-negative weights for convexity
            u_layer_non_negative = eqx.tree_at(lambda l: l.weight, self.u_layers[i], jnp.abs(self.u_layers[i].weight))
            z = self.activation(u_layer_non_negative(z) + self.w_layers[i + 1](s))
        return self.final_layer(z)[0]


class HamiltonianNN(eqx.Module):
    """
    Learns a convex Hamiltonian H(s) using an LSTM to process history and
    a FICNN to guarantee convexity.
    MODIFIED: Now outputs a sequence of H values for the entire window.
    """
    lstm: eqx.nn.LSTMCell
    ficnn: _FICNN
    state_dim: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)

    def __init__(self, key, state_dim, hidden_size, ficnn_width, ficnn_depth, ficnn_activation):
        lstm_key, ficnn_key = jax.random.split(key)

        self.state_dim = state_dim
        self.hidden_size = hidden_size

        self.lstm = eqx.nn.LSTMCell(input_size=state_dim, hidden_size=hidden_size, key=lstm_key)

        ficnn_in_size = state_dim + hidden_size
        self.ficnn = _FICNN(ficnn_key, in_size=ficnn_in_size, out_size=1,
                            width=ficnn_width, depth=ficnn_depth, activation=ficnn_activation)

    def __call__(self, s_window):
        h0 = jnp.zeros((self.hidden_size,))
        c0 = jnp.zeros((self.hidden_size,))
        initial_carry = (h0, c0)

        def lstm_scan(carry, x):
            new_carry = self.lstm(x, carry)
            return new_carry, new_carry[0]

        # CHANGED: Capture the entire sequence of hidden states from the LSTM
        _, hidden_states_sequence = jax.lax.scan(lstm_scan, initial_carry, s_window)

        # CHANGED: Create an augmented input for every time step in the window
        augmented_input_sequence = jnp.concatenate([s_window, hidden_states_sequence], axis=-1)

        # CHANGED: Apply the FICNN to each time step in the sequence using vmap
        return jax.vmap(self.ficnn)(augmented_input_sequence)


class DissipationNN(eqx.Module):
    """
    Learns a positive semi-definite dissipation matrix R(s) = L(s)L(s)^T
    using an LSTM followed by an MLP.
    MODIFIED: Now outputs a sequence of R matrices for the entire window.
    """
    lstm: eqx.nn.LSTMCell
    layers: list
    activation: callable
    state_dim: int = eqx.field(static=True)

    def __init__(self, key, state_dim, hidden_size, width, depth, activation):
        self.state_dim = state_dim
        self.activation = activation

        # CORRECTED: Changed `self.num_l_elements` to a local variable `num_l_elements`
        num_l_elements = state_dim * (state_dim + 1) // 2

        lstm_key, *layer_keys = jax.random.split(key, depth + 1)

        self.lstm = eqx.nn.LSTMCell(input_size=state_dim, hidden_size=hidden_size, key=lstm_key)

        self.layers = [
            eqx.nn.Linear(hidden_size, width, key=layer_keys[0]),
            *[eqx.nn.Linear(width, width, key=key) for key in layer_keys[1:-1]],
            eqx.nn.Linear(width, num_l_elements, key=layer_keys[-1])
        ]

    def _apply_mlp(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)

    def _construct_matrix(self, l_elements):
        L = jnp.zeros((self.state_dim, self.state_dim))
        tril_indices = jnp.tril_indices(self.state_dim)
        L = L.at[tril_indices].set(l_elements)

        positive_diag = jax.nn.softplus(jnp.diag(L))
        L = L.at[jnp.diag_indices(self.state_dim)].set(positive_diag)
        return L @ L.T

    def __call__(self, s_window):
        h0 = jnp.zeros((self.lstm.hidden_size,))
        c0 = jnp.zeros((self.lstm.hidden_size,))
        initial_carry = (h0, c0)

        def lstm_scan(carry, x):
            new_carry = self.lstm(x, carry)
            return new_carry, new_carry[0]

        # CHANGED: Capture the sequence of hidden states
        _, hidden_states_sequence = jax.lax.scan(lstm_scan, initial_carry, s_window)

        # CHANGED: Apply MLP and matrix construction to the entire sequence
        l_elements_seq = jax.vmap(self._apply_mlp)(hidden_states_sequence)
        R_matrix_seq = jax.vmap(self._construct_matrix)(l_elements_seq)

        return R_matrix_seq


class DynamicJ_NN(eqx.Module):
    """
    Learns a skew-symmetric structure matrix J(s) using an LSTM
    followed by an MLP.
    MODIFIED: Now outputs a sequence of J matrices for the entire window.
    """
    lstm: eqx.nn.LSTMCell
    layers: list
    activation: callable
    state_dim: int = eqx.field(static=True)

    def __init__(self, key, state_dim, hidden_size, width, depth, activation):
        self.state_dim = state_dim
        self.activation = activation

        # CORRECTED: Changed `self.num_unique_elements` to a local variable `num_unique_elements`
        num_unique_elements = state_dim * (state_dim - 1) // 2

        lstm_key, *layer_keys = jax.random.split(key, depth + 1)

        self.lstm = eqx.nn.LSTMCell(input_size=state_dim, hidden_size=hidden_size, key=lstm_key)

        self.layers = [
            eqx.nn.Linear(hidden_size, width, key=layer_keys[0]),
            *[eqx.nn.Linear(width, width, key=key) for key in layer_keys[1:-1]],
            eqx.nn.Linear(width, num_unique_elements, key=layer_keys[-1])
        ]

    def _apply_mlp(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)

    def _construct_matrix(self, upper_triangle_elements):
        J = jnp.zeros((self.state_dim, self.state_dim))
        triu_indices = jnp.triu_indices(self.state_dim, k=1)
        J = J.at[triu_indices].set(upper_triangle_elements)
        return J - J.T

    def __call__(self, s_window):
        h0 = jnp.zeros((self.lstm.hidden_size,))
        c0 = jnp.zeros((self.lstm.hidden_size,))
        initial_carry = (h0, c0)

        def lstm_scan(carry, x):
            new_carry = self.lstm(x, carry)
            return new_carry, new_carry[0]

        # CHANGED: Capture the sequence of hidden states
        _, hidden_states_sequence = jax.lax.scan(lstm_scan, initial_carry, s_window)

        # CHANGED: Apply MLP and matrix construction to the entire sequence
        upper_tri_elems_seq = jax.vmap(self._apply_mlp)(hidden_states_sequence)
        J_matrix_seq = jax.vmap(self._construct_matrix)(upper_tri_elems_seq)

        return J_matrix_seq


# --- The Combined Model ---
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
# 2. DATA HANDLING (omitted for brevity, no changes)
# ==============================================================================
def generate_data(file_path="error_system_data.pkl"):
    """
    Loads and prepares training data from a pre-generated pickle file containing
    multiple simulation runs.
    """
    print(f"Loading simulation data from {file_path}...")
    try:
        with open(file_path, 'rb') as f:
            # The file contains a list of result dictionaries
            all_runs_results = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}")
        print("Please run 'generate_data_for_PINN.py' to create the data file.")
        return None, None, None, None, None

    # Initialize lists to hold data from all runs
    all_t, all_s, all_q, all_s_dot, all_H = [], [], [], [], []

    # Process each simulation run
    for i, results in enumerate(all_runs_results):
        print(f"  ... processing run {i + 1}/{len(all_runs_results)}")

        # Extract data for the current run
        t = jnp.asarray(results['t'])
        s = jnp.vstack([
            results['e_x'], results['e_y'], results['e_z'],
            results['e_u'], results['e_phi']
        ]).T
        q = jnp.vstack([
            results['x1'], results['y1'], results['z1'], results['u1'], results['phi1'],
            results['x2'], results['y2'], results['z2'], results['u2'], results['phi2']
        ]).T
        s_dot_true = jnp.vstack([
            results['d_e_x'], results['d_e_y'], results['d_e_z'],
            results['d_e_u'], results['d_e_phi']
        ]).T
        H_analytical = jnp.asarray(results['Hamiltonian'])

        # Append to the main lists
        all_t.append(t)
        all_s.append(s)
        all_q.append(q)
        all_s_dot.append(s_dot_true)
        all_H.append(H_analytical)

    # Concatenate all runs into single arrays
    final_t = jnp.concatenate(all_t)
    final_s = jnp.concatenate(all_s)
    final_q = jnp.concatenate(all_q)
    final_s_dot = jnp.concatenate(all_s_dot)
    final_H = jnp.concatenate(all_H)

    print("Data loading and aggregation complete.")
    return final_t, final_s, final_q, final_s_dot, final_H


def normalize(data, mean, std):
    """Normalizes data using pre-computed statistics."""
    return (data - mean) / (std + 1e-8)


def denormalize(data, mean, std):
    """Denormalizes data using pre-computed statistics."""
    return data * std + mean


def create_windows(window_size: int, *arrays):
    """
    Creates overlapping windows from a set of time-series arrays using
    JAX-native indexing.
    """
    num_samples = arrays[0].shape[0]
    num_windows = num_samples - window_size + 1

    if num_windows <= 0:
        # Return empty arrays with correct dimensions if the input is smaller than the window
        return tuple(jnp.empty((0, window_size) + arr.shape[1:]) for arr in arrays)

    # Create a 2D array of indices that represents all windows
    start_indices = jnp.arange(num_windows)[:, None]
    window_offsets = jnp.arange(window_size)[None, :]
    indices = start_indices + window_offsets

    # Use the indices to gather the windowed data from each array
    windowed_arrays = [arr[indices] for arr in arrays]

    return tuple(windowed_arrays)


# ==============================================================================
# 3. TRAINING LOGIC (omitted for brevity, no changes)
# ==============================================================================
# --- Helper functions for the new physics-based loss terms ---

def _alpha(u1, u2, m):
    """Helper function for the dissipative field f_d."""
    conds = [
        jnp.logical_and(u1 >= 1, jnp.logical_and(u2 > -1, u2 < 1)),
        jnp.logical_and(u1 >= 1, u2 <= -1),
        jnp.logical_and(jnp.logical_and(u1 > -1, u1 < 1), u2 >= 1),
        jnp.logical_and(jnp.logical_and(u1 > -1, u1 < 1), jnp.logical_and(u2 > -1, u2 < 1)),
        jnp.logical_and(jnp.logical_and(u1 > -1, u1 < 1), u2 <= -1),
        jnp.logical_and(u1 <= -1, u2 >= 1),
        jnp.logical_and(u1 <= -1, jnp.logical_and(u2 > -1, u2 < 1)),
    ]
    choices = [2 * m - 1., -1., -1., 2 * m - 1., -1., -1., 2 * m - 1.]
    return jnp.select(conds, choices, default=-1.)


def _beta(u1, u2, m):
    """Helper function for the dissipative field f_d."""
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
        2 * m * (u1 - 1), -4 * m, -2 * m * (u1 - 1), 0.,
        -2 * m * (u1 + 1), 4 * m, 2 * m * (u1 + 1),
    ]
    return jnp.select(conds, choices, default=0.)


def f_c_fn(e, q, hr_params):
    """Calculates the conservative vector field f_c(e)."""
    e_x, e_y, e_u, e_phi = e[0], e[1], e[3], e[4]
    x1, u1 = q[0], q[3]

    k, f, rho, d, r, s = \
        hr_params['k'], hr_params['f'], hr_params['rho'], hr_params['d'], hr_params['r'], hr_params['s']

    return jnp.array([
        e_y + 2 * k * f * u1 * x1 * e_u + rho * x1 * e_phi,
        -2 * d * x1 * e_x,
        r * s * e_x,
        e_x,
        e_x
    ])


def f_d_fn(e, q, hr_params):
    """Calculates the dissipative vector field f_d(e)."""
    e_x, e_y, e_z, e_u, e_phi = e[0], e[1], e[2], e[3], e[4]
    x1, u1, phi1, u2 = q[0], q[3], q[4], q[8]

    a, b, k, h, f, rho, g_e, r, q_param, m = \
        hr_params['a'], hr_params['b'], hr_params['k'], hr_params['h'], \
            hr_params['f'], hr_params['rho'], hr_params['ge'], hr_params['r'], \
            hr_params['q'], hr_params['m']

    N_val = -3 * a * x1 ** 2 + 2 * b * x1 + k * h + k * f * u1 ** 2 + rho * phi1 - 2 * g_e
    alpha_val = _alpha(u1, u2, m)
    beta_val = _beta(u1, u2, m)

    return jnp.array([
        N_val * e_x,
        -e_y,
        -r * e_z,
        alpha_val * e_u + beta_val,
        -q_param * e_phi
    ])


@eqx.filter_jit
def loss_fn(model: Combined_sPHNN_PINN, t_batch_norm, s_true_batch_norm, q_true_batch_norm, s_dot_true_batch_norm,
            H_true_batch_norm,
            lambda_conservative: float, lambda_dissipative: float, lambda_physics: float, hr_params: dict,
            t_mean, t_std, s_mean, s_std, q_mean, q_std, s_dot_mean, s_dot_std, H_mean, H_std):
    """
    Calculates composite loss based on the ENTIRE time step of each window.
    """
    # --- Part 1: State Prediction for the entire window ---
    batch_size, window_size = t_batch_norm.shape[0], t_batch_norm.shape[1]
    t_batch_flat = t_batch_norm.reshape(-1, 1)
    all_states_pred_flat = jax.vmap(model.state_net)(t_batch_flat)
    all_states_pred_windows_norm = all_states_pred_flat.reshape(batch_size, window_size, -1)

    q_pred_windows_norm = all_states_pred_windows_norm[:, :, :10]
    s_pred_windows_norm = all_states_pred_windows_norm[:, :, 10:]

    # --- Part 2: Unified Data Fidelity Loss (on entire window) ---
    all_states_true_windows_norm = jnp.concatenate([q_true_batch_norm, s_true_batch_norm], axis=-1)
    data_loss = jnp.mean((all_states_pred_windows_norm - all_states_true_windows_norm) ** 2)

    # --- Part 3: Denormalize values for physics calculations ---
    # Use true `s` for stability during training, but predicted `q`
    s_true_windows = denormalize(s_true_batch_norm, s_mean, s_std)
    q_pred_windows = denormalize(q_pred_windows_norm, q_mean, q_std)

    # --- Part 4: Physics Calculations (vmapped over batch and window) ---

    # Calculate Jacobian of H_seq w.r.t s_seq to get grad_H at each time step
    def get_grad_H(s_window_norm):
        jac_H_fn = jax.jacfwd(model.hamiltonian_net)
        # Jacobian has shape (window_size_out, window_size_in, state_dim)
        # We need the diagonal part where input time step matches output time step
        return jnp.diagonal(jac_H_fn(s_window_norm), axis1=0, axis2=1).T

    grad_H_norm = jax.vmap(get_grad_H)(s_true_batch_norm)  # Use true s for stability
    grad_H = grad_H_norm / (s_std + 1e-8)

    # Get s_dot from autodiff of the StateNN
    def get_s_dot_autodiff(t_window_norm):
        jac_s_fn = jax.jacfwd(lambda t: model.state_net(t)[10:])
        # The jacobian of f(scalar) -> vector(5) is a vector of shape (5,).
        # Vmapping over the 64 time steps gives the correct (64, 5) shape.
        # CORRECTED: Removed the unnecessary .squeeze(-1) at the end.
        return jax.vmap(jac_s_fn)(t_window_norm.squeeze(-1))

    s_dot_autodiff_norm = jax.vmap(get_s_dot_autodiff)(t_batch_norm)
    s_dot_autodiff = s_dot_autodiff_norm * (s_std / (t_std + 1e-8))

    # Calculate analytical physics terms for the full window
    # CORRECTED: Added in_axes=(0, 0, None) to the outer vmap to handle the static hr_params
    f_c_batch = jax.vmap(jax.vmap(f_c_fn, in_axes=(0, 0, None)), in_axes=(0, 0, None))(s_true_windows, q_pred_windows,
                                                                                       hr_params)
    f_d_batch = jax.vmap(jax.vmap(f_d_fn, in_axes=(0, 0, None)), in_axes=(0, 0, None))(s_true_windows, q_pred_windows,
                                                                                       hr_params)

    # --- Part 5: Loss Components (calculated on entire window) ---
    # Physics Structure Loss
    J = jax.vmap(model.j_net)(s_true_batch_norm)
    R = jax.vmap(model.dissipation_net)(s_true_batch_norm)

    # vmap the mat-vec product (J-R) @ grad_H over the window dimension
    # CORRECTED: Use a nested vmap to map over both the batch and window dimensions.
    s_dot_from_structure = jax.vmap(jax.vmap(lambda j, r, g: (j - r) @ g))(J, R, grad_H)
    s_dot_true_batch = denormalize(s_dot_true_batch_norm, s_dot_mean, s_dot_std)
    loss_phys = jnp.mean((s_dot_true_batch - s_dot_from_structure) ** 2)

    # Conservative Loss (Lie Derivative = <grad_H, f_c>)
    # Conservative Loss (Lie Derivative = <grad_H, f_c>)
    # CORRECTED: Replaced the incorrect vmap(dot) with element-wise multiplication and sum.
    lie_derivative = jnp.sum(grad_H * f_c_batch, axis=2)
    loss_conservative = jnp.mean(lie_derivative ** 2)

    # Dissipative Loss
    # CORRECTED: Applied the same fix to the dH/dt calculations.
    dHdt_from_autodiff = jnp.sum(grad_H * s_dot_autodiff, axis=2)
    dHdt_from_equations = jnp.sum(grad_H * f_d_batch, axis=2)
    loss_dissipative = jnp.mean((dHdt_from_autodiff - dHdt_from_equations) ** 2)

    # Hamiltonian Loss (for monitoring)
    H_pred_norm = jax.vmap(model.hamiltonian_net)(s_true_batch_norm)
    H_pred = denormalize(H_pred_norm, H_mean, H_std)
    H_true = denormalize(H_true_batch_norm, H_mean, H_std)

    # Align the whole sequence
    correlation = jnp.corrcoef(H_true.flatten(), H_pred.flatten())[0, 1]
    sign = jnp.sign(correlation)
    H_pred_aligned = sign * H_pred - jnp.mean(sign * H_pred) + jnp.mean(H_true)
    loss_hamiltonian = jnp.mean((H_pred_aligned - H_true) ** 2)

    # --- Part 6: Total Loss ---
    s_dot_from_equations = f_c_batch + f_d_batch
    state_loss = data_loss + jnp.mean((s_dot_true_batch - s_dot_from_equations) ** 2)

    total_loss = (state_loss
                  + (lambda_conservative * loss_conservative)
                  + (lambda_dissipative * loss_dissipative)
                  + (lambda_physics * loss_phys))

    loss_components = {
        "total": total_loss,
        "data_unified": data_loss,
        "phys": loss_phys,
        "conservative": loss_conservative,
        "dissipative": loss_dissipative,
        "hamiltonian": loss_hamiltonian,
    }
    return total_loss, loss_components


@eqx.filter_jit
def train_step(model, opt_state, optimizer, t_batch_norm, s_batch_norm, q_batch_norm, s_dot_batch_norm, H_batch_norm,
               lambda_conservative, lambda_dissipative, lambda_physics, hr_params,
               t_mean, t_std, s_mean, s_std, q_mean, q_std, s_dot_mean, s_dot_std, H_mean, H_std):
    """Performs a single training step on a batch of windows."""
    (loss_val, loss_components), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
        model, t_batch_norm, s_batch_norm, q_batch_norm, s_dot_batch_norm, H_batch_norm,
        lambda_conservative, lambda_dissipative, lambda_physics, hr_params,
        t_mean, t_std, s_mean, s_std, q_mean, q_std, s_dot_mean, s_dot_std, H_mean, H_std
    )
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_val, loss_components


@eqx.filter_jit
def evaluate_model(model, t_batch_norm, s_batch_norm, q_batch_norm, s_dot_batch_norm, H_batch_norm,
                   lambda_conservative, lambda_dissipative, lambda_physics, hr_params,
                   t_mean, t_std, s_mean, s_std, q_mean, q_std, s_dot_mean, s_dot_std, H_mean, H_std):
    """Calculates the loss for the validation set (a batch of windows)."""
    loss_val, _ = loss_fn(
        model, t_batch_norm, s_batch_norm, q_batch_norm, s_dot_batch_norm, H_batch_norm,
        lambda_conservative, lambda_dissipative, lambda_physics, hr_params,
        t_mean, t_std, s_mean, s_std, q_mean, q_std, s_dot_mean, s_dot_std, H_mean, H_std
    )
    return loss_val


# ==============================================================================
# 4. MAIN EXECUTION LOGIC
# ==============================================================================
def main():
    """Main function to run the training and evaluation."""
    # --- Setup and Hyperparameters ---
    key = jax.random.PRNGKey(42)
    model_key, data_key = jax.random.split(key)

    # Training hyperparameters
    window_size = 32
    batch_size = 128
    validation_split = 0.2
    initial_learning_rate = 1e-3
    end_learning_rate = 5e-5
    decay_steps = 3000
    epochs = 5000

    # Physics loss hyperparameters with warmup
    lambda_conservative_max = 1
    lambda_dissipative_max = 5
    lambda_physics_max = 15
    lambda_warmup_epochs = 2000

    # System parameters
    hr_params = DEFAULT_PARAMS.copy()

    # --- Generate and Prepare Data ---
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results', 'PINN Data/',
                        'error_system_data.pkl')
    t, s, q, s_dot_true, H_analytical = generate_data(path)
    if t is None:
        sys.exit("Exiting: Data loading failed.")

    t = t.reshape(-1, 1)

    # --- 1. Create Windowed Data FIRST (from the original, ordered data) ---
    print(f"Creating data windows with size {window_size}...")
    (t_w, s_w, q_w, s_dot_w, H_w) = create_windows(
        window_size, t, s, q, s_dot_true, H_analytical
    )

    # --- 2. Shuffle the WINDOWS, not the individual points ---
    num_windows = t_w.shape[0]
    perm = jax.random.permutation(data_key, num_windows)
    t_w_shuffled, s_w_shuffled, q_w_shuffled, s_dot_w_shuffled, H_w_shuffled = \
        t_w[perm], s_w[perm], q_w[perm], s_dot_w[perm], H_w[perm]

    # --- 3. Split the shuffled windows into training and validation sets ---
    split_idx = int(num_windows * (1 - validation_split))
    t_train_w, t_val_w = jnp.split(t_w_shuffled, [split_idx])
    s_train_w, s_val_w = jnp.split(s_w_shuffled, [split_idx])
    q_train_w, q_val_w = jnp.split(q_w_shuffled, [split_idx])
    s_dot_train_w, s_dot_val_w = jnp.split(s_dot_w_shuffled, [split_idx])
    H_train_w, H_val_w = jnp.split(H_w_shuffled, [split_idx])

    # --- Compute Normalization Stats (from flat training data BEFORE windowing) ---
    num_train_samples = int(s.shape[0] * (1 - validation_split))
    t_train_flat, s_train_flat, q_train_flat, s_dot_train_flat, H_train_flat = \
        t[:num_train_samples], s[:num_train_samples], q[:num_train_samples], \
            s_dot_true[:num_train_samples], H_analytical[:num_train_samples]

    t_mean, t_std = jnp.mean(t_train_flat), jnp.std(t_train_flat)
    s_mean, s_std = jnp.mean(s_train_flat, axis=0), jnp.std(s_train_flat, axis=0)
    q_mean, q_std = jnp.mean(q_train_flat, axis=0), jnp.std(q_train_flat, axis=0)
    s_dot_mean, s_dot_std = jnp.mean(s_dot_train_flat, axis=0), jnp.std(s_dot_train_flat, axis=0)
    H_mean, H_std = jnp.mean(H_train_flat), jnp.std(H_train_flat)

    # --- 4. Normalize the Windowed Data ---
    t_train_norm = normalize(t_train_w, t_mean, t_std)
    s_train_norm = normalize(s_train_w, s_mean, s_std)
    q_train_norm = normalize(q_train_w, q_mean, q_std)
    s_dot_train_norm = normalize(s_dot_train_w, s_dot_mean, s_dot_std)
    H_train_norm = normalize(H_train_w, H_mean, H_std)

    t_val_norm = normalize(t_val_w, t_mean, t_std)
    s_val_norm = normalize(s_val_w, s_mean, s_std)
    q_val_norm = normalize(q_val_w, q_mean, q_std)
    s_dot_val_norm = normalize(s_dot_val_w, s_dot_mean, s_dot_std)
    H_val_norm = normalize(H_val_w, H_mean, H_std)

    # --- Centralized Neural Network Configuration ---
    s_dim = s_train_flat.shape[1]
    q_dim = q_train_flat.shape[1]

    nn_config = {
        "state_dim": s_dim,
        "q_dim": q_dim,
        "state_nn": {
            "width": 128,
            "depth": 3,
            "activation": jax.nn.tanh,
            "fourier_features": {
                "mapping_size": 32,
                "scale": 300,
            }
        },
        "hamiltonian_nn": {
            "hidden_size": 64,
            "ficnn": {
                "width": 128,
                "depth": 3,
                "activation": jax.nn.softplus,
            }
        },
        "dissipation_nn": {
            "hidden_size": 32,
            "width": 8,
            "depth": 3,
            "activation": jax.nn.softplus,
        },
        "j_net": {
            "hidden_size": 32,
            "width": 8,
            "depth": 3,
            "activation": jax.nn.softplus,
        }
    }

    # Initialize the combined model
    model = Combined_sPHNN_PINN(key=model_key, config=nn_config)

    # --- Training Loop (omitted for brevity, no changes) ---
    lr_schedule = optax.linear_schedule(
        init_value=initial_learning_rate,
        end_value=end_learning_rate,
        transition_steps=decay_steps
    )
    optimizer = optax.adamw(learning_rate=lr_schedule)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    train_losses, val_losses = [], []
    phys_losses, conservative_losses, dissipative_losses, hamiltonian_losses = [], [], [], []
    best_model, best_val_loss = model, jnp.inf

    num_batches = t_train_norm.shape[0] // batch_size
    if num_batches == 0 and t_train_norm.shape[0] > 0:
        print(f"Warning: batch_size ({batch_size}) > num_windows. Setting num_batches to 1.")
        num_batches = 1

    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        # Loss weight warmup schedule
        warmup_factor = jnp.minimum(1.0, (epoch + 1) / lambda_warmup_epochs)
        current_lambda_conservative = lambda_conservative_max * warmup_factor
        current_lambda_dissipative = lambda_dissipative_max * warmup_factor
        current_lambda_physics = lambda_physics_max * warmup_factor

        key, shuffle_key = jax.random.split(key)
        perm = jax.random.permutation(shuffle_key, t_train_norm.shape[0])
        t_shuffled, s_shuffled, q_shuffled, s_dot_shuffled, H_shuffled = \
            t_train_norm[perm], s_train_norm[perm], q_train_norm[perm], s_dot_train_norm[perm], H_train_norm[perm]

        epoch_losses = {k: 0.0 for k in ["total", "data_unified", "phys", "conservative", "dissipative", "hamiltonian"]}

        for i in range(num_batches):
            start, end = i * batch_size, (i + 1) * batch_size
            t_b, s_b, q_b, s_dot_b, H_b = t_shuffled[start:end], s_shuffled[start:end], q_shuffled[
                                                                                        start:end], s_dot_shuffled[
                                                                                                    start:end], H_shuffled[
                                                                                                                start:end]

            model, opt_state, train_loss_val, loss_comps = train_step(
                model, opt_state, optimizer, t_b, s_b, q_b, s_dot_b, H_b,
                current_lambda_conservative, current_lambda_dissipative, current_lambda_physics, hr_params,
                t_mean, t_std, s_mean, s_std, q_mean, q_std, s_dot_mean, s_dot_std, H_mean, H_std
            )
            for k in epoch_losses:
                if k in loss_comps:
                    epoch_losses[k] += loss_comps[k]

        avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}

        val_loss = evaluate_model(
            model, t_val_norm, s_val_norm, q_val_norm, s_dot_val_norm, H_val_norm,
            current_lambda_conservative, current_lambda_dissipative, current_lambda_physics, hr_params,
            t_mean, t_std, s_mean, s_std, q_mean, q_std, s_dot_mean, s_dot_std, H_mean, H_std
        )

        train_losses.append(avg_losses["total"])
        val_losses.append(val_loss)
        phys_losses.append(avg_losses["phys"])
        conservative_losses.append(avg_losses["conservative"])
        dissipative_losses.append(avg_losses["dissipative"])
        hamiltonian_losses.append(avg_losses["hamiltonian"])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model

        if (epoch + 1) % 100 == 0 or epoch == 0:
            log_str = (
                f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_losses['total']:.4f} | Val Loss: {val_loss:.4f} | "
                f"Data: {avg_losses['data_unified']:.4f} | "
                f"Phys: {avg_losses['phys']:.4f} | "
                f"Cons: {avg_losses['conservative']:.4f} | Diss: {avg_losses['dissipative']:.4f} | "
                f"H_Loss: {avg_losses['hamiltonian']:.4f}"
            )
            print(log_str)

    print("Training finished.")
    print(f"Best validation loss achieved: {best_val_loss:.6f}")

    # ==============================================================================
    # 5. VISUALIZATION AND ANALYSIS
    # ==============================================================================

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results', 'temp/')
    os.makedirs(output_dir, exist_ok=True)

    run_to_visualize_idx = 0

    print(f"\nGenerating visualization plots for simulation run #{run_to_visualize_idx + 1}...")

    with open(path, 'rb') as f:
        all_runs = pickle.load(f)
    if run_to_visualize_idx >= len(all_runs):
        print(f"Error: 'run_to_visualize_idx' is out of bounds. Setting to 0.")
        run_to_visualize_idx = 0
    vis_results = all_runs[run_to_visualize_idx]

    t_test = jnp.asarray(vis_results['t']).reshape(-1, 1)
    s_test = jnp.vstack(
        [vis_results['e_x'], vis_results['e_y'], vis_results['e_z'], vis_results['e_u'], vis_results['e_phi']]).T
    q_test = jnp.vstack(
        [vis_results['x1'], vis_results['y1'], vis_results['z1'], vis_results['u1'], vis_results['phi1'],
         vis_results['x2'], vis_results['y2'], vis_results['z2'], vis_results['u2'], vis_results['phi2']]).T
    s_dot_test = jnp.vstack([vis_results['d_e_x'], vis_results['d_e_y'], vis_results['d_e_z'], vis_results['d_e_u'],
                             vis_results['d_e_phi']]).T
    H_analytical_vis = jnp.asarray(vis_results['Hamiltonian'])

    t_test_norm = normalize(t_test, t_mean, t_std)

    all_states_pred_norm = jax.vmap(best_model.state_net)(t_test_norm)
    q_pred_norm = all_states_pred_norm[:, :10]
    s_pred_norm = all_states_pred_norm[:, 10:]

    s_pred = denormalize(s_pred_norm, s_mean, s_std)
    q_pred = denormalize(q_pred_norm, q_mean, q_std)

    # --- Create windows for the test data to feed into LSTM models ---
    (s_pred_norm_windows,) = create_windows(window_size, s_pred_norm)

    grad_H_norm_windows = jax.vmap(jax.grad(best_model.hamiltonian_net))(s_pred_norm_windows)
    grad_H_norm = grad_H_norm_windows[:, -1, :] # Take last gradient
    J_norm = jax.vmap(best_model.j_net)(s_pred_norm_windows)
    R_norm = jax.vmap(best_model.dissipation_net)(s_pred_norm_windows)
    
    # We only have predictions for the end of each window, so we align them with the original time axis.
    s_dot_from_structure_norm = jax.vmap(lambda j, r, g: (j - r) @ g)(J_norm, R_norm, grad_H_norm)
    s_dot_from_structure = s_dot_from_structure_norm * s_std

    f_c_batch_vis = jax.vmap(f_c_fn, in_axes=(0, 0, None))(s_pred, q_pred, hr_params)
    f_d_batch_vis = jax.vmap(f_d_fn, in_axes=(0, 0, None))(s_pred, q_pred, hr_params)
    s_dot_from_equations = f_c_batch_vis + f_d_batch_vis

    get_s_slice_autodiff_grad = lambda net, t: jax.jvp(lambda t_scalar: net(t_scalar)[10:], (t,), (jnp.ones_like(t),))[
        1]
    s_dot_autodiff_norm = jax.vmap(get_s_slice_autodiff_grad, in_axes=(None, 0))(best_model.state_net, t_test_norm)
    s_dot_autodiff = s_dot_autodiff_norm * (s_std / (t_std + 1e-8))

    print("Comparing learned Hamiltonian with analytical solution...")
    H_learned_norm = jax.vmap(best_model.hamiltonian_net)(s_pred_norm_windows)
    H_learned_aligned = denormalize(H_learned_norm, H_mean, H_std)
    
    # Align the length of analytical H with the windowed predictions
    H_analytical_vis_aligned = H_analytical_vis[window_size - 1:]
    
    correlation = jnp.corrcoef(H_analytical_vis_aligned.flatten(), H_learned_aligned.flatten())[0, 1]
    sign = jnp.sign(correlation if not jnp.isnan(correlation) else 1.0)
    H_learned_aligned = sign * H_learned_aligned - jnp.mean(sign * H_learned_aligned) + jnp.mean(H_analytical_vis_aligned)
    
    # Adjust time axis for windowed predictions
    t_test_windowed = t_test[window_size-1:]

    plt.figure(figsize=(12, 7))
    plt.plot(t_test[:2000], H_analytical_vis[:2000], label='Analytical Hamiltonian', color='blue')
    plt.plot(t_test_windowed[:2000], H_learned_aligned[:2000], label='Learned Hamiltonian (Aligned)', color='red', linestyle='--')
    plt.title("Time Evolution of Hamiltonians", fontsize=16)
    plt.xlabel("Time", fontsize=14)
    plt.ylabel("Hamiltonian Value", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'hamiltonian_comparison.png'), dpi=300)
    plt.tight_layout()

    plt.figure(figsize=(12, 7))
    plt.plot(train_losses, label='Total Training Loss')
    plt.plot(val_losses, label='Total Validation Loss')
    plt.plot(hamiltonian_losses, label='Hamiltonian Loss', color='red')
    plt.plot(phys_losses, label='Physics Loss', color='purple')
    plt.plot(conservative_losses, label='Conservative Loss', alpha=0.7)
    plt.plot(dissipative_losses, label='Dissipative Loss', alpha=0.7)
    plt.yscale('log')
    plt.title('Training, Validation, and Physics Losses Over Epochs', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (Log Scale)', fontsize=12)
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig(os.path.join(output_dir, 'training_losses.png'), dpi=300)
    plt.tight_layout()

    fig, axes = plt.subplots(s_test.shape[1], 1, figsize=(12, 12), sharex=True)
    state_labels_s_dot = [r'$\dot{e}_x$', r'$\dot{e}_y$', r'$\dot{e}_z$', r'$\dot{e}_u$', r'$\dot{e}_\phi$']
    fig.suptitle("Derivative Fidelity Comparison", fontsize=18, y=0.99)
    for i in range(s_test.shape[1]):
        axes[i].plot(t_test[:2000], s_dot_test[:2000, i], label='True Derivative', color='green', linewidth=3, alpha=0.8)
        axes[i].plot(t_test_windowed[:2000], s_dot_from_structure[:2000, i], label='sPHNN Structure', color='red', linestyle='--')
        axes[i].plot(t_test[:2000], s_dot_from_equations[:2000, i], label='Analytical Eq. (f_c+f_d)', color='purple',
                     linestyle=':')
        axes[i].plot(t_test[:2000], s_dot_autodiff[:2000, i], label='Autodiff', color='orange', linestyle='-.')
        axes[i].set_ylabel(state_labels_s_dot[i], fontsize=14)
        axes[i].grid(True)
        axes[i].legend(loc='upper right')
    axes[-1].set_xlabel("Time", fontsize=14)
    fig.savefig(os.path.join(output_dir, 'derivative_fidelity.png'), dpi=300)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    fig, axes = plt.subplots(s_test.shape[1], 1, figsize=(12, 10), sharex=True)
    state_labels_error = [r'$e_x$', r'$e_y$', r'$e_z$', r'$e_u$', r'$e_\phi$']
    fig.suptitle("Error System State 's' Prediction: True vs. Predicted", fontsize=18, y=0.99)
    for i in range(s_test.shape[1]):
        axes[i].plot(t_test[:2000], s_test[:2000, i], 'b', label='True State', alpha=0.9)
        axes[i].plot(t_test[:2000], s_pred[:2000, i], 'r--', label='Predicted State')
        axes[i].set_ylabel(state_labels_error[i], fontsize=14)
        axes[i].grid(True)
        axes[i].legend(loc='upper right')
    axes[-1].set_xlabel("Time", fontsize=14)
    fig.savefig(os.path.join(output_dir, 'error_state_s_prediction.png'), dpi=300)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    fig, axes = plt.subplots(q_test.shape[1], 1, figsize=(12, 18), sharex=True)
    state_labels_q = [r'$x_1$', r'$y_1$', r'$z_1$', r'$u_1$', r'$\phi_1$', r'$x_2$', r'$y_2$', r'$z_2$', r'$u_2$',
                      r'$\phi_2$']
    fig.suptitle("HR System State 'q' Prediction: True vs. Predicted", fontsize=18, y=0.99)
    for i in range(q_test.shape[1]):
        axes[i].plot(t_test[:2000], q_test[:2000, i], 'b', label='True State', alpha=0.9)
        axes[i].plot(t_test[:2000], q_pred[:2000, i], 'r--', label='Predicted State')
        axes[i].set_ylabel(state_labels_q[i], fontsize=14)
        axes[i].grid(True)
        axes[i].legend(loc='upper right')
    axes[-1].set_xlabel("Time", fontsize=14)
    fig.savefig(os.path.join(output_dir, 'hr_state_q_prediction.png'), dpi=300)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    plt.close('all')
    print(f"All plots saved to {output_dir}")


if __name__ == "__main__":
    main()