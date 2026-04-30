import numpy as np
from scipy.integrate import solve_ivp

# Numerical stability epsilon
EPS = 1e-9


# =============================================================================
# COMPETITION MODEL (Assembly/Import Competition)
# =============================================================================

def competition_model_odes(t, y, params, delta_f_B, delta_n_A, delta_n_B):
    """
    ODE system for the competition model.
    
    Two proteins (A, B) compete for a limiting assembly factor F that facilitates
    nuclear import. Only protein-factor complexes are imported.
    
    Parameters
    ----------
    t : float
        Time
    y : array-like
        State vector [A_f, B_f, A_n, B_n]
        - A_f, B_f: free cytoplasmic proteins
        - A_n, B_n: nuclear proteins
    params : dict
        Model parameters containing:
        - k_s: synthesis rate
        - k_import: import rate constant
        - F_total: total factor concentration
        - K_D: dissociation constant
        - delta_f_A: cytoplasmic degradation rate for A
    delta_f_B : float
        Cytoplasmic degradation rate for B
    delta_n_A : float
        Nuclear degradation rate for A
    delta_n_B : float
        Nuclear degradation rate for B
        
    Returns
    -------
    list
        Time derivatives [dA_f/dt, dB_f/dt, dA_n/dt, dB_n/dt]
    """
    A_f, B_f, A_n, B_n = y
    A_f = max(A_f, EPS)
    B_f = max(B_f, EPS)
    
    # Denominator for import flux (from conservation of F)
    denom = max(params['K_D'] + A_f + B_f, EPS)
    
    # Import fluxes (Michaelis-Menten-like competition)
    flux_A = params['k_import'] * params['F_total'] * A_f / denom
    flux_B = params['k_import'] * params['F_total'] * B_f / denom
    
    # ODEs
    dA_f = params['k_s'] - flux_A - params['delta_f_A'] * A_f
    dB_f = params['k_s'] - flux_B - delta_f_B * B_f
    dA_n = flux_A - delta_n_A * A_n
    dB_n = flux_B - delta_n_B * B_n
    
    return [dA_f, dB_f, dA_n, dB_n]


def integrate_competition_to_steady_state(params, delta_f_B, delta_n_A, delta_n_B, 
                                          t_max=200.0, atol=1e-8, rtol=1e-6):
    """
    Find steady-state concentrations for the competition model.
    
    Parameters
    ----------
    params : dict
        Competition model parameters
    delta_f_B, delta_n_A, delta_n_B : float
        Degradation rates
    t_max : float
        Integration time to reach steady state
    atol, rtol : float
        Absolute and relative tolerances
        
    Returns
    -------
    ndarray
        Steady-state concentrations [A_f, B_f, A_n, B_n]
    """
    y0 = np.ones(4)
    rhs = lambda t, y: competition_model_odes(t, y, params, delta_f_B, delta_n_A, delta_n_B)
    sol = solve_ivp(rhs, (0, t_max), y0, method='LSODA', atol=atol, rtol=rtol)
    return np.clip(sol.y[:, -1], EPS, None)


def simulate_competition_trajectory(params, delta_f_B, delta_n_A, delta_n_B, 
                                   t_end, y0, n_points=200, atol=1e-8, rtol=1e-6):
    """
    Simulate time course for the competition model.
    
    Parameters
    ----------
    params : dict
        Competition model parameters
    delta_f_B, delta_n_A, delta_n_B : float
        Degradation rates
    t_end : float
        End time for simulation
    y0 : array-like
        Initial conditions [A_f, B_f, A_n, B_n]
    n_points : int
        Number of time points to return
    atol, rtol : float
        Integration tolerances
        
    Returns
    -------
    tuple
        (trajectory, time_points)
        - trajectory: array of shape (4, n_points) with [A_f, B_f, A_n, B_n] over time
        - time_points: array of time values
    """
    if t_end <= 0.0:
        return np.array([y0]).T, np.array([0.0])
    
    dense_times = np.linspace(0.0, t_end, n_points)
    rhs = lambda t, y: competition_model_odes(t, y, params, delta_f_B, delta_n_A, delta_n_B)
    sol = solve_ivp(rhs, (0.0, t_end), y0, t_eval=dense_times, method='LSODA', 
                    atol=atol, rtol=rtol)
    return np.clip(sol.y, EPS, None), dense_times


# =============================================================================
# ASYMMETRIC FEEDBACK MODEL (Rectified Transcriptional Regulation)
# =============================================================================

def asymmetric_feedback_odes(t, y, k_s_max, S, Total_ref, delta_A, delta_B):
    """
    ODE system for asymmetric feedback model.
    
    Synthesis rate is regulated by total protein level, but only responds
    when total drops BELOW a reference level (rectified feedback).
    
    Parameters
    ----------
    t : float
        Time
    y : array-like
        State vector [A_n, B_n] (nuclear concentrations only)
    k_s_max : float
        Maximum synthesis rate
    S : float
        Feedback sensitivity (half-saturation constant)
    Total_ref : float
        Reference total protein level (threshold for feedback activation)
    delta_A, delta_B : float
        Degradation rates for A and B
        
    Returns
    -------
    list
        Time derivatives [dA_n/dt, dB_n/dt]
    """
    A_n, B_n = y
    A_n = max(A_n, EPS)
    B_n = max(B_n, EPS)
    total = A_n + B_n
    
    # Asymmetric feedback: only active when below threshold
    if total < Total_ref:
        k_s = k_s_max * S / (S + total)
    else:
        k_s = k_s_max * S / (S + Total_ref)
    
    dA_n = k_s - delta_A * A_n
    dB_n = k_s - delta_B * B_n
    
    return [dA_n, dB_n]


def integrate_feedback_to_steady_state(k_s_max, S, Total_ref, delta_A, delta_B,
                                       t_max=1000, atol=1e-8, rtol=1e-6):
    """
    Find steady-state concentrations for the feedback model.
    
    Parameters
    ----------
    k_s_max, S, Total_ref : float
        Feedback model parameters
    delta_A, delta_B : float
        Degradation rates
    t_max : float
        Integration time to reach steady state
    atol, rtol : float
        Integration tolerances
        
    Returns
    -------
    ndarray
        Steady-state concentrations [A_n, B_n]
    """
    y0 = [1.0, 1.0]
    rhs = lambda t, y: asymmetric_feedback_odes(t, y, k_s_max, S, Total_ref, delta_A, delta_B)
    sol = solve_ivp(rhs, (0, t_max), y0, method='LSODA', atol=atol, rtol=rtol)
    return np.clip(sol.y[:, -1], EPS, None)


def simulate_feedback_trajectory(k_s_max, S, Total_ref, delta_A, delta_B,
                                t_end, y0, n_points=200, atol=1e-8, rtol=1e-6):
    """
    Simulate time course for the feedback model.
    
    Parameters
    ----------
    k_s_max, S, Total_ref : float
        Feedback model parameters
    delta_A, delta_B : float
        Degradation rates
    t_end : float
        End time for simulation
    y0 : array-like
        Initial conditions [A_n, B_n]
    n_points : int
        Number of time points
    atol, rtol : float
        Integration tolerances
        
    Returns
    -------
    tuple
        (trajectory, time_points)
        - trajectory: array of shape (2, n_points) with [A_n, B_n] over time
        - time_points: array of time values
    """
    if t_end <= 0.0:
        return np.array([y0]).T, np.array([0.0])
    
    t_eval = np.linspace(0, t_end, n_points)
    rhs = lambda t, y: asymmetric_feedback_odes(t, y, k_s_max, S, Total_ref, delta_A, delta_B)
    sol = solve_ivp(rhs, (0, t_end), y0, t_eval=t_eval, method='LSODA', 
                    atol=atol, rtol=rtol)
    return np.clip(sol.y, EPS, None), sol.t
