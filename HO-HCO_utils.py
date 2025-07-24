# Utils for the numerical simulations of the article 
# "Explosive adoption of corrupt behaviors in social systems with higher-order interactions" 
# by Bretón-Fuertes et al.

import numpy as np
from scipy.integrate import solve_ivp

# Iteration of the HO-HCO system according to Eqs. (1)-(5)

def evolve_system(
    beta_m, mu_m, k_m, r,
    rho_C0=0.5, rho_H0=0.5,
    tol=1e-8, max_iter=20000
):
    """
    Simulates the temporal evolution of the system according to Eqs. (1)-(5) 

    Parameters
    ----------
    beta_m : array-like
        Vector with values beta^(1) and beta^(2).
    mu_m : array-like
        Vector with values mu^(1) and mu^(2).
    k_m : array-like
        Vector with degrees k^(1) and k^(2).
    r : float
        Reinsertion rate from O to H.
    rho_C0, rho_H0 : float
        Initial conditions.
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    rho_C_hist, rho_H_hist, rho_O_hist : arrays
        Temporal evolution of the variables.
    """
    
    rho_C = rho_C0
    rho_H = rho_H0
    rho_O = 1 - rho_C - rho_H
    
    rho_C_hist = [rho_C]
    rho_H_hist = [rho_H]
    rho_O_hist = [rho_O]
    
    for it in range(max_iter):
        # Π^{H→C}(t)
        prod_HC = ((1 - beta_m[0] * rho_C) ** k_m[0])*((1 - beta_m[1] * rho_C * rho_C) ** k_m[1])
        Pi_HC = 1 - prod_HC
        
        # Π^{C→O}(t)
        prod_CO = np.prod((1 - mu_m[0] * rho_H) ** k_m[0])*((1 - mu_m[1] * rho_H * rho_H) ** k_m[1])
        Pi_CO = 1 - prod_CO
        
        rho_C_new = (1 - Pi_CO) * rho_C + Pi_HC * rho_H
        rho_H_new = (1 - Pi_HC) * rho_H + r * rho_O
        rho_O_new = 1 - rho_C_new - rho_H_new
        
        diff = abs(rho_C_new - rho_C) + abs(rho_H_new - rho_H)
        
        rho_C, rho_H, rho_O = rho_C_new, rho_H_new, rho_O_new
        
        rho_C_hist.append(rho_C)
        rho_H_hist.append(rho_H)
        rho_O_hist.append(rho_O)
        
        if diff < tol:
            break
    
    return np.array(rho_C_hist), np.array(rho_H_hist), np.array(rho_O_hist)


def compute_stationary_state(beta_m, mu_m, k_m, r, max_iter=20000,rho_C0=0.5, rho_H0=0.5):
    """
    Returns the stationary value.
    """
    rho_C_hist, rho_H_hist, rho_O_hist = evolve_system(beta_m, mu_m, k_m, r, max_iter=max_iter,rho_C0=rho_C0, rho_H0=rho_H0)
    return rho_C_hist[-1], rho_H_hist[-1], rho_O_hist[-1]

# Integration of the HO-HC system according to Eq. (11)

def d_rho_dt(t, rho, beta1, beta2, mu1, mu2, k1, k2):
    """
    Computes the time derivative dρ/dt at time t for a given ρ(t),
    according to Eq. (11)

    Parameters:
        t : float
            Time (required by solve_ivp, unused in the computation itself).
        rho : float
            Fraction of corrupt individuals at time t.
        beta1, beta2 : float
            Corruption rates for pairwise and triplet interactions.
        mu1, mu2 : float
            Betrayal rates for pairwise and triplet interactions.
        k1, k2 : int or float
            Average number of pairwise and triplet contacts.

    Returns:
        float : Derivative dρ/dt at time t.
    """
    term1 = beta1 * k1 * rho * (1 - rho)
    term2 = beta2 * k2 * rho**2 * (1 - rho)
    term3 = mu1 * k1 * (1 - rho) * rho
    term4 = mu2 * k2 * (1 - rho)**2 * rho
    return term1 + term2 - term3 - term4


def integrate_rho(beta1, beta2, mu1, mu2, k1, k2, rho0=0.01, t_span=(0, 200), t_eval=None):
    """
    Simulates the temporal evolution of the system according to Eq. (11)

    Parameters:
        beta1, beta2, mu1, mu2, k1, k2 : model parameters (see d_rho_dt).
        rho0 : float
            Initial value of ρ(t).
        t_span : tuple
            Integration time interval (start, end).
        t_eval : array-like or None
            Time points at which to evaluate the solution. If None, defaults to 500 points.

    Returns:
        t : ndarray
            Array of time points.
        rho_t : ndarray
            Array of corresponding ρ(t) values.
    """
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 500)

    # Wrap the ODE with fixed parameters
    def rhs(t, rho): return d_rho_dt(t, rho[0], beta1, beta2, mu1, mu2, k1, k2)

    sol = solve_ivp(rhs, t_span, [rho0], t_eval=t_eval)
    return sol.t, sol.y[0]


def compute_stationary_state(beta1, beta2, mu1, mu2, k1, k2, rho0=0.1, t_max=50, num_points=500):
    """
    Returns the stationary value.
    """
    t_eval = np.linspace(0, t_max, num_points)
    _, rho_t = integrate_rho(beta1, beta2, mu1, mu2, k1, k2, rho0=rho0, t_span=(0, t_max), t_eval=t_eval)
    return rho_t[-1]