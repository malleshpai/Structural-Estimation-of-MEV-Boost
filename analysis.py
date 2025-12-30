"""
Structural Estimation of MEV-Boost Auctions

This module implements the two-stage non-linear least squares estimation procedure
described in "Structural Estimation of MEV-Boost Auctions" to estimate the distribution
of builder values in MEV-Boost auctions.

The model decomposes builder values into:
- Common component: C_t = μ_t(X) + σ_t(X) × υ_t
- Idiosyncratic component: ε_i = ω_t(X) × η_i

Where:
- μ_t(X) = μ_{t,1} X_1 + μ_{t,2} X_2 + μ_{t,3} (linear in log space)
- ω_t(X) = exp{ω_{t,1} X_1 + ω_{t,2} X_2 + ω_{t,3}}
- σ_t(X) = exp{σ_{t,1} X_1 + σ_{t,2} X_2 + σ_{t,3}}

X_1: Absolute ETH-USD price change (in basis points)
X_2: Base fee per gas (in Gwei)
X_3: Constant term

The estimation uses second-highest bids from two types of builders:
- Type 1: Integrated builders (BeaverBuild, RSync Builder)
- Type 2: Neutral builders (Titan, Flashbots)
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm
import multiprocessing
import warnings
import sys
import os

# Suppress optimization warnings that are not critical
# Set up comprehensive warning filters before any operations
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*overflow encountered.*')
warnings.filterwarnings('ignore', message='.*invalid value encountered.*')
warnings.filterwarnings('ignore', message='.*divide by zero encountered.*')
warnings.filterwarnings('ignore', message='.*resource_tracker.*')

# Suppress numpy warnings
np.seterr(all='ignore')

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

print("Loading data...")
data = pd.read_csv("aggregated_bids_with_volatility_and_mempool.csv")

# Convert winning bid amounts from Wei to ETH
data['winning_bid_amount'] = pd.to_numeric(data['winning_bid_amount'], errors='coerce') / 10**18
data['winning_bid_amount_log'] = np.log(data['winning_bid_amount'])

# Define builder types
integrated_builders = ['BeaverBuild_highest_bid', 'RSync Builder_highest_bid'] 
neutral_builders = ['Titan_highest_bid', 'Flashbots_highest_bid'] 

# Convert bid columns to numeric, coerce non-numeric values to NaN
for col in integrated_builders + neutral_builders: 
    data[col] = pd.to_numeric(data[col], errors='coerce') / 10**18

# Filter out rows where any bid is <= 0
mask = (data[integrated_builders + neutral_builders] > 0).all(axis=1)
data = data[mask].copy()

# ============================================================================
# COMPUTE ORDER STATISTICS
# ============================================================================

def get_second_highest_bids(df, builders):
    """
    Extract sorted bids for each auction and return order statistics.
    
    Args:
        df: DataFrame with bid columns
        builders: List of builder column names
        
    Returns:
        Array of sorted bids (highest to lowest) for each auction
    """
    bids_df = df[builders].transpose()
    sorted_bids = np.sort(bids_df.values, axis=0)[::-1]
    return sorted_bids

# Compute order statistics for integrated builders
sorted_integrated_bids = get_second_highest_bids(data, integrated_builders)
data['highest_bid_integrated'] = sorted_integrated_bids[0, :]
data['highest_bid_integrated_log'] = np.log(data["highest_bid_integrated"])
data['2nd_highest_bid_integrated'] = sorted_integrated_bids[1, :]
data['2nd_highest_bid_type_1_log'] = np.log(data['2nd_highest_bid_integrated'])

# Compute order statistics for neutral builders
sorted_neutral_bids = get_second_highest_bids(data, neutral_builders)
data['highest_bid_neutral'] = sorted_neutral_bids[0, :]
data['highest_bid_neutral_log'] = np.log(data["highest_bid_neutral"])
data['2nd_highest_bid_neutral'] = sorted_neutral_bids[1, :]
data['2nd_highest_bid_type_2_log'] = np.log(data['2nd_highest_bid_neutral'])

# ============================================================================
# CREATE EXPLANATORY VARIABLES
# ============================================================================

# Convert absolute price movement into basis points
data['percent_abs_price_change'] = np.abs(
    (data['cex_start_midpoint'] - data['cex_end_midpoint']) / data['cex_start_midpoint']
) * 100
data['bps_abs_price_change'] = np.abs(
    (data['cex_start_midpoint'] - data['cex_end_midpoint']) / data['cex_start_midpoint']
) * 10000
data['abs_price_change'] = np.abs(data['cex_start_midpoint'] - data['cex_end_midpoint'])

# Convert baseFeePerGas from Wei to Gwei
data['base_fee_gwei'] = data['base_fee_per_gas'] / 10**9

# Add constant term
data['constant'] = 1

# Define explanatory variables
explanatory_variables = ['bps_abs_price_change', 'base_fee_gwei', 'constant']

# ============================================================================
# THEORETICAL CONSTANTS FOR ORDER STATISTICS
# ============================================================================

# Expected value of second-highest order statistic from standard normal (r=2, n=2)
# For standard normal: E[η^(2:2)] = -1/√π
expected_second_highest_draw_normal = -1 / np.sqrt(np.pi)

# Store in dictionary for type-specific access
expected_second_highest_draw_values = {
    (1, 2): expected_second_highest_draw_normal,  # Type 1, r=2
    (2, 2): expected_second_highest_draw_normal,  # Type 2, r=2
}

# Variance of second-highest order statistic from standard normal (r=2, n=2)
# For standard normal: V[η^(2:2)] = 1 - 1/π
variance_second_highest_draw = 1 - 1 / np.pi
variance_second_highest_draw_values = [
    variance_second_highest_draw,  # Type 1
    variance_second_highest_draw,   # Type 2
]

# ============================================================================
# STAGE 1: ESTIMATE μ AND ω PARAMETERS
# ============================================================================

def calculate_Z(data, mu, omega, explanatory_variables, expected_second_highest_draw_values, t):
    """
    Calculate squared residuals for Stage 1 estimation.
    
    The objective is to minimize:
    Z = Σ [P_t^(2:n_t) - μ_t(X) - m_(2:2) × ω_t(X)]²
    
    Where:
    - P_t^(2:n_t) is the second-highest bid (in log space)
    - μ_t(X) = μ_{t,1} X_1 + μ_{t,2} X_2 + μ_{t,3} (linear in log space)
    - ω_t(X) = exp{ω_{t,1} X_1 + ω_{t,2} X_2 + ω_{t,3}}
    - m_(2:2) is the expected value of second-highest order statistic
    
    Args:
        data: DataFrame with bid data
        mu: Parameter vector for location (μ_{t,1}, μ_{t,2}, μ_{t,3})
        omega: Parameter vector for idiosyncratic scale (ω_{t,1}, ω_{t,2}, ω_{t,3})
        explanatory_variables: List of variable names
        expected_second_highest_draw_values: Dictionary of expected order statistics
        t: Builder type (1 for integrated, 2 for neutral)
        
    Returns:
        Array of squared residuals
    """
    # Calculate μ_t(X) = μ_{t,1} X_1 + μ_{t,2} X_2 + μ_{t,3}
    mu_term = np.dot(data[explanatory_variables], mu)
    
    # Calculate ω_t(X) = exp{ω_{t,1} X_1 + ω_{t,2} X_2 + ω_{t,3}}
    omega_term = np.exp(np.dot(data[explanatory_variables], omega))
    
    # Calculate Z = [P_t^(2:n_t) - μ_t(X) - m_(2:2) × ω_t(X)]²
    Z = (data[f'2nd_highest_bid_type_{t}_log'] 
         - mu_term 
         - expected_second_highest_draw_values[(t, 2)] * omega_term) ** 2
    
    return Z

def optimization_function_Z(mu_and_omega, data, explanatory_variables, 
                            expected_second_highest_draw_values, t):
    """
    Objective function for Stage 1 optimization.
    
    Minimizes sum of squared residuals across all auctions.
    """
    mu = mu_and_omega[:len(explanatory_variables)]
    omega = mu_and_omega[len(explanatory_variables):]
    Z_sum = np.sum(calculate_Z(data, mu, omega, explanatory_variables, 
                               expected_second_highest_draw_values, t))
    return Z_sum

def calculate_p_bar_and_omega_bar(data, mu, omega, explanatory_variables, 
                                  expected_second_highest_draw_values, t):
    """
    Calculate p_bar and omega_bar for use in Stage 2 estimation.
    
    p_bar = μ_t(X) + m_(2:2) × ω_t(X)
    omega_bar = ω_t(X)
    
    These are stored as new columns in the data DataFrame.
    """
    # Calculate p_bar = μ_t(X) + m_(2:2) × ω_t(X)
    mu_term = np.dot(data[explanatory_variables], mu)
    omega_term = np.exp(np.dot(data[explanatory_variables], omega))
    data.loc[:, f'p_bar_type_{t}'] = (mu_term 
                                       + expected_second_highest_draw_values[(t, 2)] 
                                       * omega_term)
    
    # Calculate omega_bar = ω_t(X)
    data.loc[:, f'omega_bar_type_{t}'] = omega_term

# ============================================================================
# STAGE 2: ESTIMATE σ PARAMETERS
# ============================================================================

def calculate_Y_b_t(data, sigma, explanatory_variables, 
                    variance_second_highest_draw_values, t):
    """
    Calculate squared residuals for Stage 2 estimation.
    
    The objective is to minimize:
    Y = Σ [P_t^(2:n_t)² - p_bar² - δ_(2:2) × omega_bar² - σ_t²(X)]²
    
    Where:
    - p_bar and omega_bar come from Stage 1
    - σ_t²(X) = exp{σ_{t,1} X_1 + σ_{t,2} X_2 + σ_{t,3}}
    - δ_(2:2) is the variance of second-highest order statistic
    
    Args:
        data: DataFrame with bid data (must have p_bar and omega_bar columns)
        sigma: Parameter vector for common shock volatility (σ_{t,1}, σ_{t,2}, σ_{t,3})
        explanatory_variables: List of variable names
        variance_second_highest_draw_values: List of variance constants
        t: Builder type (1 for integrated, 2 for neutral)
        
    Returns:
        Array of squared residuals
    """
    # Calculate σ_t²(X) = exp{σ_{t,1} X_1 + σ_{t,2} X_2 + σ_{t,3}}
    sigma_squared = np.exp(np.dot(data[explanatory_variables], sigma))
    
    # Calculate Y = [P_t^(2:n_t)² - p_bar² - δ_(2:2) × omega_bar² - σ_t²(X)]²
    term = (data[f'2nd_highest_bid_type_{t}_log']**2 
            - data[f'p_bar_type_{t}']**2 
            - variance_second_highest_draw_values[t-1] * data[f'omega_bar_type_{t}']**2 
            - sigma_squared)**2
    
    return term

def optimization_function_Y(sigma, data, explanatory_variables, 
                            variance_second_highest_draw_values, t):
    """
    Objective function for Stage 2 optimization.
    
    Minimizes sum of squared residuals across all auctions.
    """
    Y_values = calculate_Y_b_t(data, sigma, explanatory_variables, 
                             variance_second_highest_draw_values, t)
    return Y_values.sum()

# ============================================================================
# MODEL FIT STATISTICS
# ============================================================================

def calculate_r_squared(data, mu, omega, sigma, explanatory_variables, 
                       expected_second_highest_draw_values, 
                       variance_second_highest_draw_values, t):
    """
    Calculate R-squared for both stages of the regression model.
    
    Returns:
        Dictionary with stage1_r_squared, stage2_r_squared, and overall_r_squared
    """
    # Stage 1: Calculate predicted values
    mu_term = np.dot(data[explanatory_variables], mu)
    omega_term = np.exp(np.dot(data[explanatory_variables], omega))
    predicted_stage1 = (mu_term 
                        + expected_second_highest_draw_values[(t, 2)] * omega_term)
    
    # Actual values
    actual_stage1 = data[f'2nd_highest_bid_type_{t}_log']
    
    # Calculate R-squared for stage 1
    ss_res_stage1 = np.sum((actual_stage1 - predicted_stage1) ** 2)
    ss_tot_stage1 = np.sum((actual_stage1 - np.mean(actual_stage1)) ** 2)
    r_squared_stage1 = 1 - (ss_res_stage1 / ss_tot_stage1)
    
    # Stage 2: Calculate predicted values
    predicted_stage2 = np.exp(np.dot(data[explanatory_variables], sigma))
    
    # For stage 2, we need the residuals from stage 1
    residuals_stage1_squared = (actual_stage1 - predicted_stage1) ** 2
    variance_term = variance_second_highest_draw_values[t-1] * (omega_term ** 2)
    actual_stage2 = residuals_stage1_squared - variance_term
    
    # Calculate R-squared for stage 2
    ss_res_stage2 = np.sum((actual_stage2 - predicted_stage2) ** 2)
    ss_tot_stage2 = np.sum((actual_stage2 - np.mean(actual_stage2)) ** 2)
    r_squared_stage2 = 1 - (ss_res_stage2 / ss_tot_stage2) if ss_tot_stage2 > 0 else 0
    
    return {
        'stage1_r_squared': r_squared_stage1,
        'stage2_r_squared': r_squared_stage2,
        'overall_r_squared': (r_squared_stage1 + r_squared_stage2) / 2
    }

# ============================================================================
# MAIN ESTIMATION FUNCTION
# ============================================================================

def primary_analysis(data, explanatory_variables, initial_guess_mu_omega, initial_guess_sigma):
    """
    Perform two-stage estimation for both builder types.
    
    Args:
        data: DataFrame with bid data
        explanatory_variables: List of variable names
        initial_guess_mu_omega: Initial guess for [μ, ω] parameters (6 values)
        initial_guess_sigma: Initial guess for σ parameters (3 values)
        
    Returns:
        Dictionary with estimated parameters for both types:
        - mu_1, omega_1, sigma_1: Integrated builders (Type 1)
        - mu_2, omega_2, sigma_2: Neutral builders (Type 2)
    """
    # Stage 1: Estimate μ and ω for integrated builders
    first_stage_results_integrated = minimize(
        optimization_function_Z, 
        initial_guess_mu_omega, 
        args=(data, explanatory_variables, expected_second_highest_draw_values, 1), 
        method='L-BFGS-B', 
        options={'maxiter': 5000}
    ).x

    # Stage 1: Estimate μ and ω for neutral builders
    first_stage_results_neutral = minimize(
        optimization_function_Z, 
        initial_guess_mu_omega, 
        args=(data, explanatory_variables, expected_second_highest_draw_values, 2), 
        method='L-BFGS-B', 
        options={'maxiter': 5000}
    ).x

    # Split the parameters into mu and omega
    mu_integrated = first_stage_results_integrated[:len(explanatory_variables)]
    omega_integrated = first_stage_results_integrated[len(explanatory_variables):]
    mu_neutral = first_stage_results_neutral[:len(explanatory_variables)]
    omega_neutral = first_stage_results_neutral[len(explanatory_variables):]

    # Calculate p_bar and omega_bar for use in Stage 2
    calculate_p_bar_and_omega_bar(data, mu_integrated, omega_integrated, 
                                  explanatory_variables, expected_second_highest_draw_values, 1)
    calculate_p_bar_and_omega_bar(data, mu_neutral, omega_neutral, 
                                  explanatory_variables, expected_second_highest_draw_values, 2)

    # Stage 2: Estimate σ for integrated builders
    sigma_integrated = minimize(
        optimization_function_Y, 
        initial_guess_sigma, 
        args=(data, explanatory_variables, variance_second_highest_draw_values, 1), 
        method='L-BFGS-B', 
        options={'maxiter': 5000}
    ).x
    
    # Stage 2: Estimate σ for neutral builders
    sigma_neutral = minimize(
        optimization_function_Y, 
        initial_guess_sigma, 
        args=(data, explanatory_variables, variance_second_highest_draw_values, 2), 
        method='L-BFGS-B', 
        options={'maxiter': 5000}
    ).x
    
    return {
        "mu_1": mu_integrated,
        "omega_1": omega_integrated,
        "sigma_1": sigma_integrated,
        "mu_2": mu_neutral,
        "omega_2": omega_neutral,
        "sigma_2": sigma_neutral
    }

# ============================================================================
# BOOTSTRAP FOR CONFIDENCE INTERVALS
# ============================================================================

def bootstrap_iteration(args):
    """
    Perform one bootstrap iteration.
    
    Args:
        args: Tuple of (data, explanatory_variables, initial_guess_mu_omega, initial_guess_sigma)
        
    Returns:
        Dictionary of estimated parameters
    """
    data, explanatory_variables, initial_guess_mu_omega, initial_guess_sigma = args
    
    # Sample the data with replacement
    bootstrap_sample = data.sample(n=len(data), replace=True)
    
    return primary_analysis(
        data=bootstrap_sample, 
        explanatory_variables=explanatory_variables, 
        initial_guess_mu_omega=initial_guess_mu_omega, 
        initial_guess_sigma=initial_guess_sigma
    )

def calculate_confidence_interval(values, confidence_level=0.95):
    """
    Calculate confidence interval from bootstrap samples.
    
    Args:
        values: Array of bootstrap estimates
        confidence_level: Confidence level (default 0.95)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    lower_percentile = (1 - confidence_level) / 2 * 100
    upper_percentile = (1 + confidence_level) / 2 * 100
    lower_bound = float(np.round(np.percentile(values, lower_percentile), decimals=3))
    upper_bound = float(np.round(np.percentile(values, upper_percentile), decimals=3))
    return lower_bound, upper_bound

def calculate_vector_confidence_intervals(vector_values):
    """
    Calculate confidence intervals for each component of parameter vectors.
    
    Args:
        vector_values: List of parameter vectors from bootstrap samples
        
    Returns:
        List of (lower, upper) tuples for each component
    """
    ci_vectors = []
    for i in range(len(vector_values[0])):
        component_values = [v[i] for v in vector_values]
        ci_vectors.append(calculate_confidence_interval(component_values))
    return ci_vectors

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    # Suppress warnings by redirecting stderr
    original_stderr = sys.stderr
    devnull = open(os.devnull, 'w')
    sys.stderr = devnull
    
    try:
        # Initial parameter guesses
        # Format: [μ₁, μ₂, μ₃, ω₁, ω₂, ω₃]
        initial_guess = [0, 0, 0, 0, 0, -0.5]
        initial_guess_sigma = [0, 0, -0.5]  # [σ₁, σ₂, σ₃]
        
        print("="*80)
        print("           AUCTION EMPIRICS: SECOND-HIGHEST BID ANALYSIS")
        print("="*80)
        print(f"Dataset: {len(data):,} observations")
        print(f"Explanatory variables: {explanatory_variables}")
        print(f"Starting optimization...")
        
        # Run primary estimation
        results = primary_analysis(data, explanatory_variables, initial_guess, initial_guess_sigma)
        
        mu_integrated = results["mu_1"]
        omega_integrated = results["omega_1"]
        sigma_t_1 = results["sigma_1"]
        mu_neutral = results["mu_2"]
        omega_neutral = results["omega_2"]
        sigma_t_2 = results["sigma_2"]

        print("\n" + "="*80)
        print("                    PARAMETER ESTIMATES")
        print("="*80)
        
        print(f"\nINTEGRATED BUILDERS (Type 1):")
        print(f"   μ parameters: {[f'{x:.4f}' for x in mu_integrated]}")
        print(f"   ω parameters: {[f'{x:.4f}' for x in omega_integrated]}")
        print(f"   σ parameters: {[f'{x:.4f}' for x in sigma_t_1]}")
        
        print(f"\nNEUTRAL BUILDERS (Type 2):")
        print(f"   μ parameters: {[f'{x:.4f}' for x in mu_neutral]}")
        print(f"   ω parameters: {[f'{x:.4f}' for x in omega_neutral]}")
        print(f"   σ parameters: {[f'{x:.4f}' for x in sigma_t_2]}")
        
        # Calculate R-squared for both types
        print("\n" + "="*80)
        print("                         R² RESULTS")
        print("="*80)
        
        r_sq_integrated = calculate_r_squared(
            data, mu_integrated, omega_integrated, sigma_t_1, 
            explanatory_variables, expected_second_highest_draw_values, 
            variance_second_highest_draw_values, 1
        )
        print(f"\nINTEGRATED BUILDERS (Type 1):")
        print(f"   Stage 1 R²: {r_sq_integrated['stage1_r_squared']:.4f}")
        print(f"   Stage 2 R²: {r_sq_integrated['stage2_r_squared']:.4f}")
        print(f"   Overall R²: {r_sq_integrated['overall_r_squared']:.4f}")
        
        r_sq_neutral = calculate_r_squared(
            data, mu_neutral, omega_neutral, sigma_t_2, 
            explanatory_variables, expected_second_highest_draw_values, 
            variance_second_highest_draw_values, 2
        )
        print(f"\nNEUTRAL BUILDERS (Type 2):")
        print(f"   Stage 1 R²: {r_sq_neutral['stage1_r_squared']:.4f}")
        print(f"   Stage 2 R²: {r_sq_neutral['stage2_r_squared']:.4f}")
        print(f"   Overall R²: {r_sq_neutral['overall_r_squared']:.4f}")
        
        print("="*80)
        
        # Example calculation for a random auction
        random_auction_instance = data.sample()
        
        print("\n" + "="*80)
        print("                 RANDOM BLOCK DISTRIBUTION ANALYSIS")
        print("="*80)
        
        block_number = random_auction_instance["block_number"].iloc[0]
        slot = random_auction_instance["slot"].iloc[0]
        
        print(f"\nRandom Block Details:")
        print(f"   Block Number: {int(block_number):,}")
        print(f"   Slot: {int(slot):,}")
        
        sample_data = random_auction_instance[[
            'base_fee_gwei', 'bps_abs_price_change', 'winning_bid_amount',
            '2nd_highest_bid_type_1_log', '2nd_highest_bid_type_2_log'
        ]]
        print(f"\nBlock Characteristics:")
        print(sample_data.to_string(index=False))
        
        # Calculate distribution parameters for this auction
        mu_integrated_block = (mu_integrated[0] * random_auction_instance["bps_abs_price_change"].iloc[0] 
                              + mu_integrated[1] * random_auction_instance["base_fee_gwei"].iloc[0] 
                              + mu_integrated[2])
        omega_integrated_block = np.exp(
            omega_integrated[0] * random_auction_instance["bps_abs_price_change"].iloc[0] 
            + omega_integrated[1] * random_auction_instance["base_fee_gwei"].iloc[0] 
            + omega_integrated[2]
        )
        sigma_integrated_block = np.exp(
            sigma_t_1[0] * random_auction_instance["bps_abs_price_change"].iloc[0] 
            + sigma_t_1[1] * random_auction_instance["base_fee_gwei"].iloc[0] 
            + sigma_t_1[2]
        )

        mu_neutral_block = (mu_neutral[0] * random_auction_instance["bps_abs_price_change"].iloc[0] 
                           + mu_neutral[1] * random_auction_instance["base_fee_gwei"].iloc[0] 
                           + mu_neutral[2])
        omega_neutral_block = np.exp(
            omega_neutral[0] * random_auction_instance["bps_abs_price_change"].iloc[0] 
            + omega_neutral[1] * random_auction_instance["base_fee_gwei"].iloc[0] 
            + omega_neutral[2]
        )
        sigma_neutral_block = np.exp(
            sigma_t_2[0] * random_auction_instance["bps_abs_price_change"].iloc[0] 
            + sigma_t_2[1] * random_auction_instance["base_fee_gwei"].iloc[0] 
            + sigma_t_2[2]
        )

        # Calculate distribution statistics
        print(f"\nINTEGRATED BUILDERS (Type 1) Distribution:")
        print(f"   Log-Normal Parameters:")
        print(f"     μ (location): {mu_integrated_block:.4f}")
        print(f"     σ (scale):    {sigma_integrated_block:.4f}")
        print(f"   Distribution Statistics:")
        mean_integrated = np.exp(mu_integrated_block + 0.5 * sigma_integrated_block**2)
        var_integrated = (np.exp(sigma_integrated_block**2) - 1) * np.exp(2*mu_integrated_block + sigma_integrated_block**2)
        std_integrated = np.sqrt(var_integrated)
        print(f"     Mean (ETH):   {mean_integrated:.6f}")
        print(f"     Std Dev:      {std_integrated:.6f}")
        print(f"     Variance:     {var_integrated:.8f}")
        
        print(f"\nNEUTRAL BUILDERS (Type 2) Distribution:")
        print(f"   Log-Normal Parameters:")
        print(f"     μ (location): {mu_neutral_block:.4f}")
        print(f"     σ (scale):    {sigma_neutral_block:.4f}")
        print(f"   Distribution Statistics:")
        mean_neutral = np.exp(mu_neutral_block + 0.5 * sigma_neutral_block**2)
        var_neutral = (np.exp(sigma_neutral_block**2) - 1) * np.exp(2*mu_neutral_block + sigma_neutral_block**2)
        std_neutral = np.sqrt(var_neutral)
        print(f"     Mean (ETH):   {mean_neutral:.6f}")
        print(f"     Std Dev:      {std_neutral:.6f}")
        print(f"     Variance:     {var_neutral:.8f}")
        
        print(f"\nComparative Analysis:")
        print(f"   Mean Difference: {mean_integrated - mean_neutral:.6f} ETH")
        print(f"   Relative Volatility: Integrated {std_integrated/mean_integrated:.3f} vs Neutral {std_neutral/mean_neutral:.3f}")
        print(f"   Higher Mean: {'Integrated' if mean_integrated > mean_neutral else 'Neutral'}")
        
        print("="*80)
        
        # Bootstrap for confidence intervals
        print("\nRunning bootstrap for confidence intervals...")
        n_iterations = 500  # Number of bootstrap replications
        
        # Use the actual parameter estimates as initial guesses for bootstrap
        initial_guess_mu_omega_1 = list(mu_integrated) + list(omega_integrated)
        initial_guess_mu_omega_2 = list(mu_neutral) + list(omega_neutral)
        initial_guess_sigma_1 = list(sigma_t_1)
        initial_guess_sigma_2 = list(sigma_t_2)
        
        # Prepare arguments for bootstrap iterations
        args_1 = [
            (data, explanatory_variables, initial_guess_mu_omega_1, initial_guess_sigma_1) 
            for _ in range(n_iterations)
        ]
        
        print(f"   Running {n_iterations} bootstrap iterations for Integrated Builders (Type 1)...")
        with multiprocessing.Pool() as pool:
            bootstrap_results_1 = list(tqdm(
                pool.imap(bootstrap_iteration, args_1), 
                total=n_iterations
            ))
        
        args_2 = [
            (data, explanatory_variables, initial_guess_mu_omega_2, initial_guess_sigma_2) 
            for _ in range(n_iterations)
        ]
        
        print(f"   Running {n_iterations} bootstrap iterations for Neutral Builders (Type 2)...")
        with multiprocessing.Pool() as pool:
            bootstrap_results_2 = list(tqdm(
                pool.imap(bootstrap_iteration, args_2), 
                total=n_iterations
            ))
        
        # Extract parameter values and calculate confidence intervals
        mu_1_values = [result['mu_1'] for result in bootstrap_results_1]
        omega_1_values = [result['omega_1'] for result in bootstrap_results_1]
        sigma_1_values = [result['sigma_1'] for result in bootstrap_results_1]
        mu_2_values = [result['mu_2'] for result in bootstrap_results_2]
        omega_2_values = [result['omega_2'] for result in bootstrap_results_2]
        sigma_2_values = [result['sigma_2'] for result in bootstrap_results_2]

        ci_mu_1 = calculate_vector_confidence_intervals(mu_1_values)
        ci_omega_1 = calculate_vector_confidence_intervals(omega_1_values)
        ci_sigma_1 = calculate_vector_confidence_intervals(sigma_1_values)
        ci_mu_2 = calculate_vector_confidence_intervals(mu_2_values)
        ci_omega_2 = calculate_vector_confidence_intervals(omega_2_values)
        ci_sigma_2 = calculate_vector_confidence_intervals(sigma_2_values)

        # Print the confidence intervals
        print("\n" + "="*80)
        print("                   95% CONFIDENCE INTERVALS")
        print("="*80)
        
        print(f"\nINTEGRATED BUILDERS (Type 1):")
        print(f"   μ₁: [{ci_mu_1[0][0]:.3f}, {ci_mu_1[0][1]:.3f}]")
        print(f"   μ₂: [{ci_mu_1[1][0]:.3f}, {ci_mu_1[1][1]:.3f}]")
        print(f"   μ₃: [{ci_mu_1[2][0]:.3f}, {ci_mu_1[2][1]:.3f}]")
        print(f"   ω₁: [{ci_omega_1[0][0]:.3f}, {ci_omega_1[0][1]:.3f}]")
        print(f"   ω₂: [{ci_omega_1[1][0]:.3f}, {ci_omega_1[1][1]:.3f}]")
        print(f"   ω₃: [{ci_omega_1[2][0]:.3f}, {ci_omega_1[2][1]:.3f}]")
        print(f"   σ₁: [{ci_sigma_1[0][0]:.3f}, {ci_sigma_1[0][1]:.3f}]")
        print(f"   σ₂: [{ci_sigma_1[1][0]:.3f}, {ci_sigma_1[1][1]:.3f}]")
        print(f"   σ₃: [{ci_sigma_1[2][0]:.3f}, {ci_sigma_1[2][1]:.3f}]")
        
        print(f"\nNEUTRAL BUILDERS (Type 2):")
        print(f"   μ₁: [{ci_mu_2[0][0]:.3f}, {ci_mu_2[0][1]:.3f}]")
        print(f"   μ₂: [{ci_mu_2[1][0]:.3f}, {ci_mu_2[1][1]:.3f}]")
        print(f"   μ₃: [{ci_mu_2[2][0]:.3f}, {ci_mu_2[2][1]:.3f}]")
        print(f"   ω₁: [{ci_omega_2[0][0]:.3f}, {ci_omega_2[0][1]:.3f}]")
        print(f"   ω₂: [{ci_omega_2[1][0]:.3f}, {ci_omega_2[1][1]:.3f}]")
        print(f"   ω₃: [{ci_omega_2[2][0]:.3f}, {ci_omega_2[2][1]:.3f}]")
        print(f"   σ₁: [{ci_sigma_2[0][0]:.3f}, {ci_sigma_2[0][1]:.3f}]")
        print(f"   σ₂: [{ci_sigma_2[1][0]:.3f}, {ci_sigma_2[1][1]:.3f}]")
        print(f"   σ₃: [{ci_sigma_2[2][0]:.3f}, {ci_sigma_2[2][1]:.3f}]")
        
        print("="*80)
        
        # Verify that primary estimates are within their confidence intervals
        print("\nVERIFICATION: Checking if primary estimates are within confidence intervals...")
        
        def check_within_ci(estimate, ci_lower, ci_upper, param_name):
            """Check if estimate is within confidence interval."""
            within = ci_lower <= estimate <= ci_upper
            status = "PASS" if within else "FAIL"
            print(f"   {status} {param_name}: {estimate:.4f} ∈ [{ci_lower:.3f}, {ci_upper:.3f}]")
            return within
        
        print("\nIntegrated Builders (Type 1):")
        for i in range(3):
            check_within_ci(mu_integrated[i], ci_mu_1[i][0], ci_mu_1[i][1], f"μ_{i+1}")
            check_within_ci(omega_integrated[i], ci_omega_1[i][0], ci_omega_1[i][1], f"ω_{i+1}")
            check_within_ci(sigma_t_1[i], ci_sigma_1[i][0], ci_sigma_1[i][1], f"σ_{i+1}")
        
        print("\nNeutral Builders (Type 2):")
        for i in range(3):
            check_within_ci(mu_neutral[i], ci_mu_2[i][0], ci_mu_2[i][1], f"μ_{i+1}")
            check_within_ci(omega_neutral[i], ci_omega_2[i][0], ci_omega_2[i][1], f"ω_{i+1}")
            check_within_ci(sigma_t_2[i], ci_sigma_2[i][0], ci_sigma_2[i][1], f"σ_{i+1}")
    
        print("\n" + "="*80)
        print("Estimation complete!")
        print("="*80)
    finally:
        # Restore original stderr
        sys.stderr = original_stderr
        devnull.close()
