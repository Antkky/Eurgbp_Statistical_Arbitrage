import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def montecarlo(data: pd.Series, iterations: int = 250, seed: int = None):
    if not isinstance(data, pd.Series):
        raise ValueError("Input data must be a pandas Series.")

    rng = np.random.default_rng(seed)
    for _ in tqdm(range(iterations), desc="Monte Carlo"):
        sampled_data = data.sample(n=len(data), replace=True, random_state=rng.integers(1e9)).reset_index(drop=True)
        yield sampled_data.cumsum()

def run_monte_carlo(
    data: pd.Series,
    iterations: int,
    output_file: str = './tests/latest/Montecarlo.csv',
    output_image: str = './tests/latest/Montecarlo.png',
    seed: int = None
):
    equity_curves = []
    final_equities = []

    with open(output_file, 'w') as f:
        for i, curve in enumerate(montecarlo(data, iterations, seed)):
            if i == 0:
                curve.to_frame().to_csv(f, header=True, index=True)
            else:
                curve.to_frame().to_csv(f, header=False, index=True)

            equity_curves.append(curve)
            final_equities.append(curve.iloc[-1])


    mc_data = pd.concat(equity_curves, axis=1)
    plot_monte_carlo(mc_data, final_equities, min(iterations, 100), output_image)

def plot_monte_carlo(mc_data: pd.DataFrame, final_equities: list, num_curves: int = 100, output_image: str = './tests/latest/Montecarlo.png'):
    if not isinstance(mc_data, pd.DataFrame):
        raise ValueError("mc_data must be a pandas DataFrame.")

    # Ensure num_curves is an integer
    num_curves = int(num_curves)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6), gridspec_kw={'width_ratios': [3, 1]})

    # Safely sample curves
    sampled_columns = mc_data.sample(n=num_curves, axis=1, random_state=42) if mc_data.shape[1] > num_curves else mc_data
    for column in sampled_columns.columns:
        ax1.plot(sampled_columns.index, sampled_columns[column], alpha=0.3, linewidth=0.8)

    ax1.axhline(0, color='red', linestyle='--', linewidth=1)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Returns')
    ax1.set_title('Monte Carlo Simulation: Equity Curves')
    ax1.grid(True, linestyle='--', alpha=0.7)

    ax2.hist(final_equities, bins=30, orientation='horizontal', color='skyblue', edgecolor='black')
    ax2.axvline(0, color='red', linestyle='--', linewidth=1)
    ax2.set_xlabel('Frequency')
    ax2.set_ylabel('Total Returns')
    ax2.set_title('Frequency Distribution of Final Returns')
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    plt.close()
    print(f"Plot saved as {output_image}.")
