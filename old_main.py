# main.py (Simulated RAG Version - No LLM)

# ==============================================================================
#  SAGIN SIMULATOR (BASELINE VS. SIMULATED RAG)
# ==============================================================================
#  This self-contained script runs a complete simulation comparing a baseline
#  Stackelberg game with a simulated RAG approach that uses expert heuristics
#  instead of a live LLM.
#

# --- Core Libraries ---
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ==============================================================================
#  STEP 1: CONFIGURATION
# ==============================================================================
class Config:
    """
    Configuration class to hold all paths and parameters.
    ‼️ UPDATE THESE PATHS TO MATCH YOUR FOLDER STRUCTURE ‼️
    """
    # --- File Paths (use a trailing slash /) ---
    INPUT_CSV_PATH = "csvs/"
    OUTPUT_PATH = "Results/"

    # --- Experiment Parameters ---
    EXPERIMENT_DURATION = 350
    STRATEGIES = ['stackelberg', 'rag']
    USER_COUNTS = [20, 30, 40, 50]
    RELAY_COUNTS = [5, 7, 9]
    MOBILITY_FACTORS = [0.5, 1.0, 1.5]
    HORIZONS = [1, 2, 3, 4, 5]

# ==============================================================================
#  STEP 2: SIMULATION ENGINE
# ==============================================================================
def get_network_state(time, active_users, active_relays, data, mobility_factor=1.0, horizon=0):
    """Predicts network state for a future time (time + horizon)."""
    user_positions, relay_positions, relay_types = data
    future_time = time + horizon
    state = {'rates': {}, 'costs': {}, 'energy': {}, 'actual_rates': {}}

    user_pos_t = user_positions.get(time, pd.DataFrame())
    relay_pos_t = relay_positions.get(time, pd.DataFrame())
    user_pos_future = user_positions.get(future_time, pd.DataFrame())

    if user_pos_t.empty or relay_pos_t.empty: return None

    for user_id in active_users:
        if user_id not in user_pos_t.index: continue
        user_pos = user_pos_t.loc[user_id]

        for relay_id in active_relays:
            if relay_id not in relay_pos_t.index: continue
            relay_pos = relay_pos_t.loc[relay_id]
            
            predicted_user_pos = user_pos[['x', 'y', 'z']] + (user_pos['speed'] * mobility_factor * horizon)
            distance = np.linalg.norm(predicted_user_pos - relay_pos) / 1000.0

            rate = 150 / (1 + distance**2) + np.random.normal(0, 3)
            state['rates'][(user_id, relay_id)] = max(5, rate)

            latency = {'GBS': 5, 'UAV': 20, 'LEO': 150}.get(relay_types[relay_id], 50)
            energy = distance * 0.5
            state['costs'][(user_id, relay_id)] = latency + energy
            state['energy'][(user_id, relay_id)] = energy

            if not user_pos_future.empty and user_id in user_pos_future.index:
                actual_distance = np.linalg.norm(user_pos_future.loc[user_id][['x','y','z']] - relay_pos) / 1000.0
                actual_rate = 150 / (1 + actual_distance**2)
                state['actual_rates'][(user_id, relay_id)] = max(5, actual_rate)
                
    return state

def run_simulation(params, data):
    """Runs a full simulation based on a dictionary of parameters."""
    strategy, duration, n_users, n_relays, mobility, horizon = params.values()
    user_mobility_df, relay_mobility_df, relay_config_df = data['raw']
    
    results = []
    all_user_ids = sorted(user_mobility_df['vehicle_id'].unique())
    all_relay_ids = sorted(relay_mobility_df['relay_id'].unique())
    active_users = all_user_ids[:n_users]
    active_relays = all_relay_ids[:n_relays]

    for t in range(1, duration - horizon):
        state = get_network_state(t, active_users, active_relays, data['processed'], mobility, horizon)
        if not state or not state['rates']: continue

        if strategy == 'stackelberg':
            prices = {relay_id: 1.0 for relay_id in active_relays}
        else: # Simulated RAG: Smarter initial prices based on expert rules
            prices = {rid: 0.5 if 'gbs' in rid else (1.5 if 'uav' in rid else 3.0) for rid in active_relays}

        assignments = {}
        for i in range(15):
            user_choices = {user_id: max({r_id: state['rates'].get((user_id, r_id), 0) - p * state['costs'].get((user_id, r_id), -np.inf)
                                          for r_id, p in prices.items()}, key=lambda r: state['rates'].get((user_id, r), 0) - prices[r] * state['costs'].get((user_id, r), -np.inf))
                           for user_id in active_users}
            if user_choices == assignments: break
            assignments = user_choices
            
            loads = {r: list(assignments.values()).count(r) for r in active_relays}
            for r_id in active_relays:
                if loads[r_id] > n_users / n_relays: prices[r_id] *= 1.1
                elif loads[r_id] == 0: prices[r_id] *= 0.9

        achieved_rates = [state['rates'].get((u, r), 0) for u, r in assignments.items()]
        actual_rates = [state['actual_rates'].get((u, r), 0) for u, r in assignments.items()]
        
        results.append({
            'strategy': strategy, 'num_users': n_users, 'num_relays': n_relays,
            'mobility': mobility, 'time_horizon': horizon, 'iterations': i + 1,
            'total_utility': sum(achieved_rates),
            'avg_delay': np.mean([state['costs'].get((u, r), 0) for u, r in assignments.items()]),
            'total_energy': sum(state['energy'].get((u, r), 0) for u, r in assignments.items()),
            'fairness': (np.sum(achieved_rates)**2) / (len(achieved_rates) * np.sum(np.square(achieved_rates))) if sum(achieved_rates) > 0 else 0,
            'prediction_error': np.mean([abs(p - a) for p, a in zip(achieved_rates, actual_rates) if a > 0])
        })
        
    return pd.DataFrame(results)

# ==============================================================================
#  STEP 3: EXPERIMENT ORCHESTRATION & VISUALIZATION
# ==============================================================================
def run_all_experiments(data):
    """Defines and runs all experimental scenarios."""
    print("\n🔥 Starting all experiments...")
    scenarios = []
    for strategy in Config.STRATEGIES:
        for n_users in Config.USER_COUNTS: scenarios.append({'strategy': strategy, 'duration': Config.EXPERIMENT_DURATION, 'n_users': n_users, 'n_relays': 9, 'mobility': 1.0, 'horizon': 1})
        for n_relays in Config.RELAY_COUNTS: scenarios.append({'strategy': strategy, 'duration': Config.EXPERIMENT_DURATION, 'n_users': 50, 'n_relays': n_relays, 'mobility': 1.0, 'horizon': 1})
        for mobility in Config.MOBILITY_FACTORS: scenarios.append({'strategy': strategy, 'duration': Config.EXPERIMENT_DURATION, 'n_users': 50, 'n_relays': 9, 'mobility': mobility, 'horizon': 1})
        for horizon in Config.HORIZONS: scenarios.append({'strategy': strategy, 'duration': Config.EXPERIMENT_DURATION, 'n_users': 50, 'n_relays': 9, 'mobility': 1.0, 'horizon': horizon})
    
    all_results = [run_simulation(params, data) for params in tqdm(scenarios, desc="Overall Progress")]
    results_df = pd.concat(all_results, ignore_index=True)
    print("🏁 All experiments complete!")
    return results_df

def generate_and_save_graphs(results_df):
    """Takes the final DataFrame and generates all plots, saving them to files."""
    print("\n📊 Generating and saving final comparison graphs...")
    fig, axes = plt.subplots(3, 2, figsize=(20, 24))
    fig.suptitle('Performance Comparison: Baseline Stackelberg vs. Simulated RAG', fontsize=22, y=1.02)

    user_agg = results_df.groupby(['num_users', 'strategy']).mean().reset_index()
    mobility_agg = results_df.groupby(['mobility', 'strategy']).mean().reset_index()
    relay_agg = results_df.groupby(['num_relays', 'strategy']).mean().reset_index()
    horizon_agg = results_df.groupby(['time_horizon', 'strategy']).mean().reset_index()
    strategy_agg = results_df.groupby(['strategy']).mean().reset_index()

    sns.lineplot(data=user_agg, x='num_users', y='total_utility', hue='strategy', marker='o', ax=axes[0, 0]).set(title='Utility vs. User Count', xlabel='Number of Users', ylabel='Average Total Utility (Mbps)')
    sns.lineplot(data=mobility_agg, x='mobility', y='avg_delay', hue='strategy', marker='o', ax=axes[0, 1]).set(title='Delay vs. Mobility', xlabel='Mobility Factor', ylabel='Average Delay/Cost')
    sns.lineplot(data=user_agg, x='num_users', y='total_energy', hue='strategy', marker='o', ax=axes[1, 0]).set(title='Energy vs. User Count', xlabel='Number of Users', ylabel='Average Total Energy')
    sns.barplot(data=strategy_agg, x='strategy', y='fairness', hue='strategy', ax=axes[1, 1]).set(title='Overall Fairness by Strategy', xlabel='Strategy', ylabel='Average Fairness (Jain\'s Index)')
    sns.lineplot(data=relay_agg, x='num_relays', y='iterations', hue='strategy', marker='o', ax=axes[2, 0]).set(title='Convergence vs. Relay Count', xlabel='Number of Relays', ylabel='Avg. Iterations to Converge')
    sns.lineplot(data=horizon_agg, x='time_horizon', y='prediction_error', hue='strategy', marker='o', ax=axes[2, 1]).set(title='Prediction Error vs. Time Horizon', xlabel='Prediction Horizon (seconds)', ylabel='Mean Absolute Error (Mbps)')
    
    axes[0, 1].set_xticks(Config.MOBILITY_FACTORS)
    axes[1, 1].set_ylim(bottom=0.5)
    for ax in axes.flat: ax.legend(title='Strategy')
    
    plt.tight_layout()
    fig.savefig(os.path.join(Config.OUTPUT_PATH, "all_graphs_summary.png"), dpi=300)
    plt.show()
    print(f"✅ All graphs generated and saved to the {Config.OUTPUT_PATH} folder.")

# ==============================================================================
#  MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == "__main__":
    # --- Initial Setup ---
    sns.set_theme(style="darkgrid")
    os.makedirs(Config.OUTPUT_PATH, exist_ok=True)
    
    # --- Load Data ---
    print("🚀 Starting SAGIN Simulator...")
    print("  -> Loading and pre-processing data...")
    try:
        user_mobility_df = pd.read_csv(os.path.join(Config.INPUT_CSV_PATH, 'user_mobility.csv'))
        relay_config_df = pd.read_csv(os.path.join(Config.INPUT_CSV_PATH, 'relay_config.csv'))
        relay_mobility_df = pd.concat([
            pd.read_csv(os.path.join(Config.INPUT_CSV_PATH, 'ground_relay_mobility.csv')),
            pd.read_csv(os.path.join(Config.INPUT_CSV_PATH, 'air_relay_mobility.csv')),
            pd.read_csv(os.path.join(Config.INPUT_CSV_PATH, 'space_relay_mobility.csv'))
        ], ignore_index=True)
        
        # Pre-process data for faster lookups during simulation
        user_positions = {t: df.set_index('vehicle_id')[['x', 'y', 'z', 'speed']] for t, df in user_mobility_df.groupby('timestep')}
        relay_positions = {t: df.set_index('relay_id')[['x', 'y', 'z']] for t, df in relay_mobility_df.groupby('timestep')}
        relay_types = relay_config_df.set_index('relay_id')['type'].to_dict()

        # Package data for easy passing
        simulation_data = {
            'raw': (user_mobility_df, relay_mobility_df, relay_config_df),
            'processed': (user_positions, relay_positions, relay_types)
        }
        print("  -> Data loaded successfully.")
    except FileNotFoundError as e:
        print(f"❌ ERROR: Could not find a required data file. Please check your INPUT_CSV_PATH.")
        print(f"  -> Details: {e}")
        exit()

    # --- Run Experiments and Generate Outputs ---
    results = run_all_experiments(simulation_data)
    results.to_csv(os.path.join(Config.OUTPUT_PATH, "final_simulation_results.csv"), index=False)
    print(f"\n✅ Final numerical results saved to {Config.OUTPUT_PATH}final_simulation_results.csv")
    generate_and_save_graphs(results)