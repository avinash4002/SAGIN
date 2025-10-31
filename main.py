# main.py (3-Algorithm Comparison Engine - No API, Simplified Graphs)

# ==============================================================================
#  SAGIN 3-ALGORITHM COMPARISON SIMULATOR
# ==============================================================================
#  Compares:
#  1. 'stackelberg' (Baseline Game Theory)
#  2. 'rag' (Simulated RAG Heuristic)
#  3. 'heuristic' (Greedy/ILP-like Optimal Assignment)
#
#  Generates 10 simplified comparative graphs (algorithm vs. algorithm)
#  and a final results CSV.
#  VERSION 5: Simplifies graphs 4 and 9 per user request.
#

# --- Core Libraries ---
import os
import time
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
    EXPERIMENT_DURATION = 100
    STRATEGIES = ['stackelberg', 'rag', 'heuristic']
    
    # --- New Experiment Variables for 10 Graphs ---
    USER_COUNTS = [10, 20, 30, 40, 50]
    MOBILITY_FACTORS = [0.5, 1.0, 1.5, 2.0]
    RELAY_COUNTS = [5, 7, 9]
    HORIZONS = [1, 2, 3, 4, 5]
    PREDICTION_ERRORS = [0.0, 0.1, 0.2, 0.3]

# ==============================================================================
#  STEP 2: SIMULATION ENGINE
# ==============================================================================
def get_network_state(time, active_users, active_relays, data, params):
    """Predicts network state, now with added prediction error."""
    user_positions, relay_positions, relay_types = data['processed']
    mobility, horizon, pred_error = params['mobility'], params['horizon'], params['pred_error']
    
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
            relay_type = relay_types.get(relay_id)
            
            predicted_user_pos = user_pos[['x', 'y', 'z']] + (user_pos['speed'] * mobility * horizon)
            distance = np.linalg.norm(predicted_user_pos - relay_pos) / 1000.0

            base_rate = 150 / (1 + distance**2)
            noise = np.random.normal(0, base_rate * pred_error)
            predicted_rate = max(5, base_rate + noise)
            
            latency = {'GBS': 5, 'UAV': 20, 'LEO': 150}.get(relay_type, 50) + distance * 0.1
            energy = distance * 0.5
            
            state['rates'][(user_id, relay_id)] = predicted_rate
            state['costs'][(user_id, relay_id)] = latency
            state['energy'][(user_id, relay_id)] = energy

            if not user_pos_future.empty and user_id in user_pos_future.index:
                actual_distance = np.linalg.norm(user_pos_future.loc[user_id][['x','y','z']] - relay_pos) / 1000.0
                actual_rate = max(5, 150 / (1 + actual_distance**2))
                state['actual_rates'][(user_id, relay_id)] = actual_rate
            else:
                state['actual_rates'][(user_id, relay_id)] = predicted_rate
                
    return state

def run_simulation(params, data):
    """Runs a full simulation, now with 3 algorithms and detailed logging."""
    strategy, duration, n_users, n_relays = params['strategy'], params['duration'], params['n_users'], params['n_relays']
    user_mobility_df, relay_config_df, user_config_df = data['raw']
    relay_types = data['processed'][2]
    common_users = data['common_users']
    
    results = []
    all_user_ids = common_users
    all_relay_ids = sorted(relay_config_df['relay_id'].unique())
    active_users = all_user_ids[:n_users]
    active_relays = all_relay_ids[:n_relays]
    
    relay_caps = relay_config_df.set_index('relay_id')['max_bandwidth_bps'].to_dict()
    user_configs = user_config_df.set_index('user_id')

    if strategy == 'rag':
        initial_prices = {rid: 0.5 if 'gbs' in rid else (1.5 if 'uav' in rid else 3.0) for rid in active_relays}
    elif strategy == 'stackelberg':
        initial_prices = {relay_id: 1.0 for relay_id in active_relays}
        
    for t in range(1, duration - params['horizon']):
        state = get_network_state(t, active_users, active_relays, data, params)
        if not state or not state['rates']: continue

        assignments = {}
        algo_runtime = 0
        iterations = 1
        
        t_start = time.time()
        if strategy == 'stackelberg' or strategy == 'rag':
            prices = initial_prices.copy()
            for i in range(15):
                user_choices = {user_id: max({r_id: state['rates'].get((user_id, r_id), 0) - p * state['costs'].get((user_id, r_id), -np.inf)
                                              for r_id, p in prices.items()}, key=lambda r: state['rates'].get((user_id, r), 0) - prices[r] * state['costs'].get((user_id, r), -np.inf))
                               for user_id in active_users}
                
                if user_choices == assignments: break
                assignments = user_choices
                
                loads = {r: sum(user_configs.loc[u]['base_demand'] for u, relay in assignments.items() if relay == r) for r in active_relays}
                for r_id in active_relays:
                    if loads.get(r_id, 0) > relay_caps.get(r_id, 1e9): prices[r_id] *= 1.1
                    elif loads.get(r_id, 0) == 0: prices[r_id] *= 0.9
            iterations = i + 1

        elif strategy == 'heuristic':
            sorted_users = sorted(active_users, key=lambda u: user_configs.loc[u]['base_demand'], reverse=True)
            relay_loads = {r: 0 for r in active_relays}
            for user_id in sorted_users:
                best_relay = None
                best_rate = -1
                demand = user_configs.loc[user_id]['base_demand']
                for relay_id in active_relays:
                    rate = state['rates'].get((user_id, relay_id), 0)
                    if rate > best_rate and (relay_loads[relay_id] + demand) <= relay_caps.get(relay_id, 1e9):
                        best_rate = rate
                        best_relay = relay_id
                
                if best_relay:
                    assignments[user_id] = best_relay
                    relay_loads[best_relay] += demand
        
        algo_runtime = time.time() - t_start
        
        total_utility, total_energy, qos_violations = 0, 0, 0
        layer_util = {'GBS': [], 'UAV': [], 'LEO': []}
        achieved_rates, actual_rates = [], []
        
        for user_id, relay_id in assignments.items():
            if not relay_id: continue
            
            pred_rate = state['rates'].get((user_id, relay_id), 0)
            act_rate = state['actual_rates'].get((user_id, relay_id), 0)
            latency = state['costs'].get((user_id, relay_id), 0)
            energy = state['energy'].get((user_id, relay_id), 0)
            
            achieved_rates.append(pred_rate)
            actual_rates.append(act_rate)
            total_utility += act_rate
            total_energy += energy
            
            if latency > user_configs.loc[user_id]['max_delay_ms']:
                qos_violations += 1
            
            layer_type = relay_types.get(relay_id)
            if layer_type:
                demand = user_configs.loc[user_id]['base_demand']
                capacity = relay_caps.get(relay_id, 1e9)
                if capacity > 0:
                    layer_util[layer_type].append(demand / capacity)

        results.append({
            'time': t, 'strategy': strategy, 'n_users': n_users, 'n_relays': n_relays,
            'mobility': params['mobility'], 'horizon': params['horizon'], 'pred_error': params['pred_error'],
            'duration': duration,
            'total_utility': total_utility,
            'avg_latency': np.mean([state['costs'].get((u, r), 0) for u, r in assignments.items() if r]),
            'total_energy': total_energy,
            'qos_violation_rate': qos_violations / n_users if n_users > 0 else 0,
            'gai_runtime': 0,
            'algo_runtime': algo_runtime,
            'iterations': iterations,
            'util_gbs': np.mean(layer_util['GBS']) if layer_util['GBS'] else 0,
            'util_uav': np.mean(layer_util['UAV']) if layer_util['UAV'] else 0,
            'util_leo': np.mean(layer_util['LEO']) if layer_util['LEO'] else 0,
            'pred_accuracy': np.mean([abs(p - a) / a for p, a in zip(achieved_rates, actual_rates) if a > 0])
        })
        
    return pd.DataFrame(results)

def run_price_convergence_experiment(data):
    """A special experiment just for the price convergence graph."""
    print("📈 Running Price Convergence sub-experiment...")
    
    common_users = data['common_users']
    all_relay_ids = sorted(data['raw'][1]['relay_id'].unique())
    
    params = {'strategy': 'stackelberg', 'n_users': 50, 'n_relays': 9, 'duration': 2, 'mobility': 1.0, 'horizon': 1, 'pred_error': 0.1}
    # Ensure we use valid users/relays
    active_users = common_users[:min(50, len(common_users))]
    active_relays = all_relay_ids[:min(9, len(all_relay_ids))]
    
    state = get_network_state(1, active_users, active_relays, data, params)
    
    relay_ids = active_relays
    user_ids = active_users
    relay_caps = data['raw'][1].set_index('relay_id')['max_bandwidth_bps'].to_dict()
    user_configs = data['raw'][2].set_index('user_id')

    strategies = {
        'stackelberg': {r: 1.0 for r in relay_ids},
        'rag': {rid: 0.5 if 'gbs' in rid else (1.5 if 'uav' in rid else 3.0) for rid in relay_ids}
    }
    
    price_logs = []
    
    for strategy, initial_prices in strategies.items():
        prices = initial_prices.copy()
        assignments = {}
        for i in range(25):
            user_choices = {user_id: max({r_id: state['rates'].get((user_id, r_id), 0) - p * state['costs'].get((user_id, r_id), -np.inf)
                                          for r_id, p in prices.items()}, key=lambda r: state['rates'].get((user_id, r), 0) - prices[r] * state['costs'].get((user_id, r), -np.inf))
                           for user_id in user_ids}
            
            assignments = user_choices
            loads = {r: sum(user_configs.loc[u]['base_demand'] for u, relay in assignments.items() if relay == r) for r in relay_ids}
            
            # --- MODIFICATION ---
            # Log all prices and average them later
            for r_id, price in prices.items():
                price_logs.append({'iteration': i, 'strategy': strategy, 'relay': r_id, 'price': price})

            for r_id in relay_ids:
                if loads.get(r_id, 0) > relay_caps.get(r_id, 1e9): prices[r_id] *= 1.1
                elif loads.get(r_id, 0) == 0: prices[r_id] *= 0.9
                
    return pd.DataFrame(price_logs)

# ==============================================================================
#  STEP 3: EXPERIMENT ORCHESTRATION
# ==============================================================================
def run_all_experiments(data):
    """Defines and runs all new experimental scenarios."""
    print("\n🔥 Starting all experiments...")
    scenarios = []
    
    # Use a smaller N for base duration to speed up
    base_params = {'duration': Config.EXPERIMENT_DURATION, 'n_users': 40, 'n_relays': 9, 'mobility': 1.0, 'horizon': 1, 'pred_error': 0.1}
    for strategy in Config.STRATEGIES:
        scenarios.append({**base_params, 'strategy': strategy})

    # Use a smaller N for all sub-experiments to speed up
    for n_users in Config.USER_COUNTS:
        for strategy in Config.STRATEGIES:
            scenarios.append({'strategy': strategy, 'duration': 50, 'n_users': n_users, 'n_relays': 9, 'mobility': 1.0, 'horizon': 1, 'pred_error': 0.1})
            
    for mobility in Config.MOBILITY_FACTORS:
        for strategy in Config.STRATEGIES:
            scenarios.append({'strategy': strategy, 'duration': 50, 'n_users': 40, 'n_relays': 9, 'mobility': mobility, 'horizon': 1, 'pred_error': 0.1})

    for n_relays in Config.RELAY_COUNTS:
        for strategy in Config.STRATEGIES:
            scenarios.append({'strategy': strategy, 'duration': 50, 'n_users': 40, 'n_relays': n_relays, 'mobility': 1.0, 'horizon': 1, 'pred_error': 0.1})

    for horizon in Config.HORIZONS:
        for strategy in Config.STRATEGIES:
            scenarios.append({'strategy': strategy, 'duration': 50, 'n_users': 40, 'n_relays': 9, 'mobility': 1.0, 'horizon': horizon, 'pred_error': 0.1})

    for pred_error in Config.PREDICTION_ERRORS:
        for strategy in Config.STRATEGIES:
            scenarios.append({'strategy': strategy, 'duration': 50, 'n_users': 40, 'n_relays': 9, 'mobility': 1.0, 'horizon': 1, 'pred_error': pred_error})

    all_results = [run_simulation(params, data) for params in tqdm(scenarios, desc="Overall Progress")]
    
    price_conv_df = run_price_convergence_experiment(data)
    
    results_df = pd.concat(all_results, ignore_index=True)
    print("🏁 All experiments complete!")
    return results_df, price_conv_df

# ==============================================================================
#  STEP 4: VISUALIZATION (UPDATED FOR SIMPLICITY)
# ==============================================================================
def generate_and_save_graphs(results_df, price_conv_df):
    """Takes the final DataFrames and generates all 10 simplified plots."""
    print("\n📊 Generating and saving final graphs...")
    os.makedirs(Config.OUTPUT_PATH, exist_ok=True)
    
    fig, axes = plt.subplots(5, 2, figsize=(20, 30))
    fig.suptitle('SAGIN 3-Algorithm Performance Comparison', fontsize=22, y=1.02)
    
    # --- Aggregate Data ---
    user_agg = results_df.groupby(['n_users', 'strategy']).mean().reset_index()
    mobility_agg = results_df.groupby(['mobility', 'strategy']).mean().reset_index()
    relay_agg = results_df.groupby(['n_relays', 'strategy']).mean().reset_index()
    horizon_agg = results_df.groupby(['horizon', 'strategy']).mean().reset_index()
    error_agg = results_df.groupby(['pred_error', 'strategy']).mean().reset_index()
    
    # This filter is now safe because the 'duration' column exists
    time_agg = results_df[results_df['duration'] == Config.EXPERIMENT_DURATION].copy()
    
    # --- NEW: Aggregate for simplified graphs ---
    price_agg = price_conv_df.groupby(['iteration', 'strategy']).mean(numeric_only=True).reset_index()
    time_agg['avg_utilization'] = time_agg[['util_gbs', 'util_uav', 'util_leo']].mean(axis=1)

    
    try:
        with pd.ExcelWriter(os.path.join(Config.OUTPUT_PATH, "all_graph_data.xlsx")) as writer:
            user_agg.to_excel(writer, sheet_name='vs_Users', index=False)
            mobility_agg.to_excel(writer, sheet_name='vs_Mobility', index=False)
            relay_agg.to_excel(writer, sheet_name='vs_Relays', index=False)
            horizon_agg.to_excel(writer, sheet_name='vs_Horizon', index=False)
            error_agg.to_excel(writer, sheet_name='vs_PredError', index=False)
            time_agg.to_excel(writer, sheet_name='vs_Time', index=False)
            price_agg.to_excel(writer, sheet_name='Price_Convergence_Avg', index=False) # Save new agg
        print(f"✅ Aggregated data for graphs saved to {os.path.join(Config.OUTPUT_PATH, 'all_graph_data.xlsx')}")
    except Exception as e:
        print(f"Could not save Excel file. Error: {e}")

    # --- Plotting all 10 graphs ---
    sns.lineplot(data=user_agg, x='n_users', y='total_utility', hue='strategy', marker='o', ax=axes[0, 0]).set(title='1. System Utility vs. Number of Users', xlabel='Number of Users (Traffic Load)')
    sns.lineplot(data=mobility_agg, x='mobility', y='avg_latency', hue='strategy', marker='o', ax=axes[0, 1]).set(title='2. Average Latency vs. User Mobility Speed', xlabel='Mobility Factor')
    sns.lineplot(data=time_agg, x='time', y='total_energy', hue='strategy', ax=axes[1, 0]).set(title='3. Energy Utilization vs. Time', xlabel='Time (seconds)')
    
    # --- MODIFIED GRAPH 4 ---
    sns.lineplot(data=price_agg, x='iteration', y='price', hue='strategy', marker='o', ax=axes[1, 1]).set(title='4. Avg. Price Convergence of Relays', xlabel='Iteration', ylabel='Average Price')
    
    sns.lineplot(data=horizon_agg, x='horizon', y='pred_accuracy', hue='strategy', marker='o', ax=axes[2, 0]).set(title='5. Prediction Accuracy vs. DT Horizon Length', xlabel='Horizon (seconds)', ylabel='Mean Absolute % Error')
    sns.lineplot(data=relay_agg, x='n_relays', y='total_utility', hue='strategy', marker='o', ax=axes[2, 1]).set(title='6. System Utility vs. Number of Relay Nodes', xlabel='Number of Relays')
    sns.lineplot(data=user_agg, x='n_users', y='qos_violation_rate', hue='strategy', marker='o', ax=axes[3, 0]).set(title='7. QoS Violation Rate vs. Traffic Load', xlabel='Number of Users (Traffic Load)')
    sns.lineplot(data=user_agg, x='n_users', y='algo_runtime', hue='strategy', marker='o', ax=axes[3, 1]).set(title='8. Optimization Runtime vs. Number of Users', xlabel='Number of Users', ylabel='Runtime (seconds)')
    
    # --- MODIFIED GRAPH 9 ---
    sns.lineplot(data=time_agg, x='time', y='avg_utilization', hue='strategy', ax=axes[4, 0]).set(title='9. Average Resource Utilization vs. Time', xlabel='Time (seconds)', ylabel='Average Utilization')
    
    sns.lineplot(data=error_agg, x='pred_error', y='total_utility', hue='strategy', marker='o', ax=axes[4, 1]).set(title='10. System Performance vs. Prediction Error', xlabel='Prediction Error Factor')

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig_path = os.path.join(Config.OUTPUT_PATH, "all_10_graphs.png")
    fig.savefig(fig_path, dpi=300)
    print(f"✅ All 10 graphs saved to {fig_path}")
    plt.show()

# ==============================================================================
#  MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == "__main__":
    # --- Initial Setup ---
    sns.set_theme(style="darkgrid")
    os.makedirs(Config.OUTPUT_PATH, exist_ok=True)
    
    # --- Load Data ---
    print("🚀 Starting SAGIN Simulator (3-Algorithm)...")
    print("  -> Loading and pre-processing data...")
    try:
        user_mobility_df = pd.read_csv(os.path.join(Config.INPUT_CSV_PATH, 'user_mobility.csv'))
        relay_config_df = pd.read_csv(os.path.join(Config.INPUT_CSV_PATH, 'relay_config.csv'))
        user_config_df = pd.read_csv(os.path.join(Config.INPUT_CSV_PATH, 'user_config.csv'))

        print("  -> Standardizing user IDs to match 'veh' format (e.g., user_1 -> veh0)...")
        try:
            user_config_df['user_id'] = user_config_df['user_id'].apply(
                lambda x: f"veh{int(x.split('_')[1]) - 1}"
            )
            print("  -> Standardization complete.")
        except Exception as e:
            print(f"  -> WARNING: Could not auto-standardize user_id. Assuming formats already match. Error: {e}")

        mob_users = set(user_mobility_df['vehicle_id'].unique())
        cfg_users = set(user_config_df['user_id'].unique())
        common_users = sorted(list(mob_users.intersection(cfg_users)))

        if not common_users:
            print(f"❌ ERROR: No common users found between user_mobility.csv (IDs: {list(mob_users)[:3]}...) "
                  f"and user_config.csv (IDs: {list(cfg_users)[:3]}...). Check standardization logic.")
            exit()
        
        print(f"  -> Found {len(common_users)} common users. Proceeding with this set.")
        
        user_mobility_df = user_mobility_df[user_mobility_df['vehicle_id'].isin(common_users)].copy()
        user_config_df = user_config_df[user_config_df['user_id'].isin(common_users)].copy()

        relay_config_df['max_bandwidth_bps'] = relay_config_df['max_bandwidth_bps'] / 1e6
        user_config_df['base_demand'] = np.random.randint(2, 10, size=len(user_config_df))
        
        relay_mobility_df = pd.concat([
            pd.read_csv(os.path.join(Config.INPUT_CSV_PATH, 'ground_relay_mobility.csv')),
            pd.read_csv(os.path.join(Config.INPUT_CSV_PATH, 'air_relay_mobility.csv')),
            pd.read_csv(os.path.join(Config.INPUT_CSV_PATH, 'space_relay_mobility.csv'))
        ], ignore_index=True)
        
        user_positions = {t: df.set_index('vehicle_id')[['x', 'y', 'z', 'speed']] for t, df in user_mobility_df.groupby('timestep')}
        relay_positions = {t: df.set_index('relay_id')[['x', 'y', 'z']] for t, df in relay_mobility_df.groupby('timestep')}
        relay_types = relay_config_df.set_index('relay_id')['type'].to_dict()

        simulation_data = {
            'raw': (user_mobility_df, relay_config_df, user_config_df),
            'processed': (user_positions, relay_positions, relay_types),
            'common_users': common_users
        }
        print("  -> Data loaded successfully.")
    except FileNotFoundError as e:
        print(f"❌ ERROR: Could not find a required data file. Please check your INPUT_CSV_PATH.")
        print(f"  -> Details: {e}")
        exit()
    except KeyError as e:
        print(f"❌ ERROR: A required column is missing from your CSVs. Details: {e}")
        exit()
    except Exception as e:
        print(f"❌ An unexpected error occurred during data loading: {e}")
        exit()

    # --- Run Experiments and Generate Outputs ---
    results, price_df = run_all_experiments(simulation_data)
    generate_and_save_graphs(results, price_df)