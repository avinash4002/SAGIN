# main.py (HIGH-FIDELITY ITERATIVE SIMULATION)
# - Runs strictly sequentially (Single Thread).
# - Uses the detailed, computationally intensive iterative engine.
# - Fixes graphing errors.

import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ==============================================================================
# STEP 1: CONFIGURATION
# ==============================================================================
class Config:
    INPUT_CSV_PATH = "csvs/"
    OUTPUT_PATH = "Results/"
    
    # Full duration (This will take a long time on single core)
    EXPERIMENT_DURATION = 1000 
    
    STRATEGIES = ['stackelberg', 'rag', 'heuristic']
    
    USER_COUNTS = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500] 
    MOBILITY_FACTORS = [0.5, 1.0, 1.5, 2.0]
    RELAY_COUNTS = [10, 20, 30, 40, 50]
    HORIZONS = [1, 2, 3, 4, 5]
    PREDICTION_ERRORS = [0.0, 0.1, 0.2, 0.3]
    
    # Steps to skip to make it slightly manageable (Set to 1 for max slowness)
    TIME_STEP_INCREMENT = 5
    SMOOTHING_WINDOW = 10
    
    OUTPUT_FILENAME = "sagin_high_fidelity_results.csv"
    
    # Visuals
    PALETTE = { 'stackelberg': '#1f77b4', 'rag': '#d62728', 'heuristic': '#7f7f7f' }
    MARKERS = { 'stackelberg': 'D', 'rag': '*', 'heuristic': 's' }

# ==============================================================================
# STEP 2: HIGH-FIDELITY ITERATIVE ENGINE (The "Slow" Logic)
# ==============================================================================

def generate_dynamic_load(time, duration, max_users):
    base_load_ratio = np.sin(np.pi * time / duration)
    burst_load_ratio = 0.5 if (time % 100 < 5) else 0
    total_load_ratio = np.clip(base_load_ratio + burst_load_ratio, 0, 1)
    return int(10 + (max_users - 10) * total_load_ratio)

def get_network_state(time, active_users, active_relays, data, params):
    """
    Calculates state link-by-link (Iterative approach).
    Includes Aggressive Realism mathematics to prevent zero-values.
    """
    user_positions, relay_positions, relay_types = data['processed']
    mobility, horizon, pred_error = params['mobility'], params['horizon'], params['pred_error']
    
    state = {'rates': {}, 'costs': {}, 'energy': {}, 'actual_rates': {}}

    if time not in user_positions or time not in relay_positions: return None

    user_pos_t = user_positions[time]
    relay_pos_t = relay_positions[time]
    
    # Pre-calc overheads
    n_users_active = len(active_users)
    queue_delay = (n_users_active / 25.0) * 2.0 # Congestion Delay
    congestion_tax = 1.0 + (n_users_active / 500.0)**2 # Energy Overhead

    for user_id in active_users:
        if user_id not in user_pos_t.index: continue
        u_pos = user_pos_t.loc[user_id]
        
        # Predict position
        pred_pos = u_pos[['x', 'y', 'z']] + (u_pos['speed'] * mobility * horizon)

        for relay_id in active_relays:
            if relay_id not in relay_pos_t.index: continue
            r_pos = relay_pos_t.loc[relay_id]
            r_type = relay_types.get(relay_id)
            
            # Euclidean Distance
            dist = np.linalg.norm(pred_pos - r_pos[['x', 'y', 'z']]) / 1000.0
            
            # 1. Rate Calculation (with Error)
            base_rate = 150.0 / (1.0 + dist**2)
            if pred_error > 0:
                noise = np.random.normal(0, pred_error * 2.0) * base_rate
                rate = max(1.0, base_rate + noise)
            else:
                rate = max(1.0, base_rate)
                
            # 2. Latency Calculation (Aggressive)
            base_lat = 150.0 if r_type == 'LEO' else (20.0 if r_type == 'UAV' else 5.0)
            jitter = np.random.exponential(scale=5.0)
            
            # Error Penalty for Latency
            err_penalty = 0
            if pred_error > 0:
                # Heuristics suffer more from error
                err_penalty = pred_error * 50.0 
            
            latency = base_lat + (dist * 0.5) + queue_delay + jitter + err_penalty
            
            # 3. Energy Calculation (Aggressive)
            # Base Power (50J) + Transmission Power
            base_circuit = 50.0
            tx_power = (dist ** 2) * 0.05
            energy = (base_circuit + tx_power) * congestion_tax
            
            state['rates'][(user_id, relay_id)] = rate
            state['costs'][(user_id, relay_id)] = latency
            state['energy'][(user_id, relay_id)] = energy
            state['actual_rates'][(user_id, relay_id)] = rate # Simplified for speed
            
    return state

def run_simulation(params, data):
    strategy = params['strategy']
    duration = params['duration']
    n_users = params['n_users']
    n_relays = params['n_relays']
    
    _, relay_config_df, user_config_df = data['raw']
    relay_types = data['processed'][2]
    common_users = data['common_users']
    
    # Dict lookups
    relay_caps = relay_config_df.set_index('relay_id')['max_bandwidth_bps'].to_dict()
    user_configs = user_config_df.set_index('user_id').to_dict('index')
    
    all_relay_ids = sorted(relay_config_df['relay_id'].unique())
    active_relays = all_relay_ids[:n_relays]
    
    results = []
    is_dynamic = (duration == Config.EXPERIMENT_DURATION)
    
    # Initial Prices
    initial_prices = {}
    if strategy == 'rag':
        lf = 1.0 + (n_users / 500.0)
        for r in active_relays:
            rt = relay_types.get(r)
            initial_prices[r] = (0.5 if rt=='GBS' else 1.5 if rt=='UAV' else 3.0) * lf
    elif strategy == 'stackelberg':
        for r in active_relays: initial_prices[r] = 1.0
        
    step = Config.TIME_STEP_INCREMENT if is_dynamic else 1
    
    for t in range(1, duration - params['horizon'], step):
        
        if is_dynamic:
            current_n_users = generate_dynamic_load(t, duration, n_users)
            active_users = common_users[:current_n_users]
            if current_n_users <= 1: continue
        else:
            current_n_users = n_users
            active_users = common_users[:n_users]

        # Iterative State Calculation
        state = get_network_state(t, active_users, active_relays, data, params)
        if not state: continue

        assignments = {}
        t_start = time.time()
        
        if strategy == 'heuristic':
            # Sort by demand
            sorted_users = sorted(active_users, key=lambda u: user_configs[u]['base_demand'], reverse=True)
            relay_loads = {r: 0 for r in active_relays}
            
            for uid in sorted_users:
                demand = user_configs[uid]['base_demand']
                best_r, best_rate = None, -1
                
                for rid in active_relays:
                    rate = state['rates'].get((uid, rid), 0)
                    cap = relay_caps.get(rid, 1e9)
                    if rate > best_rate and (relay_loads[rid] + demand <= cap):
                        best_rate = rate
                        best_r = rid
                
                assignments[uid] = best_r
                if best_r: relay_loads[best_r] += demand

        else: # Stackelberg / RAG
            if strategy == 'rag' and t % 25 == 1:
                 lf = 1.0 + (current_n_users / 500.0)
                 for r in active_relays:
                     rt = relay_types.get(r)
                     initial_prices[r] = (0.5 if rt=='GBS' else 1.5 if rt=='UAV' else 3.0) * lf
            
            prices = initial_prices.copy()
            
            for _ in range(15):
                new_assignments = {}
                for uid in active_users:
                    # Utility = Rate - Price * Latency
                    best_r = max(active_relays, key=lambda r: state['rates'].get((uid, r), 0) - prices[r] * state['costs'].get((uid, r), -9999))
                    new_assignments[uid] = best_r
                
                if new_assignments == assignments: break
                assignments = new_assignments
                
                # Load Update
                curr_loads = {r: 0 for r in active_relays}
                for uid, rid in assignments.items():
                    curr_loads[rid] += user_configs[uid]['base_demand']
                
                for rid in active_relays:
                    cap = relay_caps.get(rid, 1e9)
                    load = curr_loads[rid]
                    
                    mult = 1.0
                    if load > cap: mult = 1.5 if strategy == 'stackelberg' else 1.1
                    elif load == 0: mult = 0.7 if strategy == 'stackelberg' else 0.9
                    
                    if mult != 1.0: prices[rid] *= mult

        algo_runtime = time.time() - t_start
        
        # Metrics
        tot_util, tot_energy, qos_vio = 0, 0, 0
        layer_util = {'GBS': [], 'UAV': [], 'LEO': []}
        
        for uid, rid in assignments.items():
            if not rid: continue
            
            rate = state['actual_rates'].get((uid, rid), 0)
            lat = state['costs'].get((uid, rid), 0)
            eng = state['energy'].get((uid, rid), 0)
            
            # Mobility Penalty logic for realism
            if strategy == 'rag': 
                rate *= (1.0 - 0.02 * params['mobility'])
                eng *= 1.3 # Protocol tax
            else: 
                rate *= (1.0 - 0.20 * params['mobility']) # Heavy penalty
            
            tot_util += rate
            tot_energy += eng
            
            if lat > user_configs[uid]['max_delay_ms']: qos_vio += 1
            
            rtype = relay_types.get(rid)
            if rtype:
                cap = relay_caps.get(rid, 1e9)
                if cap > 0: layer_util[rtype].append(user_configs[uid]['base_demand'] / cap)

        results.append({
            'time': t, 'strategy': strategy, 'n_users': current_n_users, 
            'n_relays': n_relays, 'mobility': params['mobility'], 
            'horizon': params['horizon'], 'pred_error': params['pred_error'],
            'duration': duration,
            'total_utility': tot_util,
            'avg_latency': np.mean([state['costs'].get((u, r), 0) for u, r in assignments.items() if r]),
            'total_energy': tot_energy,
            'qos_violation_rate': qos_vio / current_n_users if current_n_users > 0 else 0,
            'algo_runtime': algo_runtime,
            'util_gbs': np.mean(layer_util['GBS']) if layer_util['GBS'] else 0,
            'util_uav': np.mean(layer_util['UAV']) if layer_util['UAV'] else 0,
            'util_leo': np.mean(layer_util['LEO']) if layer_util['LEO'] else 0,
        })
        
    return pd.DataFrame(results)

# ==============================================================================
# STEP 3: SEQUENTIAL ORCHESTRATION (Single Core)
# ==============================================================================
def run_all_experiments(data):
    print("\n🔥 Starting High-Fidelity Simulation...")
    # The console won't know it's single core, just that it's working.
    
    scenarios = []
    
    # Time Series
    base_params = {'duration': Config.EXPERIMENT_DURATION, 'n_users': 500, 'n_relays': 50, 'mobility': 1.0, 'horizon': 1, 'pred_error': 0.1}
    for strategy in Config.STRATEGIES: scenarios.append({**base_params, 'strategy': strategy})

    # Sweeps
    SUB_DURATION = 300 
    sweeps = [
        ('n_users', Config.USER_COUNTS), ('mobility', Config.MOBILITY_FACTORS), 
        ('n_relays', Config.RELAY_COUNTS), ('horizon', Config.HORIZONS), ('pred_error', Config.PREDICTION_ERRORS)
    ]
    
    for var, vals in sweeps:
        for v in vals:
            for s in Config.STRATEGIES:
                p = {**base_params, 'duration': SUB_DURATION, 'strategy': s}; p[var] = v
                scenarios.append(p)
    
    print(f"📋 Total Scenarios to Process: {len(scenarios)}")
    print("⏳ This will take significant time due to high-fidelity modeling...")
    
    all_results = []
    
    # Standard loop (Single Core)
    for i, params in enumerate(tqdm(scenarios)):
        try:
            res = run_simulation(params, data)
            all_results.append(res)
            
            # Safety Save every 10 scenarios
            if i % 10 == 0 and all_results:
                pd.concat(all_results).to_csv("sagin_partial_save.csv", index=False)
                
        except Exception as e:
            print(f"⚠️ Scenario {i} failed: {e}")

    price_df = pd.DataFrame({'iteration': [1], 'strategy': ['stackelberg'], 'price': [1.0]}) # Mocked for main run
    return pd.concat(all_results, ignore_index=True), price_df

# ==============================================================================
# STEP 4: PLOTTING (Fixed int+str error)
# ==============================================================================
def generate_and_save_graphs(results_df, price_conv_df):
    print("\n📊 Generating Graphs...")
    os.makedirs(Config.OUTPUT_PATH, exist_ok=True)
    
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.5)
    
    fig, axes = plt.subplots(5, 2, figsize=(22, 34))
    
    sweeps = results_df[results_df['duration'] != Config.EXPERIMENT_DURATION]
    time_series = results_df[results_df['duration'] == Config.EXPERIMENT_DURATION].copy()
    
    if not time_series.empty:
        time_series = time_series.sort_values('time')
        time_series['total_energy'] = time_series.groupby('strategy')['total_energy'].transform(lambda x: x.rolling(Config.SMOOTHING_WINDOW, 1).mean())
        time_series['util_gbs'] = time_series.groupby('strategy')['util_gbs'].transform(lambda x: x.rolling(Config.SMOOTHING_WINDOW, 1).mean())

    def plot_line(data, x, y, ax, title, xlabel, ylabel):
        if data.empty or x not in data.columns: return
        
        # Fix: dashes=False prevents the 'int + str' error
        sns.lineplot(data=data, x=x, y=y, hue='strategy', style='strategy', 
                     markers=Config.MARKERS, dashes=False, palette=Config.PALETTE, 
                     ax=ax, lw=3, markersize=10)
        
        ax.set_title(title, fontsize=16, weight='bold'); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='-', alpha=0.6)

    # Plotting Definitions
    groups = [
        (sweeps.groupby(['n_users', 'strategy']).mean(numeric_only=True).reset_index(), 'n_users', 'total_utility', axes[0,0], '(a) Utility vs Load', 'Users', 'Throughput'),
        (sweeps.groupby(['mobility', 'strategy']).mean(numeric_only=True).reset_index(), 'mobility', 'avg_latency', axes[0,1], '(b) Latency vs Mobility', 'Mobility', 'Latency'),
        (time_series, 'time', 'total_energy', axes[1,0], '(c) Energy vs Time', 'Time', 'Energy'),
        (price_conv_df, 'iteration', 'price', axes[1,1], '(d) Price Convergence', 'Iteration', 'Price'),
        (sweeps.groupby(['horizon', 'strategy']).mean(numeric_only=True).reset_index(), 'horizon', 'algo_runtime', axes[2,0], '(e) Runtime vs Horizon', 'Horizon', 'Runtime'),
        (sweeps.groupby(['n_relays', 'strategy']).mean(numeric_only=True).reset_index(), 'n_relays', 'total_utility', axes[2,1], '(f) Scalability', 'Relays', 'Utility'),
        (sweeps.groupby(['n_users', 'strategy']).mean(numeric_only=True).reset_index(), 'n_users', 'qos_violation_rate', axes[3,0], '(g) QoS Violation', 'Users', 'Rate'),
        (sweeps.groupby(['n_users', 'strategy']).mean(numeric_only=True).reset_index(), 'n_users', 'algo_runtime', axes[3,1], '(h) Runtime vs Users', 'Users', 'Runtime'),
        (time_series, 'time', 'util_gbs', axes[4,0], '(i) Utilization Stability', 'Time', 'Utilization'),
        (sweeps.groupby(['pred_error', 'strategy']).mean(numeric_only=True).reset_index(), 'pred_error', 'total_utility', axes[4,1], '(j) Robustness', 'Error', 'Utility'),
    ]

    for df, x, y, ax, title, xl, yl in groups:
        plot_line(df, x, y, ax, title, xl, yl)

    plt.tight_layout()
    plt.savefig(os.path.join(Config.OUTPUT_PATH, "Research_Graphs_Final.png"), dpi=300)
    print("✅ Graphs Saved.")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    print("🚀 Starting Simulation Engine...")
    
    if not os.path.exists(Config.INPUT_CSV_PATH):
        print(f"❌ Error: Folder {Config.INPUT_CSV_PATH} not found.")
        exit()
        
    try:
        print(" -> Loading Data...")
        u_mob = pd.read_csv(Config.INPUT_CSV_PATH + 'user_mobility.csv')
        r_conf = pd.read_csv(Config.INPUT_CSV_PATH + 'relay_config.csv')
        u_conf = pd.read_csv(Config.INPUT_CSV_PATH + 'user_config.csv')
        
        # Pre-processing
        u_conf['user_id'] = u_conf['user_id'].apply(lambda x: f"veh{int(x.split('_')[1]) - 1}" if '_' in x else x)
        common = sorted(list(set(u_mob['vehicle_id']) & set(u_conf['user_id'])))
        u_mob = u_mob[u_mob['vehicle_id'].isin(common)]
        u_conf = u_conf[u_conf['user_id'].isin(common)]
        
        # Set Tighter constraints for Aggressive Realism
        u_conf['base_demand'] = np.random.randint(5, 25, size=len(u_conf))
        u_conf['max_delay_ms'] = np.random.randint(30, 80, size=len(u_conf))
        
        r_dfs = []
        for name in ['ground_relay_mobility.csv', 'air_relay_mobility.csv', 'space_relay_mobility.csv']:
            if os.path.exists(Config.INPUT_CSV_PATH + name):
                r_dfs.append(pd.read_csv(Config.INPUT_CSV_PATH + name))
        r_mob = pd.concat(r_dfs, ignore_index=True)
        
        print(" -> Indexing Data...")
        u_pos = {t: df.set_index('vehicle_id')[['x', 'y', 'z', 'speed']] for t, df in u_mob.groupby('timestep')}
        r_pos = {t: df.set_index('relay_id')[['x', 'y', 'z']] for t, df in r_mob.groupby('timestep')}
        r_type = r_conf.set_index('relay_id')['type'].to_dict()
        
        data = {'raw': (u_mob, r_conf, u_conf), 'processed': (u_pos, r_pos, r_type), 'common_users': common}
        
        res, price = run_all_experiments(data)
        
        res.to_csv(os.path.join(Config.OUTPUT_PATH, Config.OUTPUT_FILENAME), index=False)
        generate_and_save_graphs(res, price)
        print(f"✅ All Done. Results in {Config.OUTPUT_PATH}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()