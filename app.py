import streamlit as st
import numpy as np
import random
import math
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import linprog
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

st.set_page_config(page_title="Smart Grid Optimizer", layout="wide")

# ==========================================
# Core Classes (Env, CP, DQN)
# ==========================================
class SmartGridEnv:
    COST_SOLAR, COST_WIND, COST_GAS, COST_COAL = 0, 0, 5, 10
    EMIS_SOLAR, EMIS_WIND, EMIS_GAS, EMIS_COAL = 0, 0, 2, 5

    def __init__(self, renewable_penetration=1.0, demand_variability=0.2, emission_cap=500, base_demand=700, episode_length=24):
        self.state_dim = 3
        self.action_dim = 4
        self.renewable_penetration = renewable_penetration
        self.demand_variability = demand_variability
        self.emission_cap = emission_cap
        self.base_demand = base_demand
        self.episode_length = episode_length

    def reset(self):
        self.time_step = 0
        self.state = self._sample_state()
        return self.state.copy()

    def _sample_state(self):
        demand = self.base_demand * (1 + random.uniform(-self.demand_variability, self.demand_variability))
        solar = random.uniform(0, 300 * self.renewable_penetration)
        wind  = random.uniform(0, 200 * self.renewable_penetration)
        return np.array([demand, solar, wind], dtype=np.float32)

    def step(self, action):
        demand, solar_avail, wind_avail = self.state
        s_use = min(max(action[0], 0), solar_avail)
        w_use = min(max(action[1], 0), wind_avail)
        g_use = max(action[2], 0)
        c_use = max(action[3], 0)
        total_supply = s_use + w_use + g_use + c_use
        cost = g_use * self.COST_GAS + c_use * self.COST_COAL
        emissions = g_use * self.EMIS_GAS + c_use * self.EMIS_COAL
        
        w_cost, w_emission, w_stability = 1.0, 5.0, 10.0
        stability_penalty = w_stability * abs(total_supply - demand)
        emission_penalty  = w_emission * max(0, emissions - self.emission_cap)
        cost_penalty      = w_cost * cost
        reward = -(stability_penalty + emission_penalty + cost_penalty)
        
        self.time_step += 1
        done = self.time_step >= self.episode_length
        self.state = self._sample_state()
        return self.state.copy(), reward, done, {}

def cp_dispatch(demand, solar_avail, wind_avail, emission_cap):
    c = [0, 0, 5, 10]
    A_eq = [[1, 1, 1, 1]]
    b_eq = [demand]
    A_ub = [[0, 0, 2, 5]]
    b_ub = [emission_cap]
    bounds = [(0, solar_avail), (0, wind_avail), (0, None), (0, None)]
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    if result.success:
        return list(result.x)
    remaining = max(0, demand - solar_avail - wind_avail)
    return [solar_avail, wind_avail, remaining * 0.7, remaining * 0.3]

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim), nn.Softplus()
        )
    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = 0.99
        self.batch_size = 64
        self.eps_start = 1.0
        self.eps_end = 0.01
        self.eps_decay = 100
        self.steps_done = 0
        
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.criterion = nn.SmoothL1Loss()
        self.replay_buffer = deque(maxlen=5000)

    def get_epsilon(self):
        return self.eps_end + (self.eps_start - self.eps_end) * math.exp(-self.steps_done / self.eps_decay)

    def act(self, state, training=True):
        epsilon = self.get_epsilon() if training else 0.0
        self.steps_done += 1
        if random.random() < epsilon:
            return np.array([random.uniform(0, 300), random.uniform(0, 200), random.uniform(0, 400), random.uniform(0, 400)], dtype=np.float32)
        else:
            with torch.no_grad():
                st = torch.FloatTensor(state).unsqueeze(0)
                action = self.policy_net(st).squeeze(0).numpy()
            return action * (state[0] / (action.sum() + 1e-8))

    def store(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size: return
        batch = random.sample(list(self.replay_buffer), self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states_t = torch.FloatTensor(np.array(states))
        rewards_t = torch.FloatTensor(rewards)
        next_t = torch.FloatTensor(np.array(next_states))
        dones_t = torch.FloatTensor(dones)
        
        current_q = self.policy_net(states_t)
        with torch.no_grad():
            next_q = self.target_net(next_t)
            target_vals = rewards_t + (1 - dones_t) * self.gamma * next_q.mean(dim=1)
        
        predicted_vals = current_q.mean(dim=1)
        loss = self.criterion(predicted_vals, target_vals)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# ==========================================
# Streamlit Training Cache
# ==========================================
@st.cache_resource(show_spinner=False)
def get_trained_agent():
    env = SmartGridEnv()
    agent = DQNAgent(3, 4)
    for ep in range(150):
        state = env.reset()
        for t in range(24):
            action = agent.act(state, training=True)
            next_state, reward, done, _ = env.step(action)
            agent.store(state, action, reward, next_state, float(done))
            agent.train_step()
            state = next_state
        if (ep+1) % 10 == 0: agent.update_target()
    return agent

agent = get_trained_agent()

# ==========================================
# Streamlit UI Setup
# ==========================================
st.title("⚡ Smart Energy Grid Optimizer (Live Simulator)")
st.markdown("This dashboard compares a Deep RL Agent against a Constraint Programming (CP) baseline.")

# Sidebar Parameters
st.sidebar.header("Grid Parameters")
pen_val = st.sidebar.slider("Renewable Penetration", 0.5, 2.0, 1.0, 0.1)
var_val = st.sidebar.slider("Demand Variability", 0.0, 0.5, 0.2, 0.05)
cap_val = st.sidebar.slider("Emission Cap", 100, 1000, 500, 50)

start_sim = st.sidebar.button("▶ Run Live Simulation", type="primary")

# Layout Placeholders
col1, col2 = st.columns(2)
placeholder_dispatch = col1.empty()
placeholder_policy = col2.empty()
col3, col4 = st.columns(2)
placeholder_sens = col3.empty()
placeholder_metrics = col4.empty()

if start_sim:
    env = SmartGridEnv(pen_val, var_val, cap_val)
    state = env.reset()
    hours = list(range(24))
    demand_h, rl_solar, rl_wind, rl_gas, rl_coal = [], [], [], [], []
    rl_cost_h, rl_emiss_h, rl_supply_h = [], [], []
    cp_solar, cp_wind, cp_gas, cp_coal = [], [], [], []
    cp_cost_h, cp_emiss_h, cp_supply_h = [], [], []

    # Output 3: Pre-compute sensitivity analysis so it doesn't block animation
    test_caps = list(range(50, 1050, 100))
    avg_cp, avg_rl = [], []
    for c in test_caps:
        cp_c, rl_c = [], []
        for _ in range(5):
            d = 700 * (1 + random.uniform(-var_val, var_val))
            s = random.uniform(0, 300 * pen_val)
            w = random.uniform(0, 200 * pen_val)
            c_a = cp_dispatch(d, s, w, c)
            cp_c.append(c_a[2]*5 + c_a[3]*10)
            st_arr = np.array([d, s, w], dtype=np.float32)
            r_a = agent.act(st_arr, training=False)
            rl_c.append(max(r_a[2],0)*5 + max(r_a[3],0)*10)
        avg_cp.append(np.mean(cp_c))
        avg_rl.append(np.mean(rl_c))
    
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=test_caps, y=avg_cp, mode='lines+markers', name='CP Baseline', line=dict(color='green')))
    fig3.add_trace(go.Scatter(x=test_caps, y=avg_rl, mode='lines+markers', name='RL Agent', line=dict(color='blue')))
    fig3.update_layout(title='Output 3: Sensitivity Analysis', xaxis_title='Emission Cap', yaxis_title='Avg Cost ($)', height=380, margin=dict(l=0, r=0, t=30, b=0))
    placeholder_sens.plotly_chart(fig3, use_container_width=True)

    # Output 1 & 2: Live Animation Loop
    for h in hours:
        demand, solar_avail, wind_avail = state
        demand_h.append(demand)

        # RL
        rl_act = agent.act(state, training=False)
        s_u = min(max(rl_act[0],0), solar_avail); w_u = min(max(rl_act[1],0), wind_avail)
        g_u = max(rl_act[2],0); c_u = max(rl_act[3],0)
        rl_solar.append(s_u); rl_wind.append(w_u); rl_gas.append(g_u); rl_coal.append(c_u)
        rl_supply_h.append(s_u + w_u + g_u + c_u)
        rl_cost_h.append(g_u*5 + c_u*10)
        rl_emiss_h.append(g_u*2 + c_u*5)

        # CP
        cp_act = cp_dispatch(demand, solar_avail, wind_avail, cap_val)
        cp_solar.append(cp_act[0]); cp_wind.append(cp_act[1]); cp_gas.append(cp_act[2]); cp_coal.append(cp_act[3])
        cp_supply_h.append(sum(cp_act))
        cp_cost_h.append(cp_act[2]*5 + cp_act[3]*10)
        cp_emiss_h.append(cp_act[2]*2 + cp_act[3]*5)

        state, _, _, _ = env.step(list(rl_act))

        # Build Live Figs
        cur_h = hours[:h+1]
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=cur_h, y=rl_solar, stackgroup='one', name='Solar', line=dict(color='#FFD700')))
        fig1.add_trace(go.Scatter(x=cur_h, y=rl_wind, stackgroup='one', name='Wind', line=dict(color='#87CEEB')))
        fig1.add_trace(go.Scatter(x=cur_h, y=rl_gas, stackgroup='one', name='Gas', line=dict(color='#FF8C00')))
        fig1.add_trace(go.Scatter(x=cur_h, y=rl_coal, stackgroup='one', name='Coal', line=dict(color='#2F4F4F')))
        fig1.add_trace(go.Scatter(x=cur_h, y=demand_h, mode='lines+markers', name='Demand', line=dict(color='red', dash='dash')))
        fig1.update_layout(title=f'Output 1: Live Dispatch (Hour {h+1}/24)', xaxis=dict(range=[0, 23]), yaxis=dict(range=[0, 1500]), height=400, margin=dict(l=0, r=0, t=30, b=0))

        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig2.add_trace(go.Scatter(x=cur_h, y=rl_cost_h, name='RL Cost', line=dict(color='blue')), secondary_y=False)
        fig2.add_trace(go.Scatter(x=cur_h, y=cp_cost_h, name='CP Cost', line=dict(color='green', dash='dot')), secondary_y=False)
        fig2.add_trace(go.Scatter(x=cur_h, y=rl_emiss_h, name='RL Emiss', line=dict(color='orange')), secondary_y=True)
        fig2.add_trace(go.Scatter(x=cur_h, y=cp_emiss_h, name='CP Emiss', line=dict(color='red', dash='dot')), secondary_y=True)
        fig2.add_trace(go.Scatter(x=cur_h, y=[cap_val]*(h+1), name='Cap', line=dict(color='black', dash='dash')), secondary_y=True)
        fig2.update_layout(title='Output 2: Policy (Cost vs Emiss)', xaxis=dict(range=[0, 23]), height=400, margin=dict(l=0, r=0, t=30, b=0))

        placeholder_dispatch.plotly_chart(fig1, use_container_width=True)
        placeholder_policy.plotly_chart(fig2, use_container_width=True)
        placeholder_metrics.info(f"⏳ Simulating Hour {h+1} of 24...")
        
        time.sleep(0.15)

    # Output 4: Final Metrics
    dem_arr = np.array(demand_h)
    rl_stab = max(0, 100 - np.mean(np.abs(np.array(rl_supply_h) - dem_arr)) / np.mean(dem_arr) * 100)
    cp_stab = max(0, 100 - np.mean(np.abs(np.array(cp_supply_h) - dem_arr)) / np.mean(dem_arr) * 100)
    rl_ren = ((sum(rl_solar)+sum(rl_wind))/sum(rl_supply_h)*100) if sum(rl_supply_h) > 0 else 0
    cp_ren = ((sum(cp_solar)+sum(cp_wind))/sum(cp_supply_h)*100) if sum(cp_supply_h) > 0 else 0
    worst = sum(demand_h)*5
    rl_red = ((worst-sum(rl_emiss_h))/worst*100) if worst > 0 else 0
    cp_red = ((worst-sum(cp_emiss_h))/worst*100) if worst > 0 else 0

    placeholder_metrics.success(f\"\"\"
    ### ✅ Output 4: Performance Metrics
    | Metric | Deep RL | CP Baseline |
    |--------|:---:|:---:|
    | **Grid Stability** | {rl_stab:.1f}% | {cp_stab:.1f}% |
    | **Renewable Util** | {rl_ren:.1f}% | {cp_ren:.1f}% |
    | **Emission Red.** | {rl_red:.1f}% | {cp_red:.1f}% |
    | **Cost (24h)** | ${sum(rl_cost_h):.0f} | ${sum(cp_cost_h):.0f} |
    \"\"\")
else:
    placeholder_dispatch.info("👈 Set your parameters and click **Run Live Simulation**")
