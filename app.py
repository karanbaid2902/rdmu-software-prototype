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
import pandas as pd

st.set_page_config(page_title="Smart City Command Center", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# Premium CSS Styling (Brighter Fonts)
# ==========================================
st.markdown("""
<style>
    /* Glowing Title */
    .title-glow {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 2.5rem;
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    /* Bulletproof Bright Text Override */
    p, span, label, h1, h2, h3, h4, h5, h6, div[data-baseweb="select"] div {
        color: #F8F9FA !important;
    }
    
    /* Ensure dropdown menus (selectboxes) have dark backgrounds to match the theme */
    div[role="listbox"] {
        background-color: #12161E !important;
    }
    div[role="listbox"] span {
        color: #F8F9FA !important;
    }
    .subtitle {
        color: #B0BEC5 !important;
        font-size: 1.1rem;
        margin-bottom: 20px;
    }

    /* Glassmorphism Metric Cards */
    div[data-testid="metric-container"] {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 12px;
        padding: 15px 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.4);
        backdrop-filter: blur(10px);
    }
    div[data-testid="metric-container"] label {
        color: #B0BEC5 !important;
        font-weight: 600;
        font-size: 1rem;
    }
    div[data-testid="metric-container"] div {
        color: #FFFFFF !important;
        font-weight: 800;
        text-shadow: 0px 0px 5px rgba(255,255,255,0.2);
    }

    /* Clean Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #B0BEC5 !important;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        color: #00C9FF !important;
        border-bottom: 3px solid #00C9FF !important;
    }
    
    /* Sleek Sidebar */
    [data-testid="stSidebar"] {
        background-color: #12161E;
        border-right: 1px solid #2D3748;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# Core Classes (Env, CP, DQN)
# ==========================================
class SmartGridEnv:
    COST_SOLAR, COST_WIND = 0, 0
    EMIS_SOLAR, EMIS_WIND, EMIS_GAS, EMIS_COAL = 0, 0, 2, 5

    def __init__(self, renewable_penetration=1.0, demand_variability=0.2, emission_cap=500, base_demand=700, episode_length=24, season="Summer"):
        self.state_dim = 3
        self.action_dim = 4
        self.renewable_penetration = renewable_penetration
        self.demand_variability = demand_variability
        self.emission_cap = emission_cap
        self.base_demand = base_demand
        self.episode_length = episode_length
        self.season = season

    def reset(self):
        self.time_step = 0
        self.state = self._sample_state()
        return self.state.copy()

    def _sample_state(self):
        # Apply season filters
        d_mult = 1.2 if self.season == "Winter" else 1.0 # Higher demand in Winter
        s_mult = 0.6 if self.season == "Winter" else 1.2 # Less solar in Winter
        w_mult = 1.3 if self.season == "Winter" else 0.8 # More wind in Winter
        
        # Peak demand hours simulation (around hour 18)
        peak_factor = 1.0 + 0.4 * math.exp(-0.1 * (self.time_step - 18)**2)
        
        demand = self.base_demand * d_mult * peak_factor * (1 + random.uniform(-self.demand_variability, self.demand_variability))
        solar = random.uniform(0, 300 * self.renewable_penetration * s_mult)
        wind  = random.uniform(0, 200 * self.renewable_penetration * w_mult)
        return np.array([demand, solar, wind], dtype=np.float32)

    def step(self, action, is_tou=False):
        demand, solar_avail, wind_avail = self.state
        s_use = min(max(action[0], 0), solar_avail)
        w_use = min(max(action[1], 0), wind_avail)
        g_use = max(action[2], 0)
        c_use = max(action[3], 0)
        total_supply = s_use + w_use + g_use + c_use
        
        # TOU Pricing logic (Cost spikes during peak hours 16-20)
        cost_gas = 8 if (is_tou and 16 <= self.time_step <= 20) else 5
        cost_coal = 15 if (is_tou and 16 <= self.time_step <= 20) else 10
        
        cost = g_use * cost_gas + c_use * cost_coal
        emissions = g_use * self.EMIS_GAS + c_use * self.EMIS_COAL
        
        w_cost, w_emission, w_stability = 1.0, 5.0, 10.0
        stability_penalty = w_stability * abs(total_supply - demand)
        emission_penalty  = w_emission * max(0, emissions - self.emission_cap)
        cost_penalty      = w_cost * cost
        
        # Prevent network collapse by mapping penalties to a positive reward range
        penalties = stability_penalty + emission_penalty + cost_penalty
        reward = max(0.0, 50000.0 - penalties)
        
        self.time_step += 1
        done = self.time_step >= self.episode_length
        self.state = self._sample_state()
        return self.state.copy(), reward, done, {}

def cp_dispatch(demand, solar_avail, wind_avail, emission_cap, time_step, is_tou):
    cost_gas = 8 if (is_tou and 16 <= time_step <= 20) else 5
    cost_coal = 15 if (is_tou and 16 <= time_step <= 20) else 10
    
    c = [0, 0, cost_gas, cost_coal]
    A_eq = [[1, 1, 1, 1]]
    b_eq = [demand]
    A_ub = [[0, 0, 2, 5]]
    b_ub = [emission_cap]
    bounds = [(0, solar_avail), (0, wind_avail), (0, None), (0, None)]
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    if result.success: return list(result.x)
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
    def forward(self, x): return self.net(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim, self.action_dim = state_dim, action_dim
        self.gamma, self.batch_size = 0.99, 64
        self.eps_start, self.eps_end, self.eps_decay = 1.0, 0.01, 100
        self.steps_done = 0
        self.policy_net, self.target_net = DQN(state_dim, action_dim), DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.criterion = nn.SmoothL1Loss()
        self.replay_buffer = deque(maxlen=5000)

    def get_epsilon(self):
        return self.eps_end + (self.eps_start - self.eps_end) * math.exp(-self.steps_done / self.eps_decay)

    def act(self, state, training=True):
        self.steps_done += 1
        if training and random.random() < self.get_epsilon():
            return np.array([random.uniform(0, 300), random.uniform(0, 200), random.uniform(0, 400), random.uniform(0, 400)], dtype=np.float32)
        with torch.no_grad():
            st = torch.FloatTensor(state).unsqueeze(0)
            action = self.policy_net(st).squeeze(0).numpy() + 1e-2  # Prevent exact zero
        return action * (state[0] / (action.sum()))

    def store(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size: return
        batch = random.sample(list(self.replay_buffer), self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states_t, rewards_t = torch.FloatTensor(np.array(states)), torch.FloatTensor(rewards)
        next_t, dones_t = torch.FloatTensor(np.array(next_states)), torch.FloatTensor(dones)
        
        current_q = self.policy_net(states_t)
        with torch.no_grad():
            next_q = self.target_net(next_t)
            target_vals = rewards_t + (1 - dones_t) * self.gamma * next_q.mean(dim=1)
        
        loss = self.criterion(current_q.mean(dim=1), target_vals)
        self.optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

    def update_target(self): self.target_net.load_state_dict(self.policy_net.state_dict())

@st.cache_resource(show_spinner=False)
def get_trained_agent():
    env, agent = SmartGridEnv(season="Summer"), DQNAgent(3, 4)
    for ep in range(150):
        state = env.reset()
        for _ in range(24):
            action = agent.act(state, training=True)
            next_state, reward, done, _ = env.step(action)
            agent.store(state, action, reward, next_state, float(done))
            agent.train_step()
            state = next_state
        if (ep+1) % 10 == 0: agent.update_target()
    return agent

agent = get_trained_agent()

# ==========================================
# Layout & UI
# ==========================================
st.markdown('<p class="title-glow">⚡ Smart City Energy Command Center</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced Deep RL & Constraint Programming Optimizer Simulator</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("### 🎛️ Primary Controls")
pen_val = st.sidebar.slider("Renewable Penetration", 0.5, 2.0, 1.0, 0.1, help="Scales available solar & wind power.")
cap_val = st.sidebar.slider("Emission Cap (units)", 100, 1000, 500, 50, help="Hard limit for constraint solver.")

st.sidebar.markdown("### 🌡️ Environment Filters")
season_val = st.sidebar.selectbox("Season", ["Summer", "Winter"], help="Winter features higher demand and more wind, Summer features high solar.")
var_val = st.sidebar.slider("Demand Variability", 0.0, 0.5, 0.2, 0.05, help="Random fluctuation in hourly demand.")

with st.sidebar.expander("Advanced Settings"):
    base_dem = st.slider("Base Demand (kW)", 500, 1500, 700, 50)
    tou_val = st.checkbox("Enable Time-Of-Use Pricing", value=False, help="Spikes cost of traditional fuels during peak hours (16:00-20:00). Tests agent adaptability.")

st.sidebar.markdown("---")
start_sim = st.sidebar.button("▶ LAUNCH LIVE SIMULATION", type="primary", use_container_width=True)

# Metric placeholders at the top
m1, m2, m3, m4 = st.columns(4)
met_stab = m1.empty()
met_ren  = m2.empty()
met_red  = m3.empty()
met_cost = m4.empty()

# Initialize empty metrics
met_stab.metric("Grid Stability Index", "-- %")
met_ren.metric("Renewable Util. Rate", "-- %")
met_red.metric("Emission Reduction", "-- %")
met_cost.metric("Total 24h Cost", "$ --")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Output 1: Live Dispatch Dashboard", "📈 Output 2: Policy Visualization", "🔬 Output 3: Sensitivity Analysis", "📥 Export Data"])

# Placeholders inside tabs
with tab1:
    prog_bar = st.empty()
    chart_dispatch = st.empty()
with tab2:
    chart_policy = st.empty()
with tab3:
    chart_sens = st.empty()
with tab4:
    export_placeholder = st.empty()
    export_placeholder.info("Run a simulation to unlock raw data export.")

# Pre-render Sensitivity Analysis if not simulating
if not start_sim:
    test_caps = list(range(100, 1050, 100))
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=test_caps, y=[c*0.8 for c in test_caps], mode='lines', name='Waiting for run...', line=dict(color='#30363D', dash='dash')))
    fig3.update_layout(template="plotly_dark", title='Output 3: Sensitivity Analysis (Awaiting Run)', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400)
    chart_sens.plotly_chart(fig3, use_container_width=True)

# ==========================================
# Simulation Execution
# ==========================================
if start_sim:
    env = SmartGridEnv(pen_val, var_val, cap_val, base_dem, season=season_val)
    state = env.reset()
    hours = list(range(24))
    demand_h, rl_solar, rl_wind, rl_gas, rl_coal = [], [], [], [], []
    rl_cost_h, rl_emiss_h, rl_supply_h = [], [], []
    cp_solar, cp_wind, cp_gas, cp_coal = [], [], [], []
    cp_cost_h, cp_emiss_h, cp_supply_h = [], [], []

    # ----------------------------------------
    # Compute Sensitivity Analysis (Output 3)
    # ----------------------------------------
    test_caps = list(range(100, 1050, 100))
    avg_cp, avg_rl = [], []
    for c in test_caps:
        cp_c, rl_c = [], []
        for _ in range(3):
            d = base_dem * (1 + random.uniform(-var_val, var_val))
            s = random.uniform(0, 300 * pen_val)
            w = random.uniform(0, 200 * pen_val)
            c_a = cp_dispatch(d, s, w, c, 12, tou_val)
            cp_c.append(c_a[2]*5 + c_a[3]*10)
            r_a = agent.act(np.array([d, s, w], dtype=np.float32), training=False)
            rl_c.append(max(r_a[2],0)*5 + max(r_a[3],0)*10)
        avg_cp.append(np.mean(cp_c))
        avg_rl.append(np.mean(rl_c))
    
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=test_caps, y=avg_cp, mode='lines+markers', name='CP Baseline', line=dict(color='#00FF00', width=3)))
    fig3.add_trace(go.Scatter(x=test_caps, y=avg_rl, mode='lines+markers', name='Deep RL Agent', line=dict(color='#00C9FF', width=3)))
    fig3.update_layout(
        template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        title=f'Emission Cap vs. Cost Efficiency ({season_val})',
        xaxis_title='Emission Cap Threshold', yaxis_title='Average Hourly Cost ($)',
        height=450, uirevision='constant', hovermode='x unified'
    )
    chart_sens.plotly_chart(fig3, use_container_width=True)

    # ----------------------------------------
    # Live Animation Loop
    # ----------------------------------------
    for h in hours:
        prog_bar.progress((h + 1) / 24, text=f"⏳ Simulating Hour {h+1} / 24...")
        
        demand, solar_avail, wind_avail = state
        demand_h.append(demand)

        # TOU Costing
        cost_gas = 8 if (tou_val and 16 <= h <= 20) else 5
        cost_coal = 15 if (tou_val and 16 <= h <= 20) else 10

        # RL Agent Action
        rl_act = agent.act(state, training=False)
        s_u = min(max(rl_act[0],0), solar_avail); w_u = min(max(rl_act[1],0), wind_avail)
        g_u = max(rl_act[2],0); c_u = max(rl_act[3],0)
        rl_solar.append(s_u); rl_wind.append(w_u); rl_gas.append(g_u); rl_coal.append(c_u)
        rl_supply_h.append(s_u + w_u + g_u + c_u)
        rl_cost_h.append(g_u*cost_gas + c_u*cost_coal); rl_emiss_h.append(g_u*2 + c_u*5)

        # CP Baseline Action
        cp_act = cp_dispatch(demand, solar_avail, wind_avail, cap_val, h, tou_val)
        cp_solar.append(cp_act[0]); cp_wind.append(cp_act[1]); cp_gas.append(cp_act[2]); cp_coal.append(cp_act[3])
        cp_supply_h.append(sum(cp_act))
        cp_cost_h.append(cp_act[2]*cost_gas + cp_act[3]*cost_coal); cp_emiss_h.append(cp_act[2]*2 + cp_act[3]*5)

        state, _, _, _ = env.step(list(rl_act), is_tou=tou_val)

        # --- Update Top Metrics Live ---
        dem_arr = np.array(demand_h)
        rl_stab = max(0, 100 - np.mean(np.abs(np.array(rl_supply_h) - dem_arr)) / np.mean(dem_arr) * 100)
        rl_ren = ((sum(rl_solar)+sum(rl_wind))/sum(rl_supply_h)*100) if sum(rl_supply_h) > 0 else 0
        worst = sum(demand_h)*5
        rl_red = ((worst-sum(rl_emiss_h))/worst*100) if worst > 0 else 0
        
        met_stab.metric("Grid Stability Index", f"{rl_stab:.1f}%", f"{rl_stab - 90:.1f}%" if h > 0 else "0.0%")
        met_ren.metric("Renewable Util. Rate", f"{rl_ren:.1f}%", f"{rl_ren - 50:.1f}%" if h > 0 else "0.0%")
        met_red.metric("Emission Reduction", f"{rl_red:.1f}%", f"{rl_red - 30:.1f}%" if h > 0 else "0.0%")
        met_cost.metric("Total 24h Cost", f"${sum(rl_cost_h):.0f}", f"-${sum(cp_cost_h)-sum(rl_cost_h):.0f} vs CP" if h > 0 else "$0")

        # --- Draw Output 1: Live Dispatch ---
        cur_h = hours[:h+1]
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=cur_h, y=rl_solar, stackgroup='one', name='Solar', line=dict(color='#F4D03F'), fillcolor='rgba(244, 208, 63, 0.7)'))
        fig1.add_trace(go.Scatter(x=cur_h, y=rl_wind, stackgroup='one', name='Wind', line=dict(color='#5DADE2'), fillcolor='rgba(93, 173, 226, 0.7)'))
        fig1.add_trace(go.Scatter(x=cur_h, y=rl_gas, stackgroup='one', name='Gas', line=dict(color='#E67E22'), fillcolor='rgba(230, 126, 34, 0.7)'))
        fig1.add_trace(go.Scatter(x=cur_h, y=rl_coal, stackgroup='one', name='Coal', line=dict(color='#7B7D7D'), fillcolor='rgba(123, 125, 125, 0.7)'))
        fig1.add_trace(go.Scatter(x=cur_h, y=demand_h, mode='lines+markers', name='Demand Target', line=dict(color='#FF3366', dash='dash', width=3)))
        
        # Highlight TOU Peak Hours if enabled
        if tou_val:
            fig1.add_vrect(x0=16, x1=20, fillcolor="rgba(255, 0, 0, 0.1)", layer="below", line_width=0, annotation_text="Peak Pricing", annotation_position="top left")

        fig1.update_layout(
            template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            title=f'Live Energy Allocation vs Target Demand ({season_val})',
            xaxis=dict(range=[0, 23], title='Hour of Day'), yaxis=dict(range=[0, 2000], title='Energy Supply (kW)'),
            height=450, uirevision='constant', hovermode='x unified', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        chart_dispatch.plotly_chart(fig1, use_container_width=True)

        # --- Draw Output 2: Policy Comparison ---
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig2.add_trace(go.Scatter(x=cur_h, y=rl_cost_h, name='RL Cost', line=dict(color='#00C9FF', width=3)), secondary_y=False)
        fig2.add_trace(go.Scatter(x=cur_h, y=cp_cost_h, name='CP Cost', line=dict(color='#00FF00', dash='dot', width=2)), secondary_y=False)
        fig2.add_trace(go.Scatter(x=cur_h, y=rl_emiss_h, name='RL Emissions', line=dict(color='#FF9900', width=3)), secondary_y=True)
        fig2.add_trace(go.Scatter(x=cur_h, y=cp_emiss_h, name='CP Emissions', line=dict(color='#FF3366', dash='dot', width=2)), secondary_y=True)
        fig2.add_trace(go.Scatter(x=cur_h, y=[cap_val]*(h+1), name='Emission Cap', line=dict(color='#FFFFFF', dash='dash')), secondary_y=True)
        fig2.update_layout(
            template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            title='Cost & Emissions: Deep RL vs Constraint Programming',
            xaxis=dict(range=[0, 23], title='Hour of Day'),
            yaxis_title='Cost ($)', yaxis2_title='Emissions (units)',
            height=450, uirevision='constant', hovermode='x unified', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        chart_policy.plotly_chart(fig2, use_container_width=True)

        time.sleep(0.08) # Smooth animation

    prog_bar.empty()
    st.toast("✅ 24-Hour Simulation Complete!", icon="🎉")

    # Generate CSV Export
    df = pd.DataFrame({
        "Hour": hours, "Demand": demand_h,
        "RL_Solar": rl_solar, "RL_Wind": rl_wind, "RL_Gas": rl_gas, "RL_Coal": rl_coal,
        "RL_Cost": rl_cost_h, "RL_Emissions": rl_emiss_h,
        "CP_Cost": cp_cost_h, "CP_Emissions": cp_emiss_h
    })
    csv = df.to_csv(index=False).encode('utf-8')
    with export_placeholder:
        st.success("Simulation data is ready for export!")
        st.download_button("📥 Download Raw Data (CSV)", data=csv, file_name="smart_grid_simulation.csv", mime="text/csv", type="primary")
