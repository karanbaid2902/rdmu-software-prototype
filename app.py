"""
Smart City Energy Grid Optimizer - Command Center Dashboard
Deep Reinforcement Learning (DQN) + LP Constraint Optimisation

This dashboard SERVES a pre-trained agent (train/serve split): it loads
`dqn_grid_weights.pth` if present, otherwise trains once and caches the result.
It never retrains on every slider change.
"""
import os, time, math, random
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import linprog
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ---------- Reproducibility ----------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

st.set_page_config(page_title="Smart City Command Center", layout="wide",
                   initial_sidebar_state="expanded")

# ==========================================
# Theme (dark glassmorphism)
# ==========================================
st.markdown("""
<style>
    .stApp, .stApp > header { background-color: #0B0E14 !important; }
    html, body, [class*="st-"], [class*="st-"] *, div, span, p, label,
    h1, h2, h3, h4, h5, h6, li { color: #FFFFFF !important; }
    div[data-testid="stMetricLabel"] *, div[data-testid="stMetricValue"] *,
    div[data-testid="stMetricDelta"] * { color: #FFFFFF !important; }
    button[data-baseweb="tab"] *, div[data-baseweb="tab-list"] * { color: #FFFFFF !important; }
    div[data-baseweb="select"] *, div[role="listbox"] * { color: #FFFFFF !important; }
    .title-glow {
        font-family: 'Inter', sans-serif; font-weight: 800; font-size: 2.5rem;
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0px;
    }
    .subtitle { color: #8B95A5 !important; font-size: 1.0rem; margin-top: -6px; }
    div[data-testid="metric-container"] {
        background-color: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.30) !important; border-radius: 12px;
        padding: 15px 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.4); backdrop-filter: blur(10px);
    }
    div[data-baseweb="select"] > div, div[role="listbox"] { background-color: #12161E !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; background-color: transparent; }
    .stTabs [aria-selected="true"] p { border-bottom: 3px solid #00C9FF !important; }
    div[data-testid="stAlert"] {
        background-color: rgba(0,201,255,0.1) !important; border: 1px solid #00C9FF !important; }
    [data-testid="stSidebar"] { background-color: #12161E !important; border-right: 1px solid #2D3748 !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# Core: Environment, LP baseline, DQN  (identical to the corrected notebook)
# ==========================================
GAS_FRACTIONS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
N_ACTIONS = len(GAS_FRACTIONS)

class SmartGridEnv:
    COST_GAS, COST_COAL = 6.0, 3.0
    EMIS_GAS, EMIS_COAL = 2.0, 5.0
    GAS_CAP,  COAL_CAP  = 500.0, 700.0
    W_EMISSION, W_UNMET = 4.0, 20.0
    REWARD_SCALE        = 1000.0

    def __init__(self, renewable_penetration=1.0, demand_variability=0.2, emission_cap=1500,
                 base_demand=700, episode_length=24, season="Summer", is_tou=False):
        self.state_dim, self.action_dim = 5, N_ACTIONS
        self.renewable_penetration = renewable_penetration
        self.demand_variability = demand_variability
        self.emission_cap = emission_cap
        self.base_demand = base_demand
        self.episode_length = episode_length
        self.season = season
        self.is_tou = is_tou

    def reset(self):
        self.time_step = 0
        self.raw = self._sample()
        return self._norm(self.raw)

    def _sample(self):
        d_mult = 1.2 if self.season == "Winter" else 1.0
        s_mult = 0.6 if self.season == "Winter" else 1.2
        w_mult = 1.3 if self.season == "Winter" else 0.8
        peak = 1.0 + 0.4 * math.exp(-0.1 * (self.time_step - 18) ** 2)
        demand = self.base_demand * d_mult * peak * (1 + random.uniform(-self.demand_variability, self.demand_variability))
        solar = random.uniform(0, 300 * self.renewable_penetration * s_mult)
        wind  = random.uniform(0, 200 * self.renewable_penetration * w_mult)
        return np.array([demand, solar, wind, self.time_step, self.emission_cap], dtype=np.float32)

    def _norm(self, raw):
        d, s, w, t, cap = raw
        return np.array([d/1000.0, s/300.0, w/200.0, t/24.0, cap/1500.0], dtype=np.float32)

    def _prices(self):
        peak = self.is_tou and 16 <= self.time_step <= 20
        return (self.COST_GAS * (1.6 if peak else 1.0), self.COST_COAL * (1.6 if peak else 1.0))

    def dispatch(self, raw, action_idx):
        demand, solar_av, wind_av, _, cap = raw
        cost_gas, cost_coal = self._prices()
        gas_frac = GAS_FRACTIONS[action_idx]
        solar_u, wind_u = solar_av, wind_av
        residual = max(0.0, demand - solar_u - wind_u)
        gas_u  = min(gas_frac * residual,       self.GAS_CAP)
        coal_u = min((1 - gas_frac) * residual, self.COAL_CAP)
        unmet = max(0.0, demand - (solar_u + wind_u + gas_u + coal_u))
        add = min(unmet, self.GAS_CAP  - gas_u);  gas_u  += add; unmet -= add
        add = min(unmet, self.COAL_CAP - coal_u); coal_u += add; unmet -= add
        cost = gas_u * cost_gas + coal_u * cost_coal
        emissions = gas_u * self.EMIS_GAS + coal_u * self.EMIS_COAL
        return dict(solar=solar_u, wind=wind_u, gas=gas_u, coal=coal_u, cost=cost,
                    emissions=emissions, unmet=unmet, supply=solar_u + wind_u + gas_u + coal_u, demand=demand)

    def step(self, action_idx):
        d = self.dispatch(self.raw, action_idx)
        over = max(0.0, d["emissions"] - self.emission_cap)
        reward = -(d["cost"] + self.W_EMISSION * over + self.W_UNMET * d["unmet"]) / self.REWARD_SCALE
        self.time_step += 1
        done = self.time_step >= self.episode_length
        self.raw = self._sample()
        return self._norm(self.raw), reward, done, d


def lp_dispatch(raw, env):
    demand, solar_av, wind_av, _, cap = raw
    cost_gas, cost_coal = env._prices()
    c = [0, 0, cost_gas, cost_coal]
    A_eq = [[1, 1, 1, 1]]; b_eq = [demand]
    A_ub = [[0, 0, env.EMIS_GAS, env.EMIS_COAL]]; b_ub = [cap]
    bounds = [(0, solar_av), (0, wind_av), (0, env.GAS_CAP), (0, env.COAL_CAP)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if res.success:
        s, w, g, co = res.x; feasible = True
    else:
        s, w = solar_av, wind_av
        residual = max(0.0, demand - s - w)
        g = min(residual, env.GAS_CAP); co = min(residual - g, env.COAL_CAP); feasible = False
    return dict(solar=s, wind=w, gas=g, coal=co, cost=g*cost_gas + co*cost_coal,
                emissions=g*env.EMIS_GAS + co*env.EMIS_COAL, supply=s + w + g + co,
                demand=demand, feasible=feasible)


class DQN(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(),
                                 nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, n_actions))
    def forward(self, x): return self.net(x)


class DQNAgent:
    def __init__(self, state_dim, n_actions):
        self.n_actions = n_actions
        self.gamma, self.batch_size = 0.95, 64
        self.eps_start, self.eps_end, self.eps_decay = 1.0, 0.02, 2000
        self.steps_done = 0
        self.policy_net = DQN(state_dim, n_actions)
        self.target_net = DQN(state_dim, n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict()); self.target_net.eval()
        self.opt = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.loss_fn = nn.SmoothL1Loss()
        self.buffer = deque(maxlen=10000)

    def epsilon(self):
        return self.eps_end + (self.eps_start - self.eps_end) * math.exp(-self.steps_done / self.eps_decay)

    def act(self, state, training=True):
        if training:
            self.steps_done += 1
            if random.random() < self.epsilon(): return random.randrange(self.n_actions)
        with torch.no_grad():
            q = self.policy_net(torch.FloatTensor(state).unsqueeze(0))
            return int(q.argmax(dim=1).item())

    def store(self, *t): self.buffer.append(t)

    def train_step(self):
        if len(self.buffer) < self.batch_size: return None
        s, a, r, ns, d = zip(*random.sample(self.buffer, self.batch_size))
        s = torch.FloatTensor(np.array(s)); ns = torch.FloatTensor(np.array(ns))
        a = torch.LongTensor(a).unsqueeze(1); r = torch.FloatTensor(r); d = torch.FloatTensor(d)
        q_sa = self.policy_net(s).gather(1, a).squeeze(1)
        with torch.no_grad():
            q_next = self.target_net(ns).max(dim=1)[0]
            target = r + (1 - d) * self.gamma * q_next
        loss = self.loss_fn(q_sa, target)
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0); self.opt.step()
        return loss.item()

    def update_target(self): self.target_net.load_state_dict(self.policy_net.state_dict())


# ==========================================
# Train/serve split: load weights, else train ONCE and cache
# ==========================================
WEIGHTS_PATH = "dqn_grid_weights.pth"

def _random_config_env():
    return SmartGridEnv(renewable_penetration=random.uniform(0.5, 2.0),
                        demand_variability=random.uniform(0.0, 0.5),
                        emission_cap=random.choice([400, 800, 1200, 1600, 2000, 2600, 3200]),
                        base_demand=random.randint(500, 1500),
                        season=random.choice(["Summer", "Winter"]),
                        is_tou=random.choice([True, False]))

@st.cache_resource(show_spinner=False)
def get_agent():
    agent = DQNAgent(5, N_ACTIONS)
    if os.path.exists(WEIGHTS_PATH):
        agent.policy_net.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
        agent.policy_net.eval()
        return agent, "loaded"
    # Domain-randomised training so ONE agent serves every slider setting
    for ep in range(500):
        env = _random_config_env(); s = env.reset()
        for _ in range(24):
            a = agent.act(s, training=True)
            ns, r, done, _ = env.step(a)
            agent.store(s, a, r, ns, float(done)); agent.train_step(); s = ns
        if (ep + 1) % 10 == 0: agent.update_target()
    try: torch.save(agent.policy_net.state_dict(), WEIGHTS_PATH)
    except Exception: pass
    return agent, "trained"


# ==========================================
# UI
# ==========================================
st.markdown('<p class="title-glow">&#9889; Smart City Energy Command Center</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Deep Reinforcement Learning (DQN) vs LP Constraint Optimisation &mdash; live dispatch simulator</p>', unsafe_allow_html=True)

st.sidebar.markdown("### Primary Controls")
pen_val = st.sidebar.slider("Renewable Penetration", 0.5, 2.0, 1.0, 0.1, help="Scales available solar & wind power.")
cap_val = st.sidebar.slider("Emission Cap (units)", 400, 3200, 1500, 100, help="Emission ceiling; part of the agent's state.")

st.sidebar.markdown("### Environment")
season_val = st.sidebar.selectbox("Season", ["Summer", "Winter"], help="Winter: higher demand & wind. Summer: high solar.")
var_val = st.sidebar.slider("Demand Variability", 0.0, 0.5, 0.2, 0.05, help="Random hourly demand fluctuation.")

with st.sidebar.expander("Advanced Settings"):
    base_dem = st.slider("Base Demand (kW)", 500, 1500, 700, 50)
    tou_val = st.checkbox("Enable Time-Of-Use Pricing", value=False,
                          help="Raises fossil-fuel cost during peak hours (16:00-20:00).")

st.sidebar.markdown("---")
start_sim = st.sidebar.button("LAUNCH LIVE SIMULATION", type="primary", use_container_width=True)

# Load the pre-trained agent (one-time)
with st.spinner("Preparing Deep RL agent..."):
    agent, source = get_agent()
st.sidebar.caption(f"DQN agent: {'pre-trained weights loaded' if source=='loaded' else 'trained & cached'} \u2705")

# Top metrics
m1, m2, m3, m4 = st.columns(4)
met_stab, met_ren, met_red, met_cost = m1.empty(), m2.empty(), m3.empty(), m4.empty()
met_stab.metric("Grid Stability Index", "-- %")
met_ren.metric("Renewable Util. Rate", "-- %")
met_red.metric("Emission Reduction", "-- %")
met_cost.metric("Total 24h Cost (RL)", "$ --")

tab1, tab2, tab3, tab4 = st.tabs(["Output 1: Live Dispatch", "Output 2: Policy Comparison",
                                  "Output 3: Sensitivity Analysis", "Output 4: Export"])
with tab1:
    prog_bar = st.empty(); chart_dispatch = st.empty()
with tab2:
    chart_policy = st.empty()
with tab3:
    chart_sens = st.empty()
with tab4:
    export_placeholder = st.empty()

DARK = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#FFFFFF"), height=460, hovermode="x unified",
            hoverlabel=dict(bgcolor="#12161E", font_color="#FFFFFF"))

# Idle placeholders
if not start_sim:
    chart_dispatch.info("Adjust the controls and press **LAUNCH LIVE SIMULATION**.")
    f = go.Figure(); f.update_layout(title="Output 3: Sensitivity (awaiting run)", **DARK)
    chart_sens.plotly_chart(f, use_container_width=True)
    export_placeholder.info("Run a simulation to unlock raw-data export.")

# ==========================================
# Simulation
# ==========================================
if start_sim:
    sim = SmartGridEnv(pen_val, var_val, cap_val, base_dem, season=season_val, is_tou=tou_val)
    state = sim.reset()
    hours = list(range(24))
    demand_h = []
    rl = {k: [] for k in ["solar", "wind", "gas", "coal", "cost", "emis", "supply", "unmet"]}
    cp = {k: [] for k in ["cost", "emis", "supply"]}

    # ---- Output 3: Monte-Carlo sensitivity (compute up front) ----
    caps = list(range(600, 3001, 300)); N_MC = 40
    rl_m, rl_s, lp_m, lp_s = [], [], [], []
    for cap in caps:
        e = SmartGridEnv(pen_val, var_val, cap, base_dem, season=season_val, is_tou=tou_val)
        a_rl, a_lp = [], []
        for _ in range(N_MC):
            e.time_step = random.randint(0, 23); raw = e._sample()
            a_rl.append(e.dispatch(raw, agent.act(e._norm(raw), training=False))["cost"])
            a_lp.append(lp_dispatch(raw, e)["cost"])
        rl_m.append(np.mean(a_rl)); rl_s.append(np.std(a_rl))
        lp_m.append(np.mean(a_lp)); lp_s.append(np.std(a_lp))
    rl_m, rl_s, lp_m, lp_s = map(np.array, (rl_m, rl_s, lp_m, lp_s))
    f3 = go.Figure()
    f3.add_trace(go.Scatter(x=caps+caps[::-1], y=list(rl_m+rl_s)+list((rl_m-rl_s)[::-1]),
                            fill="toself", fillcolor="rgba(0,201,255,0.15)", line=dict(width=0),
                            showlegend=False, hoverinfo="skip"))
    f3.add_trace(go.Scatter(x=caps, y=rl_m, mode="lines+markers", name="Deep RL Agent", line=dict(color="#00C9FF", width=3)))
    f3.add_trace(go.Scatter(x=caps+caps[::-1], y=list(lp_m+lp_s)+list((lp_m-lp_s)[::-1]),
                            fill="toself", fillcolor="rgba(0,255,0,0.12)", line=dict(width=0),
                            showlegend=False, hoverinfo="skip"))
    f3.add_trace(go.Scatter(x=caps, y=lp_m, mode="lines+markers", name="LP Optimum", line=dict(color="#00FF00", width=3)))
    f3.update_layout(title=f"Emission Cap vs Cost (mean &plusmn; 1 std, {season_val})",
                     xaxis_title="Emission Cap Threshold", yaxis_title="Average Hourly Cost ($)", **DARK)
    chart_sens.plotly_chart(f3, use_container_width=True)

    # ---- 24h rollout with animated Output 1 ----
    for h in hours:
        prog_bar.progress((h + 1) / 24, text=f"Simulating Hour {h + 1} / 24...")
        raw = sim.raw.copy()
        a = agent.act(state, training=False)
        d = sim.dispatch(raw, a); lp = lp_dispatch(raw, sim)
        demand_h.append(d["demand"])
        for k in ["solar", "wind", "gas", "coal", "cost", "supply", "unmet"]: rl[k].append(d[k])
        rl["emis"].append(d["emissions"])
        cp["cost"].append(lp["cost"]); cp["emis"].append(lp["emissions"]); cp["supply"].append(lp["supply"])
        state, _, _, _ = sim.step(a)

        # live metrics (REAL values; cost delta vs LP optimum)
        dem = np.array(demand_h)
        stab = 100 * (1 - sum(rl["unmet"]) / sum(demand_h))
        ren = 100 * (sum(rl["solar"]) + sum(rl["wind"])) / max(sum(rl["supply"]), 1e-9)
        residual_coal = sum(max(0, demand_h[i] - rl["solar"][i] - rl["wind"][i]) for i in range(len(demand_h))) * 5
        red = 100 * (1 - sum(rl["emis"]) / residual_coal) if residual_coal > 0 else 0
        gap = (sum(rl["cost"]) / sum(cp["cost"]) - 1) * 100 if sum(cp["cost"]) > 0 else 0
        met_stab.metric("Grid Stability Index", f"{stab:.1f}%")
        met_ren.metric("Renewable Util. Rate", f"{ren:.1f}%")
        met_red.metric("Emission Reduction", f"{red:.1f}%")
        met_cost.metric("Total 24h Cost (RL)", f"${sum(rl['cost']):,.0f}", f"{gap:+.1f}% vs LP optimum",
                        delta_color="inverse")

        cur = hours[:h + 1]
        f1 = go.Figure()
        for nm, col, fc in [("solar", "#F4D03F", "rgba(244,208,63,0.7)"), ("wind", "#5DADE2", "rgba(93,173,226,0.7)"),
                            ("gas", "#E67E22", "rgba(230,126,34,0.7)"), ("coal", "#7B7D7D", "rgba(123,125,125,0.7)")]:
            f1.add_trace(go.Scatter(x=cur, y=rl[nm], stackgroup="one", name=nm.capitalize(),
                                    line=dict(color=col), fillcolor=fc))
        f1.add_trace(go.Scatter(x=cur, y=demand_h, mode="lines+markers", name="Demand Target",
                                line=dict(color="#FFFFFF", dash="dash", width=3)))
        if tou_val:
            f1.add_vrect(x0=16, x1=20, fillcolor="rgba(255,0,0,0.10)", layer="below", line_width=0,
                         annotation_text="Peak Pricing", annotation_position="top left")
        f1.update_layout(title=f"Live Energy Allocation vs Target Demand ({season_val})",
                         xaxis=dict(range=[0, 23], title="Hour of Day"),
                         yaxis=dict(title="Energy Supply (kW)"), uirevision="x",
                         legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), **DARK)
        chart_dispatch.plotly_chart(f1, use_container_width=True)
        time.sleep(0.05)

    prog_bar.empty()

    # ---- Output 2: cost & emissions, RL vs LP (rendered once) ----
    f2 = make_subplots(specs=[[{"secondary_y": True}]])
    f2.add_trace(go.Scatter(x=hours, y=rl["cost"], name="RL Cost", line=dict(color="#00C9FF", width=3)), secondary_y=False)
    f2.add_trace(go.Scatter(x=hours, y=cp["cost"], name="LP Cost", line=dict(color="#00FF00", dash="dot", width=2)), secondary_y=False)
    f2.add_trace(go.Scatter(x=hours, y=rl["emis"], name="RL Emissions", line=dict(color="#FF9900", width=3)), secondary_y=True)
    f2.add_trace(go.Scatter(x=hours, y=cp["emis"], name="LP Emissions", line=dict(color="#FF3366", dash="dot", width=2)), secondary_y=True)
    f2.add_trace(go.Scatter(x=hours, y=[cap_val]*24, name="Emission Cap", line=dict(color="#FFFFFF", dash="dash")), secondary_y=True)
    f2.update_layout(title="Cost & Emissions: Deep RL vs LP Constraint Solver", xaxis_title="Hour of Day", **DARK)
    f2.update_yaxes(title_text="Cost ($)", secondary_y=False)
    f2.update_yaxes(title_text="Emissions (units)", secondary_y=True)
    chart_policy.plotly_chart(f2, use_container_width=True)

    st.toast("24-Hour Simulation Complete!", icon="\u2705")

    # ---- Output 4: export ----
    df = pd.DataFrame({"Hour": hours, "Demand": demand_h,
                       "RL_Solar": rl["solar"], "RL_Wind": rl["wind"], "RL_Gas": rl["gas"], "RL_Coal": rl["coal"],
                       "RL_Cost": rl["cost"], "RL_Emissions": rl["emis"],
                       "LP_Cost": cp["cost"], "LP_Emissions": cp["emis"]})
    with export_placeholder:
        st.success("Simulation data ready for export.")
        st.download_button("Download Raw Data (CSV)", data=df.to_csv(index=False).encode("utf-8"),
                           file_name="smart_grid_simulation.csv", mime="text/csv", type="primary")
        st.dataframe(df, use_container_width=True)
