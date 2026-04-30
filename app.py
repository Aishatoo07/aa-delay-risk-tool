import streamlit as st
import pandas as pd
import numpy as np
import pickle
try:
    import shap
    SHAP_AVAILABLE = True
except:
    SHAP_AVAILABLE = False
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="AA Flight Delay Risk Tool",
    page_icon="✈",
    layout="wide"
)

# ============================================
# CUSTOM STYLING — Professional Ops Style
# ============================================
st.markdown("""
<style>
    .main { background-color: #0a1628; }
    .stApp { background-color: #0a1628; }
    
    .header-box {
        background: linear-gradient(
            135deg, #003366, #0055a5
        );
        padding: 20px 30px;
        border-radius: 10px;
        margin-bottom: 25px;
        border-left: 5px solid #ff6600;
    }
    .header-title {
        color: white;
        font-size: 28px;
        font-weight: bold;
        margin: 0;
    }
    .header-sub {
        color: #aac4e0;
        font-size: 14px;
        margin: 5px 0 0 0;
    }
    .metric-box {
        background: #0d2137;
        border: 1px solid #1a3a5c;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .risk-high {
        background: #3d0000;
        border: 2px solid #ff2222;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .risk-medium {
        background: #2d1f00;
        border: 2px solid #ff9900;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .risk-low {
        background: #002d00;
        border: 2px solid #00cc44;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .rec-box {
        background: #0d2137;
        border-left: 4px solid #ff6600;
        border-radius: 8px;
        padding: 15px 20px;
        margin: 8px 0;
    }
    .section-title {
        color: #aac4e0;
        font-size: 13px;
        font-weight: bold;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }
    .divider {
        border: none;
        border-top: 1px solid #1a3a5c;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# LOAD MODEL + EXPLAINER
# ============================================
@st.cache_resource
def load_model():
    with open('xgb_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('shap_explainer.pkl', 'rb') as f:
        explainer = pickle.load(f)
    with open('feature_cols.json', 'r') as f:
        cols = json.load(f)
    return model, explainer, cols

model, explainer, feature_cols = load_model()

# ============================================
# HEADER
# ============================================
st.markdown("""
<div class="header-box">
    <p class="header-title">
        ✈ AA Flight Delay Risk & Recovery Tool
    </p>
    <p class="header-sub">
        ORAA-Style Operational Decision Support — 
        American Airlines Hub Operations
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================
# SIDEBAR — FLIGHT INPUTS
# ============================================
with st.sidebar:
    st.markdown("### ✈ Flight Details")
    st.markdown("---")

    origin = st.selectbox(
        "Origin Hub",
        options=["DFW", "CLT", "MIA"],
        help="American Airlines hub airport"
    )

    dest_options = {
        "DFW": ["LAX", "ORD", "JFK", "MIA", 
                "PHX", "CLT", "BOS", "SEA"],
        "CLT": ["DFW", "LAX", "ORD", "JFK", 
                "MIA", "PHX", "BOS", "ATL"],
        "MIA": ["DFW", "LAX", "ORD", "JFK", 
                "CLT", "PHX", "BOS", "ATL"]
    }

    dest = st.selectbox(
        "Destination",
        options=dest_options[origin]
    )

    month = st.selectbox(
        "Month",
        options={
            1:"January",   2:"February",
            3:"March",     4:"April",
            5:"May",       6:"June",
            7:"July",      8:"August",
            9:"September", 10:"October",
            11:"November", 12:"December"
        }.items(),
        format_func=lambda x: x[1]
    )
    month_num = month[0]

    day_options = {
        1:"Monday",   2:"Tuesday",
        3:"Wednesday",4:"Thursday",
        5:"Friday",   6:"Saturday",
        7:"Sunday"
    }
    day = st.selectbox(
        "Day of Week",
        options=day_options.items(),
        format_func=lambda x: x[1]
    )
    day_num = day[0]

    dep_hour = st.slider(
        "Departure Hour",
        min_value=0,
        max_value=23,
        value=8,
        help="0 = midnight, 12 = noon, 17 = 5pm"
    )

    distance = st.number_input(
        "Distance (miles)",
        min_value=100,
        max_value=3000,
        value=870
    )

    st.markdown("---")
    predict_btn = st.button(
        "🔍 Analyze Flight Risk",
        use_container_width=True
    )

# ============================================
# POLICY LAYER FUNCTION
# ============================================
def get_recommendations(prob, origin, 
                         dep_hour, day_num,
                         month_num):
    recommendations = []

    if prob >= 0.60:
        recommendations = [
            "🔴 Flag flight for ops team review",
            "⏱ Add 15-min turnaround buffer",
            "🔗 Activate connection protection protocol",
            "🚪 Pre-assign backup gate",
            "📢 Brief ground crew on delay risk"
        ]
        if dep_hour >= 16:
            recommendations.append(
                "🌙 Consider earlier departure bank"
            )
        if month_num in [6, 7, 8, 12]:
            recommendations.append(
                "📅 Peak season — escalate to senior ops"
            )

    elif prob >= 0.30:
        recommendations = [
            "🟡 Monitor flight status closely",
            "⏱ Add 5-min buffer if possible",
            "🔗 Watch connection windows",
            "📋 Standard ops procedures apply"
        ]
        if day_num == 5:
            recommendations.append(
                "📅 Friday traffic — increase monitoring"
            )

    else:
        recommendations = [
            "🟢 No action required",
            "✅ Standard ops procedures apply",
            "📋 Normal turnaround expected"
        ]

    return recommendations

# ============================================
# FEATURE BUILDER
# ============================================
def build_features(origin, dest, month_num,
                   day_num, dep_hour, distance):

    # Season mapping
    season_map = {
        12:0, 1:0, 2:0,   # Winter = 0
        3:1,  4:1, 5:1,   # Spring = 1
        6:2,  7:2, 8:2,   # Summer = 2
        9:3, 10:3, 11:3   # Fall   = 3
    }

    # Airport encoding
    origin_enc = {"DFW": 0, "CLT": 1, "MIA": 2}
    dest_enc   = {
        "ATL":0,"BOS":1,"CLT":2,"DFW":3,
        "JFK":4,"LAX":5,"MIA":6,"ORD":7,
        "PHX":8,"SEA":9
    }

    # Historical avg delays (from your dataset)
    avg_delay_map = {
        ("DFW", 17): 18.5, ("DFW", 8): 8.2,
        ("CLT", 17): 15.3, ("CLT", 8): 6.1,
        ("MIA", 17): 20.1, ("MIA", 8): 7.4
    }

    avg_delay_origin_hour = avg_delay_map.get(
        (origin, dep_hour), 12.0
    )

    avg_delay_route_map = {
        ("DFW","LAX"):14.2, ("DFW","ORD"):16.8,
        ("DFW","JFK"):15.1, ("CLT","MIA"):11.3,
        ("MIA","ORD"):18.2, ("CLT","ORD"):13.5
    }
    avg_delay_route = avg_delay_route_map.get(
        (origin, dest), 12.0
    )

    features = {
        'MONTH':                 month_num,
        'DAY_OF_WEEK':           day_num,
        'DEP_HOUR':              dep_hour,
        'IS_WEEKEND':            1 if day_num >= 6 else 0,
        'PEAK_HOUR':             1 if dep_hour in [7,8,17,18] else 0,
        'DISTANCE':              distance,
        'GROUND_TIME':           22.0,
        'AIR_RATIO':             0.85,
        'FLIGHT_DENSITY':        45,
        'AVG_DELAY_ORIGIN_HOUR': avg_delay_origin_hour,
        'AVG_DELAY_ROUTE':       avg_delay_route,
        'TAXI_OUT':              16.0,
        'TAXI_IN':               8.0,
        'ORIGIN_ENC':            origin_enc.get(origin, 0),
        'DEST_ENC':              dest_enc.get(dest, 0),
        'SEASON_ENC':            season_map[month_num]
    }

    return pd.DataFrame([features])[feature_cols]

# ============================================
# MAIN DASHBOARD OUTPUT
# ============================================
if predict_btn:

    # Build features
    X_input = build_features(
        origin, dest, month_num,
        day_num, dep_hour, distance
    )

    # Predict
    prob     = model.predict_proba(X_input)[0][1]
    prob_pct = prob * 100

    # Risk level
    if prob >= 0.60:
        risk_level = "HIGH"
        risk_color = "#ff2222"
        risk_class = "risk-high"
        risk_emoji = "🔴"
    elif prob >= 0.30:
        risk_level = "MEDIUM"
        risk_color = "#ff9900"
        risk_class = "risk-medium"
        risk_emoji = "🟡"
    else:
        risk_level = "LOW"
        risk_color = "#00cc44"
        risk_class = "risk-low"
        risk_emoji = "🟢"

    # Get recommendations
    recs = get_recommendations(
        prob, origin, dep_hour, 
        day_num, month_num
    )

    # ----------------------------------------
    # ROW 1 — Flight Summary
    # ----------------------------------------
    st.markdown(
        f"### {origin} → {dest} | "
        f"{list(day_options.values())[day_num-1]} | "
        f"Hour {dep_hour:02d}:00"
    )
    st.markdown('<hr class="divider">', 
                unsafe_allow_html=True)

    # ----------------------------------------
    # ROW 2 — Key Metrics
    # ----------------------------------------
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="{risk_class}">
            <p class="section-title">Delay Risk</p>
            <h1 style="color:{risk_color};margin:0">
                {prob_pct:.1f}%
            </h1>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="{risk_class}">
            <p class="section-title">Risk Level</p>
            <h1 style="color:{risk_color};margin:0">
                {risk_emoji} {risk_level}
            </h1>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-box">
            <p class="section-title">Route</p>
            <h2 style="color:white;margin:0">
                {origin} → {dest}
            </h2>
            <p style="color:#aac4e0;margin:0">
                {distance} miles
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-box">
            <p class="section-title">Departure</p>
            <h2 style="color:white;margin:0">
                {dep_hour:02d}:00
            </h2>
            <p style="color:#aac4e0;margin:0">
                {list(day_options.values())[day_num-1]}
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ----------------------------------------
    # ROW 3 — Recommendations + SHAP
    # ----------------------------------------
    col_left, col_right = st.columns([1, 1.5])

    with col_left:
        st.markdown(
            '<p class="section-title">'
            '⚙ Operational Recommendations'
            '</p>',
            unsafe_allow_html=True
        )
        for rec in recs:
            st.markdown(f"""
            <div class="rec-box">
                <p style="color:white;margin:0">
                    {rec}
                </p>
            </div>
            """, unsafe_allow_html=True)

    with col_right:
        st.markdown(
            '<p class="section-title">'
            '📊 SHAP — Why This Prediction?'
            '</p>',
            unsafe_allow_html=True
        )

        # SHAP waterfall for this flight
        shap_vals = explainer.shap_values(X_input)
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('#0d2137')
        ax.set_facecolor('#0d2137')

        shap.plots.waterfall(
            shap.Explanation(
                values=shap_vals[0],
                base_values=explainer.expected_value,
                data=X_input.iloc[0],
                feature_names=feature_cols
            ),
            show=False,
            max_display=10
        )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ----------------------------------------
    # ROW 4 — Cost Estimate
    # ----------------------------------------
    st.markdown('<hr class="divider">',
                unsafe_allow_html=True)
    st.markdown(
        '<p class="section-title">'
        '💰 Estimated Operational Impact'
        '</p>',
        unsafe_allow_html=True
    )

    col_a, col_b, col_c = st.columns(3)

    avg_delay_min  = 28
    intervention   = 10
    est_saving     = int(prob * avg_delay_min)
    est_cost       = est_saving * 74

    with col_a:
        st.metric(
            "Expected Delay",
            f"{est_saving} min",
            delta=None
        )
    with col_b:
        st.metric(
            "Est. Cost if Unmanaged",
            f"${est_cost:,}",
            delta=None
        )
    with col_c:
        st.metric(
            "Potential Saving",
            f"{intervention} min",
            "with early intervention"
        )

else:
    # ----------------------------------------
    # DEFAULT STATE — No prediction yet
    # ----------------------------------------
    st.markdown("""
    <div style="
        text-align:center;
        padding:60px;
        color:#aac4e0;
    ">
        <h2>✈ Enter flight details in the 
        sidebar and click</h2>
        <h2>"Analyze Flight Risk"</h2>
        <p>to get delay risk score, SHAP 
        explanation, and operational 
        recommendations</p>
    </div>
    """, unsafe_allow_html=True)
