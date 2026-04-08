"""
Streamlit Dashboard — NCAA Basketball Transfer Portal Predictor & ACC Rankings

Two-tab interface:
  Tab 1: Transfer Success Predictor (interactive form + prediction)
  Tab 2: ACC Conference Rankings (table + charts)

Run:
  streamlit run app/streamlit_app.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import xgboost as xgb
from src.feature_engineering import get_transfer_feature_columns, get_ranking_feature_columns


# ─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NCAA Basketball Analytics",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODELS_DIR = Path("models")
PLOTS_DIR = Path("plots")


# ─── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0a0e1a;
    }

    /* Headers */
    h1, h2, h3 {
        color: #e2e8f0 !important;
        font-family: 'Segoe UI', system-ui, sans-serif !important;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px;
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
    }
    [data-testid="stMetricValue"] {
        color: #f1f5f9 !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1e293b;
        border-radius: 10px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #94a3b8;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important;
        color: white !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
    }
    [data-testid="stSidebar"] .stMarkdown p {
        color: #cbd5e1;
    }

    /* Tables */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 8px 24px;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #60a5fa, #3b82f6);
    }

    /* Prediction result card */
    .prediction-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid #334155;
        margin: 16px 0;
    }
    .prediction-card h3 {
        margin-top: 0;
    }

    /* Divider */
    hr {
        border-color: #1e293b;
    }

    /* Slider labels */
    .stSlider label, .stNumberInput label, .stSelectbox label {
        color: #cbd5e1 !important;
    }
</style>
""", unsafe_allow_html=True)


# ─── Model Loading ───────────────────────────────────────────────────────────

@st.cache_resource
def load_transfer_model():
    path = MODELS_DIR / "transfer_xgb_final.json"
    if not path.exists():
        return None
    model = xgb.XGBClassifier()
    model.load_model(str(path))
    return model


@st.cache_data
def load_acc_rankings():
    path = MODELS_DIR / "acc_rankings.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_metrics(name: str):
    path = MODELS_DIR / f"{name}_model_metrics.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🏀 NCAA Basketball Analytics")
    st.markdown("---")
    st.markdown(
        "**Transfer Portal Success Predictor** & **ACC Team Ranking Engine**"
    )
    st.markdown("")

    transfer_metrics = load_metrics("transfer")
    ranking_metrics = load_metrics("ranking")

    if transfer_metrics:
        st.markdown("#### Transfer Model")
        st.markdown(f"- PR-AUC: **{transfer_metrics.get('avg_pr_auc', 0):.3f}**")
        st.markdown(f"- ROC-AUC: **{transfer_metrics.get('avg_roc_auc', 0):.3f}**")
        st.markdown(f"- Transfers: **{transfer_metrics.get('n_transfers', 0):,}**")

    if ranking_metrics:
        st.markdown("#### Ranking Model")
        st.markdown(f"- MAE: **{ranking_metrics.get('avg_mae', 0):.2f}**")
        st.markdown(f"- R²: **{ranking_metrics.get('avg_r2', 0):.3f}**")

    st.markdown("---")
    st.markdown(
        "Built with XGBoost, LightGBM, Polars, DuckDB"
    )
    st.markdown(
        "[GitHub](https://github.com/Bh-Ayush) · "
        "[Portfolio](https://ayush-hb.com)"
    )


# ─── Main Content ────────────────────────────────────────────────────────────

st.markdown("# 🏀 NCAA Basketball Transfer Portal Analytics")
st.markdown("Predict transfer outcomes and explore ACC team rankings")
st.markdown("")

tab1, tab2 = st.tabs(["📊 Transfer Predictor", "🏆 ACC Rankings"])


# ─── Tab 1: Transfer Predictor ───────────────────────────────────────────────

with tab1:
    st.markdown("### Predict Transfer Success")
    st.markdown(
        "Enter a player's pre-transfer stats and destination school context. "
        "The model predicts whether the transfer will **improve or maintain** "
        "their adjusted offensive rating."
    )

    model = load_transfer_model()

    if model is None:
        st.error(
            "Transfer model not found. Run `python src/transfer_model.py` first."
        )
    else:
        col_left, col_right = st.columns([1, 1], gap="large")

        with col_left:
            st.markdown("#### Player Pre-Transfer Stats")

            pc1, pc2, pc3 = st.columns(3)
            with pc1:
                pre_ortg = st.number_input("ORtg", value=108.5, min_value=70.0,
                                           max_value=140.0, step=0.5,
                                           help="Offensive rating per 100 possessions")
                pre_usg = st.number_input("Usage %", value=22.0, min_value=8.0,
                                          max_value=40.0, step=0.5)
                pre_bpm = st.number_input("BPM", value=2.0, min_value=-15.0,
                                          max_value=20.0, step=0.5)
                pre_obpm = st.number_input("OBPM", value=1.5, min_value=-10.0,
                                           max_value=15.0, step=0.5)
                pre_dbpm = st.number_input("DBPM", value=0.5, min_value=-10.0,
                                           max_value=10.0, step=0.5)
                pre_mpg = st.number_input("MPG", value=28.0, min_value=3.0,
                                          max_value=40.0, step=0.5)
            with pc2:
                pre_efg = st.number_input("eFG%", value=0.52, min_value=0.20,
                                          max_value=0.75, step=0.01, format="%.3f")
                pre_ts_pct = st.number_input("TS%", value=0.56, min_value=0.25,
                                             max_value=0.75, step=0.01, format="%.3f")
                pre_ast_pct = st.number_input("AST%", value=14.0, min_value=0.0,
                                              max_value=45.0, step=0.5)
                pre_to_pct = st.number_input("TO%", value=17.0, min_value=5.0,
                                             max_value=35.0, step=0.5)
                pre_ftr = st.number_input("FTR", value=0.32, min_value=0.05,
                                          max_value=0.75, step=0.01, format="%.3f")
                pre_g = st.number_input("Games", value=30, min_value=5, max_value=40)
            with pc3:
                pre_orb_pct = st.number_input("ORB%", value=4.5, min_value=0.0,
                                              max_value=18.0, step=0.5)
                pre_drb_pct = st.number_input("DRB%", value=12.0, min_value=0.0,
                                              max_value=30.0, step=0.5)
                pre_blk_pct = st.number_input("BLK%", value=2.0, min_value=0.0,
                                              max_value=15.0, step=0.5)
                pre_stl_pct = st.number_input("STL%", value=1.8, min_value=0.0,
                                              max_value=6.0, step=0.1)
                pre_porpag = st.number_input("PORPAG", value=3.0, min_value=-10.0,
                                             max_value=15.0, step=0.5)

            st.markdown("#### Player Profile")
            pp1, pp2, pp3 = st.columns(3)
            with pp1:
                recruiting_stars = st.selectbox("Recruiting Stars", [0, 2, 3, 4, 5],
                                                index=2)
            with pp2:
                class_year = st.selectbox("Class Year",
                                          ["Freshman", "Sophomore", "Junior", "Senior"],
                                          index=1)
                class_year_ord = {"Freshman": 1, "Sophomore": 2, "Junior": 3, "Senior": 4}[class_year]
            with pp3:
                height_in = st.number_input("Height (inches)", value=77,
                                            min_value=69, max_value=87)

        with col_right:
            st.markdown("#### Origin School Context")
            oc1, oc2 = st.columns(2)
            with oc1:
                origin_adj_o = st.number_input("Origin Adj O", value=105.0,
                                               min_value=80.0, max_value=130.0, step=0.5)
                origin_adj_d = st.number_input("Origin Adj D", value=100.0,
                                               min_value=80.0, max_value=120.0, step=0.5)
            with oc2:
                origin_adj_t = st.number_input("Origin Tempo", value=68.0,
                                               min_value=58.0, max_value=78.0, step=0.5)
                origin_barthag = st.number_input("Origin Barthag", value=0.65,
                                                 min_value=0.0, max_value=1.0,
                                                 step=0.01, format="%.3f")
            origin_is_power = st.checkbox("Origin is Power Conference", value=False)

            st.markdown("#### Destination School Context")
            dc1, dc2 = st.columns(2)
            with dc1:
                dest_adj_o = st.number_input("Dest Adj O", value=112.0,
                                             min_value=80.0, max_value=130.0, step=0.5)
                dest_adj_d = st.number_input("Dest Adj D", value=95.0,
                                             min_value=80.0, max_value=120.0, step=0.5)
            with dc2:
                dest_adj_t = st.number_input("Dest Tempo", value=70.0,
                                             min_value=58.0, max_value=78.0, step=0.5)
                dest_barthag = st.number_input("Dest Barthag", value=0.85,
                                               min_value=0.0, max_value=1.0,
                                               step=0.01, format="%.3f")
            dest_is_power = st.checkbox("Destination is Power Conference", value=True)

        # ── Predict ──
        st.markdown("---")

        if st.button("🔮 Predict Transfer Outcome", use_container_width=True):
            # Compute derived features
            features = {
                "pre_ortg": pre_ortg, "pre_usg": pre_usg, "pre_efg": pre_efg,
                "pre_ts_pct": pre_ts_pct, "pre_ast_pct": pre_ast_pct,
                "pre_to_pct": pre_to_pct, "pre_orb_pct": pre_orb_pct,
                "pre_drb_pct": pre_drb_pct, "pre_blk_pct": pre_blk_pct,
                "pre_stl_pct": pre_stl_pct, "pre_ftr": pre_ftr,
                "pre_porpag": pre_porpag, "pre_bpm": pre_bpm,
                "pre_obpm": pre_obpm, "pre_dbpm": pre_dbpm,
                "pre_mpg": pre_mpg, "pre_g": pre_g,
                "recruiting_stars": recruiting_stars,
                "class_year_ord": class_year_ord, "height_in": height_in,
                "origin_adj_o": origin_adj_o, "origin_adj_d": origin_adj_d,
                "origin_adj_t": origin_adj_t, "origin_barthag": origin_barthag,
                "origin_is_power": int(origin_is_power),
                "dest_adj_o": dest_adj_o, "dest_adj_d": dest_adj_d,
                "dest_adj_t": dest_adj_t, "dest_barthag": dest_barthag,
                "dest_is_power": int(dest_is_power),
                # Deltas
                "delta_adj_o": dest_adj_o - origin_adj_o,
                "delta_adj_d": dest_adj_d - origin_adj_d,
                "delta_barthag": dest_barthag - origin_barthag,
                "delta_tempo": dest_adj_t - origin_adj_t,
                "conf_jump_direction": int(dest_is_power) - int(origin_is_power),
                # Engineered
                "pre_ortg_usg_product": pre_ortg * pre_usg / 100,
                "pre_minutes_share": pre_mpg / 40,
                "pre_ast_to_ratio": pre_ast_pct / max(pre_to_pct, 1),
                "pre_shooting_composite": pre_efg + 0.1 * pre_ftr,
                "pre_total_reb_pct": pre_orb_pct + pre_drb_pct,
                "pre_defensive_stocks": pre_blk_pct + pre_stl_pct,
                # Interactions
                "usg_x_quality_delta": pre_usg * (dest_barthag - origin_barthag),
                "mpg_x_conf_jump": pre_mpg * float(int(dest_is_power) - int(origin_is_power)),
                "bpm_x_off_delta": pre_bpm * (dest_adj_o - origin_adj_o),
            }

            feature_cols = get_transfer_feature_columns()
            X = pd.DataFrame([{col: features[col] for col in feature_cols}])

            prob = float(model.predict_proba(X)[0, 1])
            predicted = "✅ Success" if prob >= 0.5 else "❌ Decline"
            color = "#22c55e" if prob >= 0.5 else "#ef4444"

            # Result display
            st.markdown("---")
            r1, r2, r3, r4 = st.columns(4)
            with r1:
                st.metric("Prediction", predicted)
            with r2:
                st.metric("Success Probability", f"{prob:.1%}")
            with r3:
                confidence = "High" if (prob >= 0.7 or prob <= 0.3) else \
                             "Medium" if (prob >= 0.6 or prob <= 0.4) else "Low"
                st.metric("Confidence", confidence)
            with r4:
                delta = dest_barthag - origin_barthag
                st.metric("Quality Delta", f"{delta:+.3f}")

            # Probability gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={"suffix": "%", "font": {"size": 40, "color": "#e2e8f0"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#475569",
                             "tickfont": {"color": "#94a3b8"}},
                    "bar": {"color": color},
                    "bgcolor": "#1e293b",
                    "bordercolor": "#334155",
                    "steps": [
                        {"range": [0, 30], "color": "#7f1d1d"},
                        {"range": [30, 50], "color": "#78350f"},
                        {"range": [50, 70], "color": "#1e3a5f"},
                        {"range": [70, 100], "color": "#14532d"},
                    ],
                    "threshold": {
                        "line": {"color": "#f1f5f9", "width": 2},
                        "thickness": 0.8,
                        "value": 50,
                    },
                },
            ))
            fig_gauge.update_layout(
                height=250,
                margin=dict(t=30, b=10, l=30, r=30),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#e2e8f0"},
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Context summary
            st.markdown("#### Transfer Context Summary")
            ctx1, ctx2, ctx3 = st.columns(3)
            with ctx1:
                st.markdown(f"**Barthag Jump**: {origin_barthag:.3f} → {dest_barthag:.3f}")
                st.markdown(f"**Off. Efficiency**: {origin_adj_o:.1f} → {dest_adj_o:.1f}")
            with ctx2:
                st.markdown(f"**Def. Efficiency**: {origin_adj_d:.1f} → {dest_adj_d:.1f}")
                st.markdown(f"**Tempo**: {origin_adj_t:.1f} → {dest_adj_t:.1f}")
            with ctx3:
                jump = "Power → Power" if origin_is_power and dest_is_power else \
                       "Mid → Power" if not origin_is_power and dest_is_power else \
                       "Power → Mid" if origin_is_power and not dest_is_power else \
                       "Mid → Mid"
                st.markdown(f"**Conference Jump**: {jump}")
                st.markdown(f"**Player**: {class_year}, {height_in}in, {recruiting_stars}⭐")


# ─── Tab 2: ACC Rankings ────────────────────────────────────────────────────

with tab2:
    st.markdown("### ACC 2025-26 Predicted Team Rankings")
    st.markdown(
        "Predicted efficiency margins for the **current ACC membership** "
        "(18 teams including Cal, Stanford, SMU post-realignment). "
        "Trained on all D1 teams, filtered to ACC for presentation."
    )

    rankings = load_acc_rankings()

    if rankings is None:
        st.error(
            "Rankings not found. Run `python src/ranking_model.py` first."
        )
    else:
        # Key metrics
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Teams Ranked", len(rankings))
        with m2:
            st.metric("Top Team", rankings.iloc[0]["team"])
        with m3:
            avg_em = rankings["predicted_eff_margin"].mean()
            st.metric("Avg Pred. EM", f"{avg_em:+.1f}")
        with m4:
            ranking_m = load_metrics("ranking")
            st.metric("Model R²", f"{ranking_m.get('avg_r2', 0):.3f}")

        st.markdown("---")

        # Two-column layout: table + chart
        chart_col, table_col = st.columns([1.2, 1], gap="large")

        with chart_col:
            # Horizontal bar chart
            df_plot = rankings.sort_values("predicted_rank", ascending=True).copy()
            df_plot["color"] = df_plot["predicted_eff_margin"].apply(
                lambda x: "#22c55e" if x > 25 else "#3b82f6" if x > 15 else "#f59e0b" if x > 0 else "#ef4444"
            )
            df_plot["label"] = df_plot.apply(
                lambda r: f"{int(r['predicted_rank'])}. {r['team']}", axis=1
            )

            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=df_plot["label"],
                x=df_plot["predicted_eff_margin"],
                orientation="h",
                marker_color=df_plot["color"],
                text=df_plot["predicted_eff_margin"].apply(lambda x: f"{x:+.1f}"),
                textposition="outside",
                textfont=dict(color="#e2e8f0", size=11),
            ))
            fig.update_layout(
                height=600,
                margin=dict(t=20, b=20, l=10, r=40),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(
                    title="Predicted Efficiency Margin",
                    gridcolor="#1e293b",
                    zerolinecolor="#475569",
                    tickfont=dict(color="#94a3b8"),
                    titlefont=dict(color="#94a3b8"),
                ),
                yaxis=dict(
                    tickfont=dict(color="#e2e8f0", size=12),
                    autorange="reversed",
                ),
                font=dict(color="#e2e8f0"),
            )
            st.plotly_chart(fig, use_container_width=True)

        with table_col:
            st.markdown("#### Detailed Rankings")

            display_df = rankings[[
                "predicted_rank", "team", "predicted_eff_margin",
                "current_eff_margin", "predicted_change", "rank_change",
                "returning_production_pct",
            ]].copy()

            display_df.columns = [
                "Rank", "Team", "Pred EM", "Curr EM",
                "Δ EM", "Δ Rank", "Ret. Prod %"
            ]
            display_df["Ret. Prod %"] = (display_df["Ret. Prod %"] * 100).round(1)
            display_df["Rank"] = display_df["Rank"].astype(int)
            display_df["Δ Rank"] = display_df["Δ Rank"].astype(int)

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                height=580,
            )

        st.markdown("---")

        # Scatter: current vs predicted
        st.markdown("#### Current vs Predicted Efficiency Margin")

        fig_scatter = px.scatter(
            rankings,
            x="current_eff_margin",
            y="predicted_eff_margin",
            text="team",
            color="rank_change",
            color_continuous_scale=["#ef4444", "#6b7280", "#22c55e"],
            color_continuous_midpoint=0,
            labels={
                "current_eff_margin": "Current Efficiency Margin",
                "predicted_eff_margin": "Predicted Efficiency Margin",
                "rank_change": "Rank Change",
            },
        )
        # Diagonal reference line
        min_val = min(rankings["current_eff_margin"].min(),
                      rankings["predicted_eff_margin"].min()) - 3
        max_val = max(rankings["current_eff_margin"].max(),
                      rankings["predicted_eff_margin"].max()) + 3
        fig_scatter.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode="lines", line=dict(dash="dash", color="#475569"),
            showlegend=False,
        ))
        fig_scatter.update_traces(
            textposition="top center",
            marker=dict(size=12),
            textfont=dict(size=10, color="#e2e8f0"),
        )
        fig_scatter.update_layout(
            height=500,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="#1e293b", tickfont=dict(color="#94a3b8"),
                        titlefont=dict(color="#94a3b8")),
            yaxis=dict(gridcolor="#1e293b", tickfont=dict(color="#94a3b8"),
                        titlefont=dict(color="#94a3b8")),
            font=dict(color="#e2e8f0"),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.caption(
            "Points above the diagonal are projected to improve. "
            "Points below are projected to regress."
        )

        # Model evaluation plots
        st.markdown("---")
        st.markdown("#### Model Evaluation Plots")

        eval_col1, eval_col2 = st.columns(2)

        pr_curve = PLOTS_DIR / "transfer_pr_curve.png"
        cal_curve = PLOTS_DIR / "transfer_calibration.png"
        rank_actual = PLOTS_DIR / "ranking_actual_vs_pred.png"
        rank_resid = PLOTS_DIR / "ranking_residuals.png"

        with eval_col1:
            if pr_curve.exists():
                st.image(str(pr_curve), caption="Transfer Model — PR Curve")
            if rank_actual.exists():
                st.image(str(rank_actual), caption="Ranking Model — Actual vs Predicted")

        with eval_col2:
            if cal_curve.exists():
                st.image(str(cal_curve), caption="Transfer Model — Calibration")
            if rank_resid.exists():
                st.image(str(rank_resid), caption="Ranking Model — Residuals")
