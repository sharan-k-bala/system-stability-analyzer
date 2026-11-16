import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="System Stability Analyzer",
    page_icon="📈",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS - DASHBOARD DESIGN
# --------------------------------------------------

st.markdown("""
<style>
:root {
    --bg: #101010;
    --panel: #1b1b1b;
    --panel-raised: #242424;
    --border: #383838;
    --text: #eeeeee;
    --muted: #a6a6a6;
    --accent: #bdbdbd;
}

.stApp {
    background: #101010;
    color: var(--text);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: #171717;
    border-right: 1px solid #303030;
}

[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #d0d0d0;
}

.hero {
    background: linear-gradient(135deg, #101010, #252525);
    border: 1px solid #3a3a3a;
    padding: 35px;
    border-radius: 18px;
    margin-bottom: 25px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.28);
}

.hero h1 {
    color: #f2f2f2;
    font-size: 38px;
    margin-bottom: 10px;
}

.hero p {
    color: #bdbdbd;
    font-size: 17px;
}

.section-title {
    color: #d0d0d0;
    font-size: 27px;
    font-weight: bold;
    margin-top: 25px;
    margin-bottom: 15px;
}

.metric-card {
    background: linear-gradient(145deg, #1c1c1c, #262626);
    border: 1px solid #3a3a3a;
    border-radius: 15px;
    padding: 22px;
    text-align: center;
    min-height: 135px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.18);
}

.metric-label {
    color: #a8a8a8;
    font-size: 15px;
    margin-bottom: 12px;
}

.metric-value {
    color: #f0f0f0;
    font-size: 28px;
    font-weight: bold;
}

.stability-card {
    background: linear-gradient(135deg, #202020, #292929);
    border: 1px solid #666666;
    border-radius: 16px;
    padding: 25px;
    text-align: center;
    margin-top: 15px;
}

.stability-card h2 {
    color: #d6d6d6;
    font-size: 27px;
}

.stability-card p {
    color: #c2c2c2;
    font-size: 16px;
}

.info-card {
    background: #1d1d1d;
    border: 1px solid #383838;
    border-left: 4px solid #777777;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 15px;
    color: #dedede;
}

div[data-testid="stTabs"] button {
    font-size: 17px;
    font-weight: bold;
    color: #c7c7c7;
}

/* Keep Streamlit widgets in the same monochrome palette */
.stButton button, .stDownloadButton button {
    background: #2b2b2b;
    color: #eeeeee;
    border: 1px solid #555555;
}
.stButton button:hover, .stDownloadButton button:hover {
    background: #3a3a3a;
    color: #ffffff;
    border-color: #888888;
}

[data-testid="stDataFrame"], [data-testid="stTable"] {
    border: 1px solid #383838;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("""
<div class="hero">
    <h1>📈 System Stability Analyzer</h1>
    <p>
        A Mathematical Approach to Dynamic System Analysis
        Using Eigenvalues and Eigenvectors
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
This interactive application evaluates the stability of a discrete-time
dynamic system by analyzing its eigenvalues, eigenvectors, and state evolution.
""")

st.latex(r"x_{t+1} = Ax_t")

# --------------------------------------------------
# SIDEBAR INPUTS
# --------------------------------------------------

st.sidebar.markdown("## ⚙️ System Configuration")
st.sidebar.markdown("Modify the matrix and initial conditions.")

st.sidebar.markdown("### Matrix A")

a11 = st.sidebar.number_input("a11", value=0.8, step=0.1)
a12 = st.sidebar.number_input("a12", value=0.1, step=0.1)
a21 = st.sidebar.number_input("a21", value=0.2, step=0.1)
a22 = st.sidebar.number_input("a22", value=0.7, step=0.1)

A = np.array([
    [a11, a12],
    [a21, a22]
])

st.sidebar.markdown("### Initial State")

x1 = st.sidebar.number_input("Initial State 1", value=1.0)
x2 = st.sidebar.number_input("Initial State 2", value=2.0)

X = np.array([x1, x2])

time_steps = st.sidebar.slider(
    "Simulation Time Steps",
    min_value=5,
    max_value=50,
    value=20
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Change the matrix values to observe how the system stability changes."
)

# --------------------------------------------------
# EIGENVALUE CALCULATION
# --------------------------------------------------

eigenvalues, eigenvectors = np.linalg.eig(A)

# Stability Classification
# For discrete-time systems, eigenvalues on the unit circle require
# checking whether their algebraic and geometric multiplicities match.
unit_circle_tol = 1e-8
eigenvalue_magnitudes = np.abs(eigenvalues)

if np.all(eigenvalue_magnitudes < 1 - unit_circle_tol):
    stability = "ASYMPTOTICALLY STABLE"
    stability_color = "#bdbdbd"
    stability_description = (
        "All eigenvalues lie strictly inside the unit circle. "
        "The state variables converge to zero over time."
    )

elif np.any(eigenvalue_magnitudes > 1 + unit_circle_tol):
    stability = "UNSTABLE"
    stability_color = "#8f8f8f"
    stability_description = (
        "At least one eigenvalue lies outside the unit circle. "
        "The system is unstable and its state may diverge."
    )

else:
    # Check each distinct eigenvalue on the unit circle. A unit-circle
    # eigenvalue is semisimple when geometric multiplicity equals its
    # algebraic multiplicity (for this 2x2 matrix, this detects defective
    # repeated eigenvalues that cause unbounded growth).
    unit_eigenvalues = [
        value for value in eigenvalues
        if abs(abs(value) - 1) <= unit_circle_tol
    ]
    checked = []
    unit_circle_semisimple = True

    for value in unit_eigenvalues:
        if any(abs(value - previous) <= unit_circle_tol for previous in checked):
            continue
        checked.append(value)

        algebraic_multiplicity = sum(
            abs(candidate - value) <= unit_circle_tol
            for candidate in eigenvalues
        )
        geometric_multiplicity = 2 - np.linalg.matrix_rank(
            A.astype(complex) - value * np.eye(2), tol=unit_circle_tol
        )

        if geometric_multiplicity < algebraic_multiplicity:
            unit_circle_semisimple = False
            break

    if unit_circle_semisimple:
        stability = "MARGINALLY STABLE"
        stability_color = "#d0d0d0"
        stability_description = (
            "All eigenvalues are inside or on the unit circle, and the "
            "unit-circle eigenvalues are semisimple. The system remains "
            "bounded, but does not necessarily converge to zero."
        )
    else:
        stability = "UNSTABLE"
        stability_color = "#8f8f8f"
        stability_description = (
            "A repeated unit-circle eigenvalue is defective (its geometric "
            "multiplicity is smaller than its algebraic multiplicity). "
            "The system can exhibit unbounded growth."
        )

# --------------------------------------------------
# SUMMARY METRIC CARDS
# --------------------------------------------------

st.markdown(
    '<div class="section-title">📊 System Overview</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Eigenvalue 1</div>
        <div class="metric-value">{eigenvalues[0]:.4f}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Eigenvalue 2</div>
        <div class="metric-value">{eigenvalues[1]:.4f}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    spectral_radius = np.max(np.abs(eigenvalues))

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Spectral Radius</div>
        <div class="metric-value">{spectral_radius:.4f}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Time Steps</div>
        <div class="metric-value">{time_steps}</div>
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# STABILITY STATUS
# --------------------------------------------------

st.markdown(f"""
<div class="stability-card"
     style="border-color: {stability_color};">
    <h2 style="color: {stability_color};">
        ● {stability}
    </h2>
    <p>{stability_description}</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SYSTEM MATRIX
# --------------------------------------------------

st.markdown(
    '<div class="section-title">🧮 System Matrix</div>',
    unsafe_allow_html=True
)

matrix_col1, matrix_col2 = st.columns([1, 2])

with matrix_col1:
    st.latex(r"A =")

    st.dataframe(
        pd.DataFrame(A, columns=["Column 1", "Column 2"]),
        hide_index=True,
        use_container_width=True
    )

with matrix_col2:
    st.markdown("""
    <div class="info-card">
        <h4>Matrix Interpretation</h4>
        The system matrix defines how the current state variables
        influence the next state of the dynamic system.
        <br><br>
        The eigenvalues of this matrix determine the stability
        characteristics of the system.
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# EIGENVALUES AND EIGENVECTORS
# --------------------------------------------------

st.markdown(
    '<div class="section-title">🔬 Eigenvalue Analysis</div>',
    unsafe_allow_html=True
)

eig_col1, eig_col2 = st.columns(2)

with eig_col1:
    st.markdown("#### Eigenvalues")

    eigenvalue_df = pd.DataFrame({
        "Eigenvalue": eigenvalues,
        "Magnitude": np.abs(eigenvalues)
    })

    st.dataframe(
        eigenvalue_df.round(4),
        hide_index=True,
        use_container_width=True
    )

with eig_col2:
    st.markdown("#### Eigenvectors")

    eigenvector_df = pd.DataFrame(
        eigenvectors,
        columns=["Eigenvector 1", "Eigenvector 2"]
    )

    st.dataframe(
        eigenvector_df.round(4),
        use_container_width=True
    )

# --------------------------------------------------
# SYSTEM SIMULATION
# --------------------------------------------------

states = []

x = X.copy()

for t in range(time_steps):
    states.append(x.copy())
    x = A @ x

states = np.array(states)

# --------------------------------------------------
# VISUALIZATION TABS
# --------------------------------------------------

st.markdown(
    '<div class="section-title">📈 Dynamic System Visualization</div>',
    unsafe_allow_html=True
)

tab1, tab2, tab3 = st.tabs([
    "📉 State Evolution",
    "🔵 Phase Portrait",
    "📋 Simulation Data"
])

# --------------------------------------------------
# TAB 1 - STATE EVOLUTION
# --------------------------------------------------

with tab1:

    fig, ax = plt.subplots(figsize=(11, 5))

    fig.patch.set_facecolor("#1b1b1b")
    ax.set_facecolor("#1b1b1b")

    ax.plot(
        states[:, 0],
        color="#d4d4d4",
        marker="o",
        linewidth=3,
        markersize=6,
        label="State 1"
    )

    ax.plot(
        states[:, 1],
        color="#858585",
        marker="s",
        linewidth=3,
        markersize=6,
        label="State 2"
    )

    ax.set_xlabel("Time Steps", fontsize=12)
    ax.set_ylabel("State Values", fontsize=12)

    ax.set_title(
        "State Variables Over Time",
        fontsize=17,
        fontweight="bold"
    )

    ax.legend(facecolor="#252525", edgecolor="#555555", labelcolor="#dddddd")
    ax.tick_params(colors="#c7c7c7")
    ax.xaxis.label.set_color("#dddddd")
    ax.yaxis.label.set_color("#dddddd")
    ax.title.set_color("#eeeeee")
    for spine in ax.spines.values():
        spine.set_color("#555555")
    ax.grid(True, linestyle="--", color="#555555", alpha=0.35)

    st.pyplot(fig)

    st.caption(
        "This graph represents how the system state changes "
        "over the selected number of time steps."
    )

# --------------------------------------------------
# TAB 2 - PHASE PORTRAIT
# --------------------------------------------------

with tab2:

    fig2, ax2 = plt.subplots(figsize=(9, 5))

    fig2.patch.set_facecolor("#1b1b1b")
    ax2.set_facecolor("#1b1b1b")

    ax2.plot(
        states[:, 0],
        states[:, 1],
        color="#b0b0b0",
        linewidth=2.5,
        marker="o",
        markersize=5
    )

    ax2.scatter(
        states[0, 0],
        states[0, 1],
        color="#d8d8d8",
        s=120,
        label="Initial State"
    )

    ax2.scatter(
        states[-1, 0],
        states[-1, 1],
        color="#707070",
        s=120,
        label="Final State"
    )

    ax2.set_xlabel("State 1")
    ax2.set_ylabel("State 2")

    ax2.set_title(
        "Phase Portrait",
        fontsize=17,
        fontweight="bold"
    )

    ax2.legend(facecolor="#252525", edgecolor="#555555", labelcolor="#dddddd")
    ax2.tick_params(colors="#c7c7c7")
    ax2.xaxis.label.set_color("#dddddd")
    ax2.yaxis.label.set_color("#dddddd")
    ax2.title.set_color("#eeeeee")
    for spine in ax2.spines.values():
        spine.set_color("#555555")
    ax2.grid(True, linestyle="--", color="#555555", alpha=0.35)

    st.pyplot(fig2)

    st.caption(
        "The phase portrait shows the trajectory of the system "
        "in the state space."
    )

# --------------------------------------------------
# TAB 3 - SIMULATION DATA
# --------------------------------------------------

with tab3:

    df = pd.DataFrame(
        states,
        columns=["State 1", "State 2"]
    )

    df.index.name = "Time Step"

    st.dataframe(
        df.round(4),
        use_container_width=True
    )

    csv = df.to_csv().encode("utf-8")

    st.download_button(
        label="⬇️ Download Simulation Data",
        data=csv,
        file_name="system_stability_results.csv",
        mime="text/csv"
    )

# --------------------------------------------------
# MATHEMATICAL INTERPRETATION
# --------------------------------------------------

st.markdown(
    '<div class="section-title">📚 Mathematical Interpretation</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="info-card">

<h4>Stability Criterion</h4>

For a discrete-time linear system:

</div>
""", unsafe_allow_html=True)

st.latex(r"x_{t+1} = Ax_t")

st.markdown("""
The system is asymptotically stable if all eigenvalues of the
system matrix have magnitudes less than one.
""")

st.latex(r"|\lambda_i| < 1")

st.markdown("""
The eigenvalues determine whether the state variables converge
toward equilibrium, remain bounded, or diverge.

The eigenvectors describe the directions associated with
the corresponding eigenvalues.
""")

# --------------------------------------------------
# FINAL CONCLUSION
# --------------------------------------------------

st.markdown(
    '<div class="section-title">📝 Project Conclusion</div>',
    unsafe_allow_html=True
)

st.markdown(f"""
<div class="info-card">

The system has been analyzed using matrix-based mathematical modeling.

<strong>Final Stability Classification:</strong>
{stability}

The calculated eigenvalues and simulated state trajectories
provide insight into the dynamic behavior of the system.

</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; color:#94a3b8; padding:25px;">
    System Stability Analysis | Mathematics Micro Project
</div>
""", unsafe_allow_html=True)