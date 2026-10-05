"""
QuantumNet: Diabetic Retinopathy Detection & Quantum Scoring Web Interface
Built with Streamlit, TensorFlow (ResNet-50), and Google Cirq (4-Qubit VQC).
"""

import os
import io
import time
import pickle
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt

# Pipeline import
from pipeline import QuantumNetPipeline

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="QuantumNet | Diabetic Retinopathy Detection",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. CUSTOM CSS STYLING
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Main container styling */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0d9488 100%);
        padding: 24px 32px;
        border-radius: 16px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.25);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #ffffff;
    }
    .main-header p {
        margin-top: 8px;
        margin-bottom: 0;
        font-size: 1.05rem;
        opacity: 0.9;
        color: #e2e8f0;
    }
    
    /* Result Badge Cards */
    .badge-dr {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border: 2px solid #ef4444;
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        color: #991b1b;
        margin-bottom: 16px;
    }
    .badge-nondr {
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        border: 2px solid #22c55e;
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        color: #166534;
        margin-bottom: 16px;
    }
    
    /* Score Metric Card */
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 4px;
    }
    
    /* Risk advice card */
    .advice-card {
        background-color: #f1f5f9;
        border-left: 5px solid #0284c7;
        border-radius: 8px;
        padding: 14px 18px;
        margin-top: 14px;
        font-size: 0.95rem;
        color: #334155;
    }
    
    /* Quantum circuit display box */
    .circuit-box {
        font-family: 'Courier New', Courier, monospace;
        background-color: #0f172a;
        color: #38bdf8;
        padding: 16px;
        border-radius: 10px;
        overflow-x: auto;
        font-size: 0.82rem;
        line-height: 1.4;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 3. PIPELINE INITIALIZATION & CACHING
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing QuantumNet ResNet-50 & Cirq Pipeline...")
def get_pipeline():
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    return QuantumNetPipeline(models_dir=models_dir)


pipeline = get_pipeline()


# -----------------------------------------------------------------------------
# 4. SIDEBAR CONTROLS & CHECKPOINT MANAGEMENT
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/visible--v1.png", width=64)
    st.title("QuantumNet Controls")
    st.markdown("**Hybrid ResNet-50 + 4-Qubit VQC**")
    st.markdown("---")

    # Image Input Selection
    st.subheader("1. Input Retinal Image")
    input_source = st.radio(
        "Choose image source:",
        ["🖼️ Sample Retinas (Instant Demo)", "📁 Upload My Retinal Image"],
        index=0
    )

    selected_image = None
    image_name = ""

    if input_source == "🖼️ Sample Retinas (Instant Demo)":
        samples_dir = os.path.join(os.path.dirname(__file__), "samples")
        sample_options = {
            "🔴 Severe DR (APTOS Test #6 - Grade 2, Risk Score: 98.6%)": "aptos_test6_e499434242cc_dr.png",
            "🔴 Moderate DR (APTOS Test #1 - Grade 2, Risk Score: 96.4%)": "aptos_test1_44e951e45dca_dr.png",
            "🟢 Healthy Retina (APTOS Test #2 - Non-DR, Risk Score: 2.5%)": "aptos_test2_ef8c39eb9157_nondr.png",
            "🟢 Healthy Retina (APTOS Test #0 - Non-DR, Risk Score: 22.0%)": "aptos_test0_ae2c3f6312ef_nondr.png",
            "🔴 Proliferative DR (APTOS #00163)": "aptos_dr_severe_00163.png",
            "🟠 Moderate DR (APTOS #e4994)": "aptos_dr_moderate_e4994.png",
            "🟡 Mild DR (APTOS #bb452)": "aptos_dr_mild_bb452.png",
            "🟢 Healthy Human Retina": "real_normal_retina.png",
        }

        sample_choice = st.selectbox(
            "Select sample fundus image:",
            list(sample_options.keys()),
            index=0
        )

        target_file = sample_options[sample_choice]
        target_path = os.path.join(samples_dir, target_file)
        if os.path.exists(target_path):
            selected_image = Image.open(target_path)
            image_name = target_file
        else:
            st.error(f"Sample {target_file} not found.")

    else:
        uploaded_file = st.file_uploader(
            "Upload fundus eye scan (PNG/JPG):",
            type=["png", "jpg", "jpeg"]
        )
        if uploaded_file is not None:
            selected_image = Image.open(uploaded_file)
            image_name = uploaded_file.name

    st.markdown("---")

    # Classification Threshold
    st.subheader("2. Decision Threshold")
    threshold = st.slider(
        "DR Decision Cutoff:",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Scores at or above this threshold classify as Diabetic Retinopathy (DR). Default is 0.50."
    )

    st.markdown("---")

    # Model Checkpoint Management
    st.subheader("3. Model Checkpoints")
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    
    # Status badges
    vqc_status = "✅ Custom Trained (best.pkl)" if pipeline.is_custom_vqc_loaded else "🟡 Pre-calibrated Baseline"
    pca_status = "✅ Custom PCA (2048->4)" if pipeline.is_custom_pca_loaded else "🟡 Default Projection"
    resnet_status = "✅ Fine-tuned ResNet" if pipeline.is_custom_resnet_loaded else "ℹ️ ResNet50 (ImageNet)"

    st.markdown(f"- **VQC:** {vqc_status}")
    st.markdown(f"- **PCA:** {pca_status}")
    st.markdown(f"- **ResNet:** {resnet_status}")

    with st.expander("📂 Connect Google Drive Checkpoints"):
        st.write(
            "To use your exact trained checkpoints from Colab, copy them into the `models/` directory or upload them below:"
        )
        uploaded_vqc = st.file_uploader("Upload best.pkl:", type=["pkl"], key="vqc_uploader")
        if uploaded_vqc is not None:
            vqc_save_path = os.path.join(models_dir, "best.pkl")
            with open(vqc_save_path, "wb") as f:
                f.write(uploaded_vqc.getbuffer())
            st.success("Uploaded best.pkl! Reloading...")
            pipeline.reload_models()
            st.rerun()

        uploaded_pca = st.file_uploader("Upload pca_2048_to_4.joblib:", type=["joblib"], key="pca_uploader")
        if uploaded_pca is not None:
            pca_save_path = os.path.join(models_dir, "pca_2048_to_4.joblib")
            with open(pca_save_path, "wb") as f:
                f.write(uploaded_pca.getbuffer())
            st.success("Uploaded PCA! Reloading...")
            pipeline.reload_models()
            st.rerun()

    st.markdown("---")
    st.caption("🔬 Google Cirq Quantum Simulator | ResNet-50 | APTOS 2019")


# -----------------------------------------------------------------------------
# 5. MAIN PAGE CONTENT
# -----------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>QuantumNet: Diabetic Retinopathy Detection</h1>
    <p>Hybrid Classical-Quantum Architecture: Fine-Tuned ResNet-50 & 4-Qubit Variational Quantum Circuit (VQC) with Cirq</p>
</div>
""", unsafe_allow_html=True)

if not (pipeline.is_custom_vqc_loaded and pipeline.is_custom_pca_loaded and pipeline.is_custom_resnet_loaded):
    st.warning(
        "⚠️ **Running in Presentation Baseline Mode:** "
        "Your trained Google Colab checkpoint files (`best.pkl`, `pca_2048_to_4.joblib`, and `best_finetuned_resnet.weights.h5`) "
        "are not yet in the `models/` folder. Without your fine-tuned weights, the network runs on uncalibrated defaults. "
        "**To achieve your true 96.00% test accuracy on any scan, copy your Colab checkpoints into `models/`** (see **Tab 4** for 1-click Colab export code)."
    )

if selected_image is None:
    st.info("👈 Please select a sample retina or upload an eye fundus image in the sidebar to begin.")
    st.stop()

# Cached feature extraction and quantum simulation
@st.cache_data(show_spinner=False)
def analyze_image_cached(img_bytes, filename):
    import io
    img = Image.open(io.BytesIO(img_bytes))
    return pipeline.predict(img, threshold=0.50)

# Convert PIL image to bytes for caching
img_byte_arr = io.BytesIO()
selected_image.save(img_byte_arr, format="PNG")
img_bytes = img_byte_arr.getvalue()

# Run Prediction Pipeline with caching
with st.spinner("Analyzing Retinal Scan through Quantum Circuit..."):
    base_results = analyze_image_cached(img_bytes, image_name)

# Dynamic thresholding applied in < 0.1ms without re-running ResNet or Cirq
results = dict(base_results)
results["threshold"] = threshold
prob_dr = float(results["dr_probability"])
results["is_dr"] = prob_dr >= threshold
results["diagnosis"] = "Diabetic Retinopathy (DR)" if results["is_dr"] else "No Diabetic Retinopathy (Non-DR)"
results["predicted_class"] = 1 if results["is_dr"] else 0

if prob_dr < 0.20:
    results["risk_tier"] = "Normal / Low Risk"
    results["risk_color"] = "#28a745"
    results["clinical_advice"] = "No signs of Diabetic Retinopathy detected. Recommend standard routine annual retinal screening."
elif prob_dr < threshold:
    results["risk_tier"] = "Borderline / Mild Observation"
    results["risk_color"] = "#ffc107"
    results["clinical_advice"] = "Mild vascular irregularities noted. Re-evaluation in 6 months or follow-up retinal exam suggested."
elif prob_dr < 0.75:
    results["risk_tier"] = "Moderate Diabetic Retinopathy"
    results["risk_color"] = "#fd7e14"
    results["clinical_advice"] = "Notable lesions/microaneurysms detected. Comprehensive ophthalmologist evaluation advised within 4-6 weeks."
else:
    results["risk_tier"] = "Severe Diabetic Retinopathy"
    results["risk_color"] = "#dc3545"
    results["clinical_advice"] = "High probability of severe DR / macular edema indicators. Prompt clinical referral to a retinal specialist recommended."

# -----------------------------------------------------------------------------
# 6. TOP SECTION: IMAGE PREVIEW & DIAGNOSIS VERDICT
# -----------------------------------------------------------------------------
col_img, col_verdict = st.columns([1, 1.2], gap="large")

with col_img:
    st.subheader("📷 Retinal Fundus Scan")
    st.image(
        selected_image,
        caption=f"Input: {image_name} ({selected_image.size[0]}x{selected_image.size[1]} px)",
        use_container_width=True
    )

    with st.expander("🔍 Preprocessing Preview (224x224 RGB)"):
        st.image(
            results["resized_image"],
            caption="ResNet-50 Model Input (Normalized 224x224)",
            width=224
        )

with col_verdict:
    st.subheader("🩺 Diagnostic Assessment")

    # Primary Diagnosis Card
    if results["is_dr"]:
        st.markdown(f"""
        <div class="badge-dr">
            <h2 style="margin:0; font-size: 1.8rem;">🚨 DIABETIC RETINOPATHY (DR)</h2>
            <p style="margin: 6px 0 0 0; font-size: 1.1rem; font-weight: 600;">
                Positive for Retinopathy Indicators
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="badge-nondr">
            <h2 style="margin:0; font-size: 1.8rem;">✅ NO RETINOPATHY DETECTED (NON-DR)</h2>
            <p style="margin: 6px 0 0 0; font-size: 1.1rem; font-weight: 600;">
                Retinal Fundus Within Normal Limits
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Score Metrics Row
    m_col1, m_col2, m_col3 = st.columns(3)

    with m_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">DR Score</div>
            <div class="metric-value" style="color: {'#dc2626' if results['is_dr'] else '#475569'};">
                {results['dr_score_percent']:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

    with m_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Non-DR Score</div>
            <div class="metric-value" style="color: {'#16a34a' if not results['is_dr'] else '#475569'};">
                {results['non_dr_score_percent']:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

    with m_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Risk Tier</div>
            <div class="metric-value" style="font-size: 1.15rem; margin-top: 8px; color: {results['risk_color']};">
                {results['risk_tier']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Probability Progress Bar
    st.write("")
    st.write(f"**Diabetic Retinopathy Probability Score:** `{results['dr_probability']:.4f}` / 1.0000")
    st.progress(min(max(float(results["dr_probability"]), 0.0), 1.0))

    # Clinical Advice Box
    st.markdown(f"""
    <div class="advice-card">
        <strong>📋 Clinical Guidance:</strong><br>
        {results['clinical_advice']}
    </div>
    """, unsafe_allow_html=True)

    # Performance & Latency Telemetry
    timings = results.get("timings", {})
    if timings:
        st.write("")
        st.caption(
            f"⚡ **Inference Latency:** Total: `{timings.get('t_total', 0):.2f}s` | "
            f"ResNet-50: `{timings.get('t_resnet', 0)*1000:.0f}ms` | "
            f"Cirq VQC: `{timings.get('t_vqc', 0)*1000:.1f}ms` | "
            f"PCA: `{timings.get('t_pca', 0)*1000:.1f}ms` "
            f"(⚡ *Cached for instant updates*)"
        )


# -----------------------------------------------------------------------------
# 7. DEEP DIVE TABS
# -----------------------------------------------------------------------------
st.write("---")
tab1, tab2, tab3, tab4 = st.tabs([
    "⚛️ Quantum VQC & Readout",
    "🧠 ResNet-50 & PCA Pipeline",
    "📈 Paper Benchmark & 96% Metrics",
    "📖 How to Connect Checkpoints"
])

# -----------------------------------------------------------------------------
# TAB 1: QUANTUM VQC & READOUT
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("⚛️ 4-Qubit Variational Quantum Circuit (Google Cirq)")
    st.markdown(r"""
    The 4-dimensional normalized vector $x \in [-1, 1]^4$ is encoded into a 4-qubit quantum state via $R_y(x_k \cdot \pi/2)$.
    The state is processed through **6 variational layers** (24 trainable rotation angles $\theta$ and CNOT entanglement chains)
    and evaluated via **Pauli-Z expectation values** $\langle Z_q \rangle$.
    """)

    q_col1, q_col2 = st.columns([1, 1])

    with q_col1:
        st.write("##### 1. Pauli-Z Expectation Values $\\langle Z_q \\rangle$")
        z_vals = results["z_expectations"]

        z_df = pd.DataFrame({
            "Qubit": [f"Qubit {i} (q{i})" for i in range(4)],
            "Z Expectation": z_vals
        })
        st.bar_chart(z_df.set_index("Qubit"), height=260)

        st.markdown(f"""
        - **$Z_0$ (Qubit 0):** `{z_vals[0]:+.4f}`
        - **$Z_1$ (Qubit 1):** `{z_vals[1]:+.4f}`
        - **$Z_2$ (Qubit 2):** `{z_vals[2]:+.4f}`
        - **$Z_3$ (Qubit 3):** `{z_vals[3]:+.4f}`
        """)

    with q_col2:
        st.write("##### 2. Parameterized Readout & Sigmoid")
        rw = results["readout_w"]
        rb = results["readout_b"]
        logit = results["logit"]
        prob = results["dr_probability"]

        st.latex(r"\text{logit} = \sum_{q=0}^{3} w_q \langle Z_q \rangle + b")
        st.write(
            f"$$\\text{{logit}} = ({rw[0]:.2f})({z_vals[0]:.2f}) + "
            f"({rw[1]:.2f})({z_vals[1]:.2f}) + "
            f"({rw[2]:.2f})({z_vals[2]:.2f}) + "
            f"({rw[3]:.2f})({z_vals[3]:.2f}) + ({rb:.2f}) = {logit:.4f}$$"
        )
        st.latex(r"P(\text{DR}) = \sigma(\text{logit}) = \frac{1}{1 + e^{-\text{logit}}}")
        st.write(f"$$P(\\text{{DR}}) = \\sigma({logit:.4f}) = \\mathbf{{{prob:.4f}}} \\; ({prob*100:.2f}\\%)$$")

    with st.expander("📜 View Cirq Quantum Circuit Diagram"):
        st.markdown(f'<div class="circuit-box">{results["circuit_text"]}</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# TAB 2: RESNET-50 & PCA PIPELINE
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("🧠 Classical Backbone: ResNet-50 + PCA Reduction")

    p_col1, p_col2 = st.columns([1, 1])

    with p_col1:
        st.write("##### 1. ResNet-50 Feature Extraction (2048-D)")
        st.write("Features extracted from the `global_average_pooling` layer:")
        f2048 = results["features_2048"]
        st.markdown(f"""
        - **Feature Vector Shape:** `(2048,)`
        - **Max Activation:** `{np.max(f2048):.4f}`
        - **Min Activation:** `{np.min(f2048):.4f}`
        - **Mean Activation:** `{np.mean(f2048):.4f}`
        - **Non-zero Features:** `{np.count_nonzero(f2048)} / 2048`
        """)

    with p_col2:
        st.write("##### 2. PCA 2048-D $\\rightarrow$ 4-D Projection")
        pca_vals = results["pca_4d"]
        norm_vals = results["x_norm"]
        angles = results["encoding_angles"]

        pca_df = pd.DataFrame({
            "Component": ["PC1", "PC2", "PC3", "PC4"],
            "PCA Raw Value": pca_vals,
            "Normalized [-1, 1]": norm_vals,
            "Ry Angle (rad)": angles
        })
        st.dataframe(pca_df, use_container_width=True)

        st.bar_chart(pca_df.set_index("Component")["Normalized [-1, 1]"], height=200)


# -----------------------------------------------------------------------------
# TAB 3: BENCHMARK & METRICS
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("📈 QuantumNet Experimental Results vs Research Paper")
    st.markdown("""
    Results achieved on the unseen **APTOS 2019 Test Set (550 samples)** using the fine-tuned ResNet-50 + 4-Qubit VQC model:
    """)

    b_col1, b_col2 = st.columns([1.2, 1])

    with b_col1:
        metrics_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
            "QuantumNet (Our Model)": ["96.00%", "96.73%", "95.34%", "96.03%"],
            "Research Paper Baseline": ["94.11%", "96.90%", "94.84%", "95.86%"],
            "Delta (Improvement)": ["+1.89%", "-0.17%", "+0.50%", "+0.17%"]
        })
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        st.markdown("""
        > **Key Takeaway:** Our trained QuantumNet achieved **96.00% test accuracy**, outperforming the research paper baseline by **+1.89 percentage points** on the APTOS 2019 dataset.
        """)

    with b_col2:
        st.write("##### Confusion Matrix (550 Test Samples)")
        cm_data = np.array([
            [262, 9],    # True Non-DR: 262 TN, 9 FP
            [13, 266]    # True DR: 13 FN, 266 TP
        ])

        fig, ax = plt.subplots(figsize=(4, 3.5))
        cax = ax.matshow(cm_data, cmap="Blues")
        plt.title("Confusion Matrix", fontsize=11, pad=12)
        fig.colorbar(cax)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Non-DR", "DR"])
        ax.set_yticklabels(["Non-DR", "DR"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        for i in range(2):
            for j in range(2):
                color = "white" if cm_data[i, j] > 150 else "black"
                ax.text(j, i, str(cm_data[i, j]), ha="center", va="center", color=color, fontweight="bold")

        plt.tight_layout()
        st.pyplot(fig)


# -----------------------------------------------------------------------------
# TAB 4: HOW TO CONNECT CHECKPOINTS
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("📖 Checkpoint Setup & Directory Structure")
    st.markdown("""
    To link your exact Google Colab trained model files:
    
    1. **Download the following files from your Google Drive `QuantumNet/` folder**:
       - `quantum_finetuned_resnet/training/checkpoints/best.pkl`
       - `pca_finetuned_resnet/pca_2048_to_4.joblib`
       - `pca_finetuned_resnet/pca_min.npy`
       - `pca_finetuned_resnet/pca_max.npy`
       - `resnet_finetune/checkpoints/best_finetuned_resnet.weights.h5` (or `resnet50_best.keras`)
    
    2. **Place them into the `models/` directory:**
    ```text
    quantumnet_retinopathy_app/
    ├── app.py
    ├── pipeline.py
    ├── quantum_model.py
    ├── sample_generator.py
    ├── models/
    │   ├── best.pkl
    │   ├── pca_2048_to_4.joblib
    │   ├── pca_min.npy
    │   ├── pca_max.npy
    │   └── best_finetuned_resnet.weights.h5
    └── samples/
        ├── sample_normal_retina.png
        └── sample_dr_retina.png
    ```
    
    3. **⚡ 1-Click Export Code to run in Google Colab:**
       Run this snippet in a Colab cell to package and download your model files automatically:
    ```python
    import os, shutil
    from google.colab import files

    zip_dir = "/content/drive/MyDrive/QuantumNet"
    out_zip = "/content/QuantumNet_Checkpoints"
    
    # Create zip containing all necessary checkpoints
    shutil.make_archive(out_zip, 'zip', zip_dir)
    print("Download will start automatically...")
    files.download(out_zip + ".zip")
    ```
    
    4. Unzip `QuantumNet_Checkpoints.zip` and copy the `.pkl`, `.joblib`, and `.weights.h5` files directly into `models/`!
    """)
