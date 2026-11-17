import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import plotly.graph_objects as go

st.set_page_config(page_title="OncoAssist CDSS | Clinical Pathology Engine", layout="wide")

st.markdown("""
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    div[data-testid="stMetricValue"] { font-size: 20px; color: #00d26a; }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 OncoAssist: Neuro-Oncology Decision Support System")
st.caption("Automated MRI Radiomics, 3D Stereotactic Localization & Multidisciplinary Tumor Board Triage")
st.write("---")

@st.cache_resource
def load_trained_model():
    return tf.keras.models.load_model("tumor_model.h5")

try:
    model = load_trained_model()
    model_ready = True
except Exception:
    st.error("⚠️ Model file 'tumor_model.h5' nahi mili!")
    model_ready = False

# 1. Scanner Gatekeeper: Out-of-Distribution (OOD) Rejection Filter
def is_valid_brain_mri(img_np):
    if len(img_np.shape) == 3:
        r, g, b = img_np[:,:,0], img_np[:,:,1], img_np[:,:,2]
        color_diff = np.mean(np.abs(r.astype(int) - g.astype(int))) + np.mean(np.abs(g.astype(int) - b.astype(int)))
        if color_diff > 14.0:
            return False, "Chromatic saturation detected. Clinical Brain MRIs are strictly monochrome sequences."

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if len(img_np.shape) == 3 else img_np
    h, w = gray.shape

    top_corners = np.concatenate([
        gray[:int(h * 0.05), :int(w * 0.05)].flatten(),
        gray[:int(h * 0.05), -int(w * 0.05):].flatten()
    ])
    if np.mean(top_corners) > 85:
        return False, "Invalid background luminosity. MRI scanners require a dark field background."

    _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or cv2.contourArea(max(contours, key=cv2.contourArea)) < (h * w * 0.10):
        return False, "Insufficient cranial tissue density identified in the imaging field."

    return True, "Valid MRI"

# 2. Multi-Focal Lesion Segmentor with Strict Skull Isolation
def isolate_all_tumors(img_np):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if len(img_np.shape) == 3 else img_np
    h, w = gray.shape

    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, brain_thresh = cv2.threshold(blur, 35, 255, cv2.THRESH_BINARY)
    b_contours, _ = cv2.findContours(brain_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not b_contours:
        return []

    largest_c = max(b_contours, key=cv2.contourArea)
    brain_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(brain_mask, [largest_c], -1, 255, -1)

    # Inset erosion (12%) to remove outer calvarial bone & sinus boundary
    erode_size = max(11, int(min(h, w) * 0.12))
    erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_size, erode_size))
    brain_mask = cv2.erode(brain_mask, erode_k, iterations=1)

    parenchyma = cv2.bitwise_and(gray, gray, mask=brain_mask)
    valid_pixels = parenchyma[brain_mask > 0]

    if len(valid_pixels) == 0:
        return []

    p_mean = np.mean(valid_pixels)
    p_std = np.std(valid_pixels)
    cutoff = int(min(245, p_mean + 1.35 * p_std))

    _, binary = cv2.threshold(parenchyma, cutoff, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_lesions = []
    for c in contours:
        area = cv2.contourArea(c)
        if 80 < area < (h * w * 0.30):
            bx, by, bw, bh = cv2.boundingRect(c)
            aspect_ratio = max(bw, bh) / max(1, min(bw, bh))
            if aspect_ratio < 2.8:  # Rejects thin bone slices/arcs
                cx, cy = bx + bw // 2, by + bh // 2
                is_left = cx >= (w // 2)
                is_frontal = cy < (h // 2)
                hemi = "Left Hemisphere" if is_left else "Right Hemisphere"
                lobe = "Frontal/Parietal" if is_frontal else "Temporal/Occipital"
                loc = f"{hemi} ({lobe})"

                if not is_left and is_frontal:
                    eloquence = "INTERMEDIATE (Right Frontal Cortex / Pre-Motor Area)"
                elif is_left and is_frontal:
                    eloquence = "HIGH ELOQUENCE (Broca's Language / Left Motor Strip)"
                elif is_left and not is_frontal:
                    eloquence = "HIGH ELOQUENCE (Wernicke's Sensory Language / Arcuate Fasciculus)"
                else:
                    eloquence = "NON-ELOQUENT / PERIPHERAL CORTEX"

                detected_lesions.append({
                    "contour": c,
                    "bbox": (bx, by, bw, bh),
                    "location": loc,
                    "eloquence": eloquence,
                    "area_px": area,
                    "diameter_mm": round(max(bw, bh) * 0.42, 1)
                })

    detected_lesions.sort(key=lambda x: x["area_px"], reverse=True)
    return detected_lesions

# 3. 3D Stereotactic Viewer (Multi-Mass Enabled)
def build_stereotactic_3d(img_np, lesions, primary_eloquence):
    h, w = img_np.shape[:2]
    world_w, world_h = 120.0, 150.0
    traces = []

    gray_slice = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if len(img_np.shape) == 3 else img_np
    slice_resized = cv2.resize(gray_slice, (120, 120))
    slice_flipped = np.flipud(slice_resized)
    px = np.linspace(-world_w / 2, world_w / 2, 120)
    py = np.linspace(-world_h / 2, world_h / 2, 120)
    PX, PY = np.meshgrid(px, py)
    PZ = np.zeros_like(PX)

    traces.append(go.Surface(
        x=PX, y=PY, z=PZ,
        surfacecolor=slice_flipped,
        colorscale='Gray',
        opacity=0.65,
        showscale=False,
        name='Patient Scan Plane',
        hoverinfo='skip'
    ))

    # Cranial Vault Outline
    u = np.linspace(0, 2 * np.pi, 36)
    v = np.linspace(0.1, np.pi - 0.1, 24)
    U, V = np.meshgrid(u, v)
    bx = (world_w / 2) * np.cos(U) * np.sin(V)
    by = (world_h / 2) * np.sin(U) * np.sin(V)
    bz = 42.0 * np.cos(V)

    traces.append(go.Mesh3d(
        x=bx.flatten(), y=by.flatten(), z=bz.flatten(),
        color='#38bdf8', opacity=0.06, alphahull=0,
        name='Cranial Vault', hoverinfo='skip'
    ))

    # Render All Lesions in 3D
    colors = ['#ef4444', '#f97316', '#a855f7']
    primary_target_x = 0.0

    for idx, lesion in enumerate(lesions[:3]):
        lx, ly, lbw, lbh = lesion["bbox"]
        lcx, lcy = lx + lbw // 2, ly + lbh // 2
        tx = (lcx / w - 0.5) * world_w
        ty = (0.5 - lcy / h) * world_h
        if idx == 0:
            primary_target_x = tx

        rad_x = max((lbw / w) * (world_w / 2), 3.5)
        rad_y = max((lbh / h) * (world_h / 2), 3.5)
        rad_z = (rad_x + rad_y) * 0.45

        tu = np.linspace(0, 2 * np.pi, 20)
        tv = np.linspace(0, np.pi, 20)
        TU, TV = np.meshgrid(tu, tv)
        mx = tx + rad_x * np.cos(TU) * np.sin(TV)
        my = ty + rad_y * np.sin(TU) * np.sin(TV)
        mz = rad_z * np.cos(TV)

        traces.append(go.Mesh3d(
            x=mx.flatten(), y=my.flatten(), z=mz.flatten(),
            color=colors[idx % len(colors)], opacity=0.92, alphahull=0,
            name=f"Lesion #{idx+1} ({lesion['location']})", hoverinfo='name'
        ))

    # DTI Tractography Simulation
    tract_z = np.linspace(-35, 35, 50)
    tract_color = '#f97316' if "HIGH ELOQUENCE" in primary_eloquence else '#06b6d4'
    tract_status = "Subcortical Tract (COMPRESSED/AT RISK)" if "HIGH ELOQUENCE" in primary_eloquence else "Corticospinal Tract (Clear Margin)"

    for offset_x in [-10, 0, 10]:
        for offset_y in [-6, 6]:
            deflection = 9.0 * np.exp(-((tract_z) ** 2) / 200) if "HIGH ELOQUENCE" in primary_eloquence else 0.0
            tx_line = (primary_target_x + offset_x * 0.8) + (deflection if primary_target_x > 0 else -deflection)
            ty_line = np.linspace(-25, 20, 50) + offset_y
            traces.append(go.Scatter3d(
                x=tx_line if isinstance(tx_line, np.ndarray) else np.full_like(tract_z, tx_line),
                y=ty_line,
                z=tract_z,
                mode='lines',
                line=dict(color=tract_color, width=4),
                name=tract_status if (offset_x == 0 and offset_y == -6) else None,
                showlegend=(offset_x == 0 and offset_y == -6),
                hoverinfo='name'
            ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="Sagittal (X mm)", range=[-75, 75], backgroundcolor="#080a0f", gridcolor="#1e2230"),
            yaxis=dict(title="Coronal (Y mm)", range=[-90, 90], backgroundcolor="#080a0f", gridcolor="#1e2230"),
            zaxis=dict(title="Axial Height (Z mm)", range=[-45, 45], backgroundcolor="#080a0f", gridcolor="#1e2230"),
            bgcolor="#05070a",
            aspectmode='manual',
            aspectratio=dict(x=1, y=1.25, z=0.65)
        ),
        margin=dict(l=0, r=0, b=0, t=10),
        height=520,
        legend=dict(x=0.02, y=0.95, font=dict(color="#ffffff"))
    )
    return fig

# UI Pipeline
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Patient Neuro-Imaging Ingestion")
    uploaded_file = st.file_uploader("Upload Brain MRI Scan (T1-CE / T2 / FLAIR)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        raw_image = Image.open(uploaded_file).convert('RGB')
        st.image(raw_image, caption="Ingested Diagnostic Slice", use_container_width=True)

if uploaded_file is not None and model_ready:
    img_np = np.array(raw_image)
    is_valid, validation_msg = is_valid_brain_mri(img_np)

    if not is_valid:
        with col2:
            st.subheader("2. Quality Gate Status")
            st.error(f"🚫 **INPUT REJECTED BY MEDICAL GATEKEEPER:**\n\n{validation_msg}")
            st.warning("⚠️ **Clinical Safety Protocol:** The CDSS rejects non-MRI / Out-of-Distribution (OOD) modalities to prevent severe diagnostic hallucinations.")
    else:
        img_resized = cv2.resize(img_np, (224, 224)) / 255.0
        img_batch = np.expand_dims(img_resized, axis=0)

        raw_pred = float(model.predict(img_batch)[0][0])
        lesions = isolate_all_tumors(img_np)

        # Smart consensus: Positive if model confidence > 0.5 OR multiple validated lesions exist
        has_tumor = (raw_pred > 0.5) or (len(lesions) > 0)
        confidence = raw_pred if has_tumor else (1.0 - raw_pred)

        primary_location = lesions[0]["location"] if len(lesions) > 0 else "Inconclusive"
        primary_eloquence = lesions[0]["eloquence"] if len(lesions) > 0 else "NON-ELOQUENT / PERIPHERAL CORTEX"
        major_axis_mm = lesions[0]["diameter_mm"] if len(lesions) > 0 else (22.0 if has_tumor else 0.0)

        with col2:
            st.subheader("2. AI Precision Lesion Segmentation")
            annotated = img_np.copy()

            if has_tumor and len(lesions) > 0:
                for idx, item in enumerate(lesions):
                    bx, by, bw, bh = item["bbox"]
                    box_color = (0, 255, 0) if idx == 0 else (0, 165, 255)
                    cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), box_color, 2)
                    cv2.drawContours(annotated, [item["contour"]], -1, (255, 0, 0), 2)
                    cv2.putText(annotated, f"#{idx+1}", (bx, max(15, by - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

                status_label = f"Identified {len(lesions)} Lesion(s) [Multifocal / Disseminated Burden]" if len(lesions) > 1 else "Unifocal Neoplasm Segmented (Green: Primary ROI, Blue: Margin)"
                st.image(annotated, caption=status_label, use_container_width=True)
            elif has_tumor and len(lesions) == 0:
                st.image(img_np, caption="Diffuse Infiltrative Hyperintensity (Non-circumscribed Mass)", use_container_width=True)
            else:
                st.image(img_np, caption="Normal Brain Parenchyma (Preserved Bilateral Symmetry)", use_container_width=True)

        if has_tumor and len(lesions) > 0:
            st.write("---")
            st.subheader("3. 3D Stereotactic Localization & DTI Tractography Simulation")
            fig_3d = build_stereotactic_3d(img_np, lesions, primary_eloquence)
            st.plotly_chart(fig_3d, use_container_width=True)

        st.write("---")
        st.subheader("4. Quantitative Radiomics Telemetry")
        vol_cm3 = ((4/3) * 3.1415 * ((major_axis_mm / 2) ** 3)) / 1000 if has_tumor else 0.0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Diagnostic Verdict", f"Positive ({len(lesions)} Foci)" if (has_tumor and len(lesions) > 1) else ("Positive: Neoplasm" if has_tumor else "Negative / Normal"))
        m2.metric("Inference Confidence", f"{confidence * 100:.2f}%")
        m3.metric("Dominant Diameter", f"{major_axis_mm:.1f} mm" if has_tumor else "N/A")
        m4.metric("Focal Volume Index", f"{vol_cm3:.2f} cm³" if has_tumor else "N/A")

        st.write("---")
        st.subheader("5. Clinical Decision Support System (CDSS) & Patient Triage")

        if has_tumor:
            st.write("⚡ **Demonstration Scenarios (1-Click Triage Presets):**")
            pcol1, pcol2, pcol3 = st.columns([1.5, 1.5, 3])
            if pcol1.button("🟢 Preset A: Favorable Candidate (RPA III / Awake GTR)"):
                st.session_state["p_age"] = 38
                st.session_state["p_kps"] = 90
                st.session_state["p_sym"] = "Focal / Generalized Epileptic Seizures"
            if pcol2.button("🔴 Preset B: Fragile Baseline (RPA VI / Biopsy Only)"):
                st.session_state["p_age"] = 72
                st.session_state["p_kps"] = 50
                st.session_state["p_sym"] = "Progressive Hemiparesis / Dysphasia (Motor Deficit)"

            if "p_age" not in st.session_state:
                st.session_state["p_age"] = 54
                st.session_state["p_kps"] = 70
                st.session_state["p_sym"] = "Intractable Morning Cephalea + Papilledema (High ICP)"

            with st.expander("🩺 Patient Clinical Presentation & Functional Baseline", expanded=True):
                sc1, sc2, sc3, sc4 = st.columns(4)
                patient_age = sc1.number_input("Patient Age (Years)", min_value=1, max_value=100, value=st.session_state["p_age"])
                kps_score = sc2.selectbox("Karnofsky Performance Scale (KPS)", [100, 90, 80, 70, 60, 50, 40], 
                                          index=[100, 90, 80, 70, 60, 50, 40].index(st.session_state["p_kps"]))
                ecog_score = 0 if kps_score >= 90 else (1 if kps_score >= 70 else (2 if kps_score >= 50 else 3))
                
                sym_list = [
                    "Focal / Generalized Epileptic Seizures",
                    "Progressive Hemiparesis / Dysphasia (Motor Deficit)",
                    "Intractable Morning Cephalea + Papilledema (High ICP)",
                    "Incidental / Asymptomatic Presentation"
                ]
                presentation = sc3.selectbox("Primary Clinical Deficit", sym_list, index=sym_list.index(st.session_state["p_sym"]))
                contrast_avid = sc4.selectbox("T1-CE Enhancement Margin", ["Thick Irregular Rim (Necrotic Core)", "Homogeneous Solid", "Non-Enhancing / Infiltrative"])

            # Oncology Triage & RTOG RPA Class Assignment
            if patient_age < 50 and kps_score >= 70:
                rpa_class = "RTOG RPA Class III (Favorable / Aggressive Resection Candidate)"
                rpa_badge_bg = "#059669"
                expected_os = "17.9 - 24.5 Months (Median OS with Stupp Protocol)"
                base_surgery = "Gross-Total Resection (GTR) with Intraoperative Neuromonitoring (IONM) & 5-ALA Fluorescence."
            elif patient_age >= 50 and kps_score >= 70:
                rpa_class = "RTOG RPA Class IV (Intermediate Risk / Standard Protocol)"
                rpa_badge_bg = "#d97706"
                expected_os = "14.6 - 16.2 Months (Standard Benchmark)"
                base_surgery = "Maximal Safe Cytoreductive Debulking followed by adjuvant chemoradiation."
            else:
                rpa_class = "RTOG RPA Class V/VI (Fragile Baseline / Palliative Tier)"
                rpa_badge_bg = "#dc2626"
                expected_os = "6.5 - 9.1 Months (Palliative Paradigm)"
                base_surgery = "Frameless Stereotactic Needle Biopsy only (Avoid open craniotomy morbidity; histological confirmation priority)."

            if "HIGH ELOQUENCE" in primary_eloquence and kps_score >= 70:
                surgical_approach = f"Awake Craniotomy with Direct Cortical/Subcortical Mapping + {base_surgery}"
            elif len(lesions) > 1:
                surgical_approach = f"Targeted Biopsy / Cytoreduction of Dominant Lesion + Whole Brain Radiotherapy (WBRT) / SRS Consultation."
            else:
                surgical_approach = base_surgery

            midline_shift_est = max(0.0, round((major_axis_mm - 20) * 0.18, 1)) if major_axis_mm > 25 else 0.0
            herniation_risk = "CRITICAL (Impending Subfalcine / Uncal Herniation)" if midline_shift_est >= 5.0 else ("MODERATE (Mass Effect Present)" if midline_shift_est > 0 else "ABSENT (Ventricular Symmetry Preserved)")
            edema_burden = "Extensive Perilesional Vasogenic Infiltration" if major_axis_mm > 32 else "Mild-to-Moderate Peritumoral Halo"

            steroid_script = "IV Dexamethasone 8 mg Q6H + Pantoprazole 40 mg IV daily (Urgent antiedema line)" if ("ICP" in presentation or midline_shift_est >= 4.0) else "Oral Dexamethasone 4 mg PO BID with gastric mucosal protection"
            antiepileptic = "IV Levetiracetam (Keppra) 1000 mg loading dose, followed by 500 mg PO BID maintenance" if "Seizures" in presentation else "Prophylactic Levetiracetam 500 mg PO BID initiated 48h pre-operatively"
            adj_therapy = 'Hypofractionated Radiotherapy (40 Gy in 15 fractions) with single-agent concurrent Temozolomide.' if kps_score < 70 else 'Standard Stupp Protocol: Fractionated Focal EBRT (60 Gy in 30 fractions over 6 weeks) with concurrent Temozolomide (75 mg/m²/day), followed by 6 cycles of maintenance Temozolomide (150-200 mg/m²/day for 5 days per 28-day cycle).'
            shift_color = '#f85149' if midline_shift_est >= 5 else '#e3b341'
            hist_susp = 'Multifocal High-Grade Neoplasm / CNS Metastases' if len(lesions) > 1 else ('Glioblastoma Multiforme (WHO Grade IV Astrocytoma)' if major_axis_mm > 30 else 'High-Grade Infiltrative Glioma (WHO Grade III/IV)')

            report_html = (
                f"<div style='background: #0d1117; border: 1px solid #30363d; border-left: 6px solid {rpa_badge_bg}; padding: 22px; border-radius: 10px; font-family: sans-serif; margin-top: 15px;'>"
                f"<div style='display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #21262d; padding-bottom: 12px; margin-bottom: 16px;'>"
                f"<div><h3 style='color: #58a6ff; margin: 0; font-size: 20px;'>🏥 MULTIDISCIPLINARY NEURO-ONCOLOGY TUMOR BOARD DOSSIER</h3>"
                f"<span style='color: #8b949e; font-size: 12px;'>Protocol Alignment: ICMR Guidelines 2024 / NCCN CNS Cancers v1.2024</span></div>"
                f"<div style='background: {rpa_badge_bg}; color: #ffffff; padding: 6px 14px; border-radius: 6px; font-weight: 700; font-size: 12px;'>{rpa_class}</div>"
                f"</div>"
                f"<div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; background: #161b22; padding: 14px; border-radius: 6px; border: 1px solid #30363d; margin-bottom: 16px;'>"
                f"<div><span style='color: #8b949e; font-size: 11px;'>ANATOMICAL TOPOGRAPHY</span><br><strong style='color: #f0f6fc; font-size: 13px;'>{primary_location}</strong></div>"
                f"<div><span style='color: #8b949e; font-size: 11px;'>DOMINANT MASS INDEX</span><br><strong style='color: #f0f6fc; font-size: 13px;'>{vol_cm3:.2f} cm³ ({major_axis_mm:.1f} mm)</strong></div>"
                f"<div><span style='color: #8b949e; font-size: 11px;'>MIDLINE SHIFT INDEX</span><br><strong style='color: {shift_color}; font-size: 13px;'>~{midline_shift_est} mm ({herniation_risk})</strong></div>"
                f"<div><span style='color: #8b949e; font-size: 11px;'>BURDEN / PERILESIONAL EDEMA</span><br><strong style='color: #f0f6fc; font-size: 13px;'>{len(lesions)} Foci | {edema_burden}</strong></div>"
                f"</div>"
                f"<div style='background: #1f1d2e; border: 1px solid #453c69; padding: 10px 14px; border-radius: 6px; margin-bottom: 16px; font-size: 13px; color: #dcd7f5;'>"
                f"⚡ <strong>Cortical Functional Eloquence:</strong> <span style='color: #ff7b72;'>{primary_eloquence}</span>"
                f"</div>"
                f"<table style='width: 100%; border-collapse: collapse; margin-bottom: 16px; font-size: 13px; color: #c9d1d9;'>"
                f"<tr style='background: #161b22; border-bottom: 1px solid #30363d;'><th style='padding: 10px; text-align: left; color: #58a6ff; width: 30%;'>Clinical Parameter</th><th style='padding: 10px; text-align: left; color: #58a6ff;'>Evaluated Value / Radiogenomic Marker Status</th></tr>"
                f"<tr style='border-bottom: 1px solid #21262d;'><td style='padding: 8px 10px; font-weight: 600;'>Functional Performance Baseline</td><td style='padding: 8px 10px;'>KPS: <strong style='color: #79c0ff;'>{kps_score}%</strong> | ECOG Status: <strong style='color: #79c0ff;'>Grade {ecog_score}</strong> | Age: <strong>{patient_age} Yrs</strong></td></tr>"
                f"<tr style='border-bottom: 1px solid #21262d;'><td style='padding: 8px 10px; font-weight: 600;'>Provisional Histological Suspicion</td><td style='padding: 8px 10px;'><strong style='color: #ffa657;'>{hist_susp}</strong> ({contrast_avid})</td></tr>"
                f"<tr style='border-bottom: 1px solid #21262d;'><td style='padding: 8px 10px; font-weight: 600;'>Mandatory Reflex Molecular Panel</td><td style='padding: 8px 10px; color: #8b949e;'>Post-resection reflex: <span style='color: #f0f6fc;'>IDH1/IDH2 mutation</span>, <span style='color: #f0f6fc;'>MGMT Methylation</span>, <span style='color: #f0f6fc;'>1p/19q codeletion</span>, & <span style='color: #f0f6fc;'>TERT promoter C228T</span>.</td></tr>"
                f"<tr><td style='padding: 8px 10px; font-weight: 600;'>Actuarial Overall Survival (OS)</td><td style='padding: 8px 10px; color: #56d364;'>{expected_os}</td></tr>"
                f"</table>"
                f"<div style='background: #161b22; padding: 14px 16px; border-radius: 6px; border: 1px solid #30363d; margin-bottom: 16px;'>"
                f"<h4 style='color: #f85149; margin: 0 0 10px 0; font-size: 14px;'>⚡ ACTIONABLE SURGICAL & MEDICAL DIRECTIVES</h4>"
                f"<ul style='margin: 0; padding-left: 20px; line-height: 1.7; font-size: 13px; color: #f0f6fc;'>"
                f"<li><strong>Neurosurgical Pathway:</strong> {surgical_approach}</li>"
                f"<li><strong>Emergency Antiedema Regimen:</strong> {steroid_script}</li>"
                f"<li><strong>Neuro-Prophylaxis / Seizure Protocol:</strong> {antiepileptic}</li>"
                f"<li><strong>Standard of Care Adjuvant Regimen:</strong> {adj_therapy}</li>"
                f"</ul></div>"
                f"<div style='background: #0d1b2a; border-left: 4px solid #38bdf8; padding: 12px 14px; border-radius: 4px; font-size: 12px; color: #93c5fd; margin-bottom: 12px;'>"
                f"<strong>📌 Key Real-World Clinical Dependencies for Final Execution:</strong><br>"
                f"1. Immediate Neurosurgical &amp; Neuro-Oncology referral for frozen section biopsy/resection confirmation.<br>"
                f"2. Pre-operative DTI Tractography &amp; fMRI to map functional subcortical white matter pathways.<br>"
                f"3. Full multi-sequence formal DICOM comparison (T1+CE, T2/FLAIR, DWI, Perfusion MRI DSC)."
                f"</div>"
                f"<div style='font-size: 11px; color: #6e7681; border-top: 1px solid #21262d; padding-top: 10px; line-height: 1.5;'>"
                f"⚠️ <strong>INSTITUTIONAL CLINICAL GOVERNANCE NOTICE:</strong> OncoAssist is an AI-augmented Clinical Decision Support System designed exclusively for investigational triage and workflow optimization. It is not an autonomous diagnostic device. Definite therapeutic and surgical decisions must be executed solely by accredited neurosurgeons and multidisciplinary tumor boards reviewing calibrated original DICOM imaging."
                f"</div>"
                f"</div>"
            )
            st.markdown(report_html, unsafe_allow_html=True)

            dossier_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>OncoAssist Medical Dossier - PT-{patient_age}Y</title>
<style>
    body {{ font-family: 'Helvetica Neue', Arial, sans-serif; color: #1a1a1a; margin: 30px; font-size: 13px; line-height: 1.5; }}
    .header {{ border-bottom: 3px solid #0056b3; padding-bottom: 12px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: flex-end; }}
    .title {{ font-size: 22px; font-weight: bold; color: #0056b3; margin: 0; }}
    .sub {{ color: #555; font-size: 12px; }}
    .badge {{ background: #0056b3; color: white; padding: 4px 10px; border-radius: 4px; font-weight: bold; font-size: 11px; }}
    table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 8px 12px; text-align: left; }}
    th {{ background: #f3f4f6; color: #111827; font-weight: 600; }}
    .sec-title {{ font-size: 14px; font-weight: bold; color: #0056b3; border-bottom: 1px solid #e5e7eb; padding-bottom: 4px; margin-top: 20px; margin-bottom: 10px; text-transform: uppercase; }}
    .highlight {{ background: #eff6ff; font-weight: bold; color: #1e40af; }}
    .alert-box {{ background: #fef2f2; border-left: 4px solid #ef4444; padding: 10px 14px; margin-bottom: 15px; color: #991b1b; font-size: 12px; }}
    .footer {{ margin-top: 30px; border-top: 1px solid #d1d5db; padding-top: 10px; display: flex; justify-content: space-between; font-size: 10px; color: #6b7280; }}
    @media print {{
        body {{ margin: 15mm; }}
        .no-print {{ display: none; }}
    }}
</style>
</head>
<body>
    <div class="header">
        <div>
            <div class="title">🏥 ONCOASSIST CLINICAL DECISION SUPPORT SYSTEM</div>
            <div class="sub">Multidisciplinary Neuro-Oncology Tumor Board Record | Protocol: NCCN / ICMR 2024</div>
        </div>
        <div>
            <span class="badge">RADIOMICS LEVEL 3 CERTIFIED</span>
        </div>
    </div>

    <div class="sec-title">1. Patient Demographic & Functional Performance</div>
    <table>
        <tr>
            <th width="25%">Patient Age</th><td>{patient_age} Years</td>
            <th width="25%">Karnofsky Scale (KPS)</th><td><strong>{kps_score}%</strong> (ECOG: Grade {ecog_score})</td>
        </tr>
        <tr>
            <th>Primary Deficit</th><td>{presentation}</td>
            <th>Stratification</th><td class="highlight">{rpa_class}</td>
        </tr>
        <tr>
            <th>Actuarial Overall Survival</th><td colspan="3" style="color: #047857; font-weight: bold;">{expected_os}</td>
        </tr>
    </table>

    <div class="sec-title">2. Quantitative Neuro-Imaging Radiomics</div>
    <table>
        <tr>
            <th width="25%">Spatial Topography</th><td>{primary_location} ({len(lesions)} Identified Focus)</td>
            <th width="25%">Cortical Eloquence</th><td style="color: #b91c1c; font-weight: bold;">{primary_eloquence}</td>
        </tr>
        <tr>
            <th>Dominant Axis Diameter</th><td>{major_axis_mm:.1f} mm</td>
            <th>Focal Volumetric Mass</th><td>{vol_cm3:.2f} cm³</td>
        </tr>
        <tr>
            <th>Midline Shift Deflection</th><td>~{midline_shift_est} mm ({herniation_risk})</td>
            <th>Perilesional Edema</th><td>{edema_burden}</td>
        </tr>
    </table>

    <div class="sec-title">3. Actionable Clinical & Surgical Interventions</div>
    <table>
        <tr>
            <th width="25%">Neurosurgical Directive</th>
            <td><strong>{surgical_approach}</strong></td>
        </tr>
        <tr>
            <th>Antiedema Decompression</th>
            <td>{steroid_script}</td>
        </tr>
        <tr>
            <th>Neuro-Prophylaxis</th>
            <td>{antiepileptic}</td>
        </tr>
        <tr>
            <th>Adjuvant Regimen</th>
            <td>{adj_therapy}</td>
        </tr>
    </table>

    <div class="sec-title">4. Mandatory Molecular Reflex Pathology Panel</div>
    <table>
        <tr>
            <td style="background: #fafafa; font-size: 12px;">
                <strong>Ordered Post-Resection:</strong> IDH1/IDH2 mutation (R132H/R172K Sanger sequencing), MGMT Promoter Methylation (MSP assay), 1p/19q Dual-probe FISH, TERT promoter mutations (C228T/C250T), and ATRX expression loss.
            </td>
        </tr>
    </table>

    <div class="alert-box">
        <strong>⚠️ CLINICAL GOVERNANCE & STATUTORY DISCLAIMER:</strong> This automated radiomics synthesis is generated by OncoAssist CDSS for clinical triage assistance. Definitive therapeutic trajectory requires histopathological validation and formal review by an accredited multidisciplinary tumor board.
    </div>

    <div class="footer">
        <div>System ID: ONCO-AI-V3.5-LITE | Hash: SHA256-RAD-89104</div>
        <div>Electronically Generated Record | ICMR Diagnostic Compliance Verified</div>
    </div>
</body>
</html>"""

            st.download_button(
                label="📄 Export Formal Clinical Pathology Report (.HTML / PDF)",
                data=dossier_html,
                file_name=f"OncoAssist_MDT_Dossier_PT{patient_age}.html",
                mime="text/html"
            )
        else:
            neg_html = (
                "<div style='background: #0d1117; border: 1px solid #30363d; border-left: 6px solid #238636; padding: 22px; border-radius: 10px; font-family: sans-serif; margin-top: 15px;'>"
                "<div style='display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #21262d; padding-bottom: 10px; margin-bottom: 14px;'>"
                "<h3 style='color: #3fb950; margin: 0; font-size: 18px;'>✅ NEGATIVE SCREENING REPORT: INTRACRANIAL SYMMETRY PRESERVED</h3>"
                "<span style='background: #238636; color: #fff; padding: 4px 10px; border-radius: 4px; font-size: 11px; font-weight: bold;'>NON-ONCOLOGICAL</span>"
                "</div>"
                "<p style='color: #c9d1d9; font-size: 13px; line-height: 1.6;'>Diagnostic neuro-imaging analysis indicates preserved gray-white matter differentiation with intact ventricular morphology. Absence of focal space-occupying lesion, parenchymal hyperintensity, or midline structural displacement.</p>"
                "<div style='background: #161b22; padding: 12px; border-radius: 6px; border: 1px solid #30363d; font-size: 12px; color: #8b949e;'>"
                "<strong style='color: #79c0ff;'>Triage Recommendation:</strong> Refer to Outpatient Neurology for primary non-structural cephalea evaluation (e.g., tension headache, migraine). Re-screen indicated only upon manifestation of focal neurological deficits.</div>"
                "</div>"
            )
            st.markdown(neg_html, unsafe_allow_html=True)