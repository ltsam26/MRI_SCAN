import streamlit as st
import torch
from torchvision import models, transforms
import pydicom
from PIL import Image
import numpy as np
import cv2
import io
from fpdf import FPDF
from datetime import datetime
import tempfile
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------
# Device Configuration
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("resnet_mri.pth", map_location=device))
    model = model.to(device)
    model.eval()
    return model


model = load_model()

# -----------------------------
# Preprocessing Functions
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def preprocess_image(img):
    if len(np.array(img).shape) == 2:
        img = Image.fromarray(np.stack([np.array(img)] * 3, axis=-1))
    img_array = np.array(img)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        img_array = img_array[y:y + h, x:x + w]
    return Image.fromarray(img_array)


# -----------------------------
# Grad-CAM Implementation
# -----------------------------
def generate_gradcam(model, image_tensor, target_class):
    gradients, activations = [], []

    def forward_hook(module, input, output): activations.append(output)
    def backward_hook(module, grad_input, grad_output): gradients.append(grad_output[0])

    conv_layer = model.layer4[1].conv2
    conv_layer.register_forward_hook(forward_hook)
    conv_layer.register_backward_hook(backward_hook)
    output = model(image_tensor)
    model.zero_grad()
    output[0, target_class].backward()
    gradient = gradients[0].cpu().detach().numpy()[0]
    activation = activations[0].cpu().detach().numpy()[0]
    weights = np.mean(gradient, axis=(1, 2))
    grad_cam = np.zeros(activation.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        grad_cam += w * activation[i]
    grad_cam = np.maximum(grad_cam, 0)
    grad_cam = cv2.resize(grad_cam, (224, 224))
    grad_cam = (grad_cam - grad_cam.min()) / (grad_cam.max() - grad_cam.min() + 1e-8)
    return grad_cam


# -----------------------------
# Page Configuration & Custom CSS
# -----------------------------
st.set_page_config(page_title="Brain MRI Tumor Detector", page_icon="🧠", layout="wide")

st.markdown("""
    <style>
    body, .stApp {
        background: linear-gradient(135deg, #eaf3ff 0%, #fdfdff 100%);
        font-family: 'Segoe UI', sans-serif;
        color: #1c1c1c;
    }
    .main-header {
        text-align: center;
        font-size: 42px;
        font-weight: 800;
        color: #004aad;
        margin-bottom: 10px;
    }
    .sub-text {
        text-align: center;
        color: #555;
        font-size: 16px;
        margin-bottom: 20px;
    }
    .patient-card, .info-card {
        background-color: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0px 4px 16px rgba(0,0,0,0.08);
        margin-bottom: 20px;
        border-left: 5px solid #0072ff;
    }
    .stTabs [role="tab"] {
        background-color: #f3f6fb;
        border-radius: 8px;
        padding: 10px 25px;
        font-weight: 600;
        color: #004aad;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0072ff;
        color: white !important;
    }
    .stButton>button {
        background-color: #0072ff;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        transition: 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #005bb5;
    }
    .tumor {
        background-color: #e63946;
        color: white;
        padding: 10px;
        border-radius: 8px;
        font-weight: 600;
        text-align: center;
    }
    .no-tumor {
        background-color: #2ecc71;
        color: white;
        padding: 10px;
        border-radius: 8px;
        font-weight: 600;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown('<h1 class="main-header">🧠 Brain MRI Tumor Detection System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">AI-powered MRI classification with Grad-CAM visual explanations</p>', unsafe_allow_html=True)

# -----------------------------
# Session State Initialization
# -----------------------------
if "patient_submitted" not in st.session_state:
    st.session_state.patient_submitted = False
if "patient_metadata" not in st.session_state:
    st.session_state.patient_metadata = {}

# -----------------------------
# Sidebar: Patient Info & File Upload
# -----------------------------
with st.sidebar:
    st.header("🏥 Patient Information")
    with st.form("patient_form"):
        patient_name = st.text_input("Full Name")
        patient_age = st.text_input("Age")
        patient_gender = st.selectbox("Gender", ["", "Male", "Female", "Other"])
        study_date = st.date_input("MRI Study Date", datetime.today())
        submitted = st.form_submit_button("Save Info")

    if submitted:
        if patient_name and patient_age and patient_gender:
            st.session_state.patient_submitted = True
            st.session_state.patient_metadata = {
                "Patient Name": patient_name,
                "Age": patient_age,
                "Gender": patient_gender,
                "Study Date": study_date.strftime("%Y-%m-%d")
            }
            st.success("✅ Patient information saved. Proceed to upload MRI scans.")
        else:
            st.error("⚠️ Please fill in all required fields.")

    if st.session_state.patient_submitted:
        st.header("📤 Upload MRI Scans")
        uploaded_files = st.file_uploader(
            "Upload MRI/DICOM files",
            type=["dcm", "jpg", "jpeg", "png"],
            accept_multiple_files=True
        )


# -----------------------------
# Main Analysis
# -----------------------------
if st.session_state.patient_submitted and uploaded_files:
    st.markdown("<div class='sub-text'>Processing uploaded MRI images...</div>", unsafe_allow_html=True)
    progress_bar = st.progress(0)
    patient_results, slice_images = [], []

    for i, uploaded_file in enumerate(uploaded_files):
        progress_bar.progress((i + 1) / len(uploaded_files))

        if uploaded_file.name.endswith(".dcm"):
            dicom = pydicom.dcmread(io.BytesIO(uploaded_file.read()))
            img = dicom.pixel_array
            img = cv2.convertScaleAbs(img, alpha=(255.0 / img.max()))
            img = Image.fromarray(img)
            meta = st.session_state.patient_metadata
            meta["Age"] = dicom.get("PatientAge", meta["Age"])
            meta["Gender"] = dicom.get("PatientSex", meta["Gender"])
            meta["Study Date"] = dicom.get("StudyDate", meta["Study Date"])
        else:
            img = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")

        img = preprocess_image(img)
        input_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
            pred_class = np.argmax(probs)
            pred_label = ["No Tumor", "Tumor"][pred_class]
            patient_results.append((uploaded_file.name, pred_label, probs))

        grad_cam = generate_gradcam(model, input_tensor, pred_class)
        grad_cam_img = np.array(img.resize((224, 224)))
        heatmap = cv2.applyColorMap(np.uint8(255 * grad_cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        combined = np.uint8(heatmap * 0.4 + grad_cam_img * 0.6)
        slice_images.append((img, combined, pred_label, probs))

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Patient Info", "🧩 MRI & Grad-CAM", "📊 Probability", "📄 Report PDF"])

    # Tab 1
    with tab1:
        for key, value in st.session_state.patient_metadata.items():
            st.markdown(f"<div class='patient-card'><b>{key}:</b> {value}</div>", unsafe_allow_html=True)

    # Tab 2
    with tab2:
        for i, (orig, gradcam_img, pred, probs) in enumerate(slice_images):
            st.markdown(f"<div class='info-card'><b>Slice {i + 1}:</b> Prediction - {pred}</div>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.image(orig, caption="Original MRI", use_container_width=True)
            with col2:
                st.image(gradcam_img, caption="Grad-CAM Visualization", use_container_width=True)

    # Tab 3 - Probability with Table + Bar Chart
    with tab3:
        df = pd.DataFrame([
            {"File": f, "Prediction": p, "No Tumor %": pr[0]*100, "Tumor %": pr[1]*100}
            for f, p, pr in patient_results
        ])
        st.dataframe(df, use_container_width=True)

        # Bar Chart
        fig, ax = plt.subplots(figsize=(8, len(df) * 0.5))
        for idx, row in df.iterrows():
            color = "#2ecc71" if row["Prediction"] == "No Tumor" else "#e63946"
            ax.barh(row["File"], row["Tumor %"], color=color)
        ax.set_xlabel("Tumor Probability (%)")
        ax.set_ylabel("MRI File")
        st.pyplot(fig)

    # Tab 4 - Report PDF
    with tab4:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(200, 10, "Brain MRI Tumor Classification Report", ln=True, align="C")
        pdf.ln(10)
        pdf.set_font("Arial", "", 12)
        for k, v in st.session_state.patient_metadata.items():
            pdf.cell(200, 10, f"{k}: {v}", ln=True)
        pdf.ln(10)
        for i, (orig, gradcam_img, pred, probs) in enumerate(slice_images):
            pdf.set_font("Arial", "B", 12)
            pdf.cell(200, 10,
                     f"Slice {i + 1}: {pred} (No Tumor: {probs[0]*100:.1f}%, Tumor: {probs[1]*100:.1f}%)", ln=True)
            # Save both images
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp1:
                Image.fromarray(np.array(orig.resize((224, 224)))).save(tmp1.name)
                pdf.image(tmp1.name, w=80)
            pdf.ln(2)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp2:
                Image.fromarray(gradcam_img).save(tmp2.name)
                pdf.image(tmp2.name, w=80)
            pdf.ln(10)
        pdf_output = "mri_report.pdf"
        pdf.output(pdf_output)
        with open(pdf_output, "rb") as f:
            st.download_button("⬇️ Download PDF Report", f, "mri_report.pdf", mime="application/pdf")

# -----------------------------
# Footer
# -----------------------------
st.markdown("""
<hr style='border-top:1px solid #ccc'>
<p style='text-align:center;color:#555;font-size:14px'>
Brain MRI Tumor Detection Dashboard © 2025 
</p>
""", unsafe_allow_html=True)
