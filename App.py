import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

# ===========================
# KONFIGURASI HALAMAN
# ===========================
st.set_page_config(
    page_title="Prediksi Interaksi Postingan Facebook",
    page_icon="📊",
    layout="wide"
)

st.title("📱 Prediksi Total Interaksi Postingan Facebook")
st.markdown("""
Aplikasi ini memprediksi **Total Interaksi** (komentar, like, share, engagement)
berdasarkan atribut postingan Facebook seperti tipe, waktu posting, jangkauan, dan lainnya.

---

### 🧠 Model
Model ini menggunakan **Random Forest Regressor** yang telah dilatih pada dataset Facebook.
""")

# ===========================
# LOAD MODEL DAN PREPROCESSOR
# ===========================
@st.cache_resource
def load_models():
    try:
        st.write("📂 Files di direktori saat ini:", os.listdir())
        if not os.path.exists("rf_model.joblib"):
            raise FileNotFoundError("File rf_model.joblib tidak ditemukan di direktori kerja.")
        rf = joblib.load("rf_model.joblib")
        encoders = joblib.load("encoders.joblib")
        scaler = joblib.load("scaler.joblib")
        return rf, encoders, scaler
    except Exception as e:
        st.error(f"❌ Gagal memuat model atau file pendukung: {e}")
        return None, None, None

rf, encoders, scaler = load_models()

if not rf or not encoders or not scaler:
    st.stop()

# ===========================
# FORM INPUT
# ===========================
st.sidebar.header("📝 Input Data Postingan Baru")

col1, col2 = st.columns(2)

with col1:
    type_input = st.selectbox("Tipe Postingan", ["Photo", "Status", "Link", "Video"])
    category = st.selectbox("Kategori (1–3)", [1, 2, 3])
    post_month = st.number_input("Bulan Postingan (1–12)", min_value=1, max_value=12, value=6)
    post_weekday = st.number_input("Hari Postingan (1=Senin ... 7=Minggu)", min_value=1, max_value=7, value=3)
    post_hour = st.number_input("Jam Postingan (0–23)", min_value=0, max_value=23, value=12)

with col2:
    paid = st.selectbox("Status Paid", ["0 (Tidak)", "1 (Ya)"])
    lifetime_reach = st.number_input("Lifetime Post Total Reach", min_value=0.0, value=10000.0, step=100.0)
    lifetime_impressions = st.number_input("Lifetime Post Total Impressions", min_value=0.0, value=20000.0, step=100.0)
    lifetime_likes = st.number_input("Lifetime Post Reach by People Who Like Your Page", min_value=0.0, value=5000.0, step=100.0)

# ===========================
# PREDIKSI
# ===========================
if st.button("🚀 Prediksi Total Interaksi"):
    try:
        data_baru = {
            'Page total likes': [139441],
            'Type': [type_input],
            'Category': [category],
            'Post Month': [post_month],
            'Post Weekday': [post_weekday],
            'Post Hour': [post_hour],
            'Paid': [int(paid.split()[0])],
            'Lifetime Post Total Reach': [lifetime_reach],
            'Lifetime Post Total Impressions': [lifetime_impressions],
            'Lifetime Engaged Users': [300],
            'Lifetime Post Consumers': [200],
            'Lifetime Post Consumptions': [250],
            'Lifetime Post Impressions by people who have liked your Page': [5000],
            'Lifetime Post reach by people who like your Page': [lifetime_likes],
            'Lifetime People who have liked your Page and engaged with your post': [200],
            'comment': [0],
            'like': [0],
            'share': [0],
        }

        new_df = pd.DataFrame(data_baru)

        for col, encoder in encoders.items():
            new_df[col] = encoder.transform(new_df[col].astype(str))

        new_df['Total Interactions'] = 0
        cols_to_transform = ['Lifetime Post Total Reach', 'Lifetime Post Total Impressions', 'Total Interactions']
        new_df[cols_to_transform] = scaler.transform(new_df[cols_to_transform])

        X_cols = [
            'Page total likes', 'Type', 'Category', 'Post Month', 'Post Weekday',
            'Post Hour', 'Paid', 'Lifetime Post Total Reach',
            'Lifetime Post Total Impressions', 'Lifetime Engaged Users',
            'Lifetime Post Consumers', 'Lifetime Post Consumptions',
            'Lifetime Post Impressions by people who have liked your Page',
            'Lifetime Post reach by people who like your Page',
            'Lifetime People who have liked your Page and engaged with your post',
            'comment', 'like', 'share'
        ]
        new_df_processed = new_df[X_cols]

        pred_scaled = rf.predict(new_df_processed)
        pred_array = np.array([[0, 0, pred_scaled[0]]])
        pred_asli = scaler.inverse_transform(pred_array)[0, 2]

        # ===========================
        # HASIL PREDIKSI & INSIGHT
        # ===========================
        st.success(f"🎯 Prediksi Total Interaksi: **{int(round(pred_asli))}**")
        st.balloons()

        # METRIK CEPAT
        colm1, colm2, colm3 = st.columns(3)
        colm1.metric("Reach", f"{int(lifetime_reach):,}")
        colm2.metric("Impressions", f"{int(lifetime_impressions):,}")
        colm3.metric("Jam Posting", f"{post_hour}:00")

        st.markdown("---")
        st.markdown("### 🔍 Rangkuman Data Input")
        st.dataframe(new_df)

        # ===========================
        # VISUALISASI GRAFIK
        # ===========================
        st.markdown("### 📊 Visualisasi Prediksi dan Faktor Utama")

        data_vis = pd.DataFrame({
            'Faktor': ['Reach', 'Impressions', 'Likes by Fans'],
            'Nilai': [lifetime_reach, lifetime_impressions, lifetime_likes]
        })

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(data_vis['Faktor'], data_vis['Nilai'], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax.set_xlabel('Nilai')
        ax.set_title('Perbandingan Faktor Utama')
        st.pyplot(fig)

        st.markdown("### 📈 Interpretasi")
        st.info(f"""
        📌 **Prediksi interaksi:** sekitar **{int(round(pred_asli))}** total aksi pengguna.  
        🔹 Postingan bertipe **{type_input}** dan kategori **{category}** cenderung memberi hasil yang berbeda tergantung **jam posting** ({post_hour}:00).  
        🔹 Semakin tinggi **Reach** dan **Impressions**, potensi interaksi juga meningkat.  
        🔹 Pastikan melakukan **boosting (Paid Ads)** untuk meningkatkan jangkauan postingan.
        """)

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat prediksi: {e}")

# ===========================
# FOOTER
# ===========================
st.markdown("""
---
👨‍💻 **Dibuat oleh:** SUWAN GANTENK  
📦 **Model:** Random Forest Regressor  
💬 Aplikasi ini membantu Anda memahami faktor-faktor yang memengaruhi interaksi postingan Facebook.
""")
