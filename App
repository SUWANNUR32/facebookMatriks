import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===========================
# LOAD MODEL DAN PREPROCESSOR
# ===========================
rf = joblib.load("rf_model.joblib")
encoders = joblib.load("encoders.joblib")
scaler = joblib.load("scaler.joblib")

# ===========================
# KONFIGURASI HALAMAN
# ===========================
st.set_page_config(page_title="Prediksi Interaksi Postingan Facebook", page_icon="📊", layout="wide")

st.title("📱 Prediksi Total Interaksi Postingan Facebook")
st.markdown("""
Aplikasi ini memprediksi **Total Interaksi** (komentar, like, share, engagement)
berdasarkan atribut postingan Facebook seperti tipe, waktu posting, jangkauan, dan lainnya.

---

### 🧠 Model
Model ini menggunakan **Random Forest Regressor** yang telah dilatih pada dataset Facebook.
""")

# ===========================
# INPUT FORM
# ===========================
st.sidebar.header("📝 Input Data Postingan Baru")

col1, col2 = st.columns(2)

with col1:
    type_input = st.selectbox("Tipe Postingan", ["Photo", "Status", "Link", "Video"])
    category = st.selectbox("Kategori (1–3)", [1, 2, 3])
    post_month = st.number_input("Bulan Postingan (1–12)", 1, 12, 6)
    post_weekday = st.number_input("Hari Postingan (1=Senin ... 7=Minggu)", 1, 7, 3)
    post_hour = st.number_input("Jam Postingan (0–23)", 0, 23, 12)

with col2:
    paid = st.selectbox("Status Paid", ["0 (Tidak)", "1 (Ya)"])
    lifetime_reach = st.number_input("Lifetime Post Total Reach", min_value=0.0, value=10000.0)
    lifetime_impressions = st.number_input("Lifetime Post Total Impressions", min_value=0.0, value=20000.0)
    lifetime_likes = st.number_input("Lifetime Post Reach by People Who Like Your Page", min_value=0.0, value=5000.0)

# ===========================
# PREDIKSI
# ===========================
if st.button("🚀 Prediksi Total Interaksi"):
    # Buat DataFrame baru
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

    # Label Encoding
    for col, encoder in encoders.items():
        new_df[col] = encoder.transform(new_df[col].astype(str))

    # Scaling
    new_df['Total Interactions'] = 0
    cols_to_transform = ['Lifetime Post Total Reach', 'Lifetime Post Total Impressions', 'Total Interactions']
    new_df[cols_to_transform] = scaler.transform(new_df[cols_to_transform])

    # Reorder columns
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

    # Prediksi
    pred_scaled = rf.predict(new_df_processed)
    pred_array = np.array([[0, 0, pred_scaled[0]]])
    pred_asli = scaler.inverse_transform(pred_array)[0, 2]

    # ===========================
    # HASIL PREDIKSI
    # ===========================
    st.success(f"🎯 Prediksi Total Interaksi: **{int(round(pred_asli))}**")
    st.balloons()

    st.markdown("---")
    st.markdown("### 🔍 Rangkuman Data Input")
    st.dataframe(data_baru)

    st.markdown("### 📊 Interpretasi Hasil")
    st.info("""
    Semakin tinggi **Reach** dan **Impressions**, biasanya meningkatkan interaksi.
    Waktu posting (jam & hari) juga berpengaruh besar pada engagement.
    """)

# ===========================
# FOOTER
# ===========================
st.markdown("""
---
👨‍💻 **Dibuat oleh:** Anda  
📦 **Model:** Random Forest Regressor  
💬 Gunakan aplikasi ini untuk membantu menganalisis performa postingan Facebook secara cepat.
""")
