"""
Dashboard Prediksi Kalori Terbakar
Model Machine Learning untuk estimasi kalori yang terbakar saat olahraga
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Kalori Terbakar",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model dan preprocessor
@st.cache_resource
def load_model():
    """Load model dan preprocessor yang telah disimpan"""
    try:
        model = joblib.load('best_calorie_model.pkl')
        scaler = joblib.load('scaler.pkl')
        with open('feature_columns.pkl', 'rb') as f:
            features = pickle.load(f)
        return model, scaler, features
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.info("Pastikan file model (best_calorie_model.pkl, scaler.pkl, feature_columns.pkl) berada di direktori yang sama")
        return None, None, None

# Fungsi untuk menghitung BMR (Basal Metabolic Rate)
def calculate_bmr(weight, height, age, gender):
    """Menghitung BMR menggunakan rumus Mifflin-St Jeor"""
    if gender == "Laki-laki":
        return 10 * weight + 6.25 * height - 5 * age + 5
    else:
        return 10 * weight + 6.25 * height - 5 * age - 161

# Fungsi untuk membuat prediksi
def predict_calories(model, scaler, features, input_data):
    """Melakukan prediksi kalori berdasarkan input pengguna"""
    # Buat DataFrame dari input
    df_input = pd.DataFrame([input_data])
    
    # Hitung BMR dan fitur interaksi
    df_input['BMR'] = df_input.apply(
        lambda row: calculate_bmr(row['Weight'], row['Height'], row['Age'], row['Gender']),
        axis=1
    )
    df_input['BMR_x_Duration'] = df_input['BMR'] * df_input['Duration']
    
    # Encoding Gender
    df_input['Gender_male'] = (df_input['Gender'] == "Laki-laki").astype(int)
    
    # Pilih fitur yang sesuai
    X = df_input[features]
    
    # Normalisasi fitur numerik
    numeric_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 
                        'Body_Temp', 'BMR', 'BMR_x_Duration']
    X[numeric_features] = scaler.transform(X[numeric_features])
    
    # Prediksi
    prediction = model.predict(X)[0]
    
    return prediction

# Main app
def main():
    # Header
    st.title("🔥 Prediksi Kalori yang Terbakar")
    st.markdown("""
    Masukkan data diri dan aktivitas olahraga Anda untuk mendapatkan estimasi 
    jumlah kalori yang terbakar selama berolahraga.
    """)
    
    # Sidebar untuk input
    with st.sidebar:
        st.header("📊 Data Diri")
        
        gender = st.radio(
            "Jenis Kelamin",
            options=["Laki-laki", "Perempuan"],
            horizontal=True
        )
        
        age = st.number_input(
            "Usia (tahun)",
            min_value=15, max_value=100, value=30, step=1
        )
        
        col1, col2 = st.columns(2)
        with col1:
            height = st.number_input("Tinggi Badan (cm)", min_value=100, max_value=250, value=170, step=1)
        with col2:
            weight = st.number_input("Berat Badan (kg)", min_value=30, max_value=200, value=70, step=1)
        
        st.divider()
        st.header("🏃 Aktivitas Olahraga")
        
        duration = st.number_input("Durasi Olahraga (menit)", min_value=1, max_value=180, value=30, step=5)
        heart_rate = st.number_input("Denyut Jantung (bpm)", min_value=60, max_value=200, value=120, step=5)
        body_temp = st.number_input("Suhu Tubuh (°C)", min_value=35.0, max_value=42.0, value=37.5, step=0.1, format="%.1f")
        
        predict_button = st.button("🔮 Prediksi Kalori", type="primary", use_container_width=True)
    
    # Load model
    model, scaler, features = load_model()
    
    if predict_button and model is not None:
        input_data = {
            'Age': age, 'Height': height, 'Weight': weight,
            'Duration': duration, 'Heart_Rate': heart_rate,
            'Body_Temp': body_temp, 'Gender': gender
        }
        
        with st.spinner("Menghitung prediksi..."):
            prediction = predict_calories(model, scaler, features, input_data)
            bmr = calculate_bmr(weight, height, age, gender)
            calories_per_hour = (prediction / duration) * 60
        
        st.balloons()
        
        # Hasil
        col1, col2, col3 = st.columns(3)
        col1.metric("🔥 Kalori Terbakar", f"{prediction:.0f} kkal")
        col2.metric("⚡ Intensitas", f"{calories_per_hour:.0f} kkal/jam")
        col3.metric("💪 BMR", f"{bmr:.0f} kkal/hari")
        
        # Interpretasi
        st.divider()
        if calories_per_hour < 300:
            st.success(f"✅ Intensitas Ringan - {calories_per_hour:.0f} kkal/jam")
        elif calories_per_hour < 500:
            st.info(f"📈 Intensitas Sedang - {calories_per_hour:.0f} kkal/jam")
        else:
            st.warning(f"⚡ Intensitas Tinggi - {calories_per_hour:.0f} kkal/jam")
    
    elif not predict_button and model is not None:
        st.info("👈 Masukkan data di sidebar, lalu klik 'Prediksi Kalori'")
        
        # Tampilkan info fitur
        st.subheader("📊 Faktor yang Mempengaruhi Kalori Terbakar")
        feature_importance = {
            'Durasi Olahraga': 0.65,
            'Denyut Jantung': 0.15,
            'Suhu Tubuh': 0.08,
            'Berat Badan': 0.05,
            'Usia': 0.03,
            'Tinggi Badan': 0.02,
            'BMR': 0.01,
            'Gender': 0.01
        }
        
        fig = px.bar(
            x=list(feature_importance.values()),
            y=list(feature_importance.keys()),
            orientation='h',
            title="Tingkat Pengaruh Fitur",
            labels={'x': 'Pengaruh', 'y': 'Fitur'},
            color=list(feature_importance.values()),
            color_continuous_scale='Oranges'
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
