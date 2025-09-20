import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ==========================
# 🎯 Configuración general
# ==========================
st.set_page_config(page_title="Análisis Estadístico", page_icon="📊", layout="wide")

# 🌿 Fondo con CSS
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1501785888041-af3ef285b470");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
[data-testid="stSidebar"] {
    background-color: rgba(255, 255, 255, 0.85);
    border-radius: 15px;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# ==========================
# 📂 Cargar datos
# ==========================
dataset_file = "Crop_recommendation.csv"

if not os.path.exists(dataset_file):
    st.error(f"⚠️ No se encontró el archivo `{dataset_file}`. Por favor colócalo en la carpeta.")
    st.stop()

@st.cache_data
def load_data():
    return pd.read_csv(dataset_file)

df = load_data()

# ==========================
# 📌 Menú de navegación
# ==========================
st.sidebar.title("📌 Menú principal")
pagina = st.sidebar.radio("Selecciona una sección:", 
                          ["📂 Exploración de datos", "📊 Visualización", "🌱 Predicción"])

# ==========================
# 📂 Exploración de datos
# ==========================
if pagina == "📂 Exploración de datos":
    st.header("📂 Exploración de Datos")
    st.write("Vista previa de los datos cargados:")
    st.dataframe(df.head())

# ==========================
# 📊 Visualización
# ==========================
elif pagina == "📊 Visualización":
    st.header("📊 Visualizaciones")

    # Seleccionar variable para histograma
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    variable = st.sidebar.selectbox("Elige una variable para graficar:", numeric_cols)

    st.subheader(f"Histograma de {variable}")
    fig, ax = plt.subplots()
    sns.histplot(df[variable], kde=True, ax=ax)
    st.pyplot(fig)

    # Heatmap de correlación
    st.subheader("Mapa de calor de correlaciones")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="YlGnBu", ax=ax)
    st.pyplot(fig)

# ==========================
# 🌱 Predicción (modelo simple)
# ==========================
elif pagina == "🌱 Predicción":
    st.header("🌱 Predicción de cultivo con RandomForest")

    X = df.drop("label", axis=1)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    st.write("Exactitud del modelo:", model.score(X_test, y_test))

    st.subheader("Hacer una predicción")
    inputs = {}
    for col in X.columns:
        inputs[col] = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))

    if st.button("Predecir cultivo recomendado"):
        pred = model.predict([list(inputs.values())])
        st.success(f"🌾 Cultivo recomendado: **{pred[0]}**")
