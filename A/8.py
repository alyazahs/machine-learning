import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Sidebar
st.sidebar.title("Select Page")
page = st.sidebar.selectbox("Menu", ["Home", "Dataset", "Visualization"])

# Membaca data dari file CSV
df = pd.read_csv("pinjaman.csv")

# Dataset untuk grafik
data_viz = {
    "Pendapatan": df['Pendapatan'],
    "Jumlah_Pinjam": df['Jumlah_Pinjam'],
}
df_viz = pd.DataFrame(data_viz)

# Halaman Home
if page == "Home":
    st.title("Home Page")
    
    # Menampilkan gambar
    st.image("image.png", caption="Illustration Image", use_container_width=True)
    
    # Menampilkan dataset
    st.write("Dataset:")
    st.dataframe(df)
    
    # Menampilkan grafik
    st.write("Visualization:")
    fig, ax = plt.subplots(figsize=(8, 4))
    df_viz.plot(kind="bar", ax=ax)
    plt.title("Pendapatan VS Jumlah Pinjaman")
    plt.xlabel("Index")
    plt.ylabel("Value")
    st.pyplot(fig)

# Halaman Dataset
elif page == "Dataset":
    st.title("Dataset Page")
    
    # Menampilkan dataset
    st.write("Dataset:")
    st.dataframe(df)

# Halaman Visualization
elif page == "Visualization":
    st.title("Visualization Page")
    
    # Menampilkan grafik
    fig, ax = plt.subplots(figsize=(8, 4))
    df_viz.plot(kind="bar", ax=ax)
    plt.title("Pendapatan VS Jumlah Pinjaman")
    plt.xlabel("Index")
    plt.ylabel("Value")
    st.pyplot(fig)
