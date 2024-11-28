import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

model_filename = 'model.pkl'

with open(model_filename, 'rb') as file:
    model = pickle.load(file)

df = pd.read_csv("heart.csv")

# Fungsi untuk halaman Deskripsi
def show_deskripsi():
    st.header("Prediksi Penyakit Jantung")
    st.write("Selamat datang di aplikasi prediksi kegagalan mesin berbasis web.")
    st.write("<div style='text-align: justify;'>"
    "Aplikasi ini menggunakan teknologi <i>Machine Learning</i> untuk memberikan prediksi yang akurat terkait risiko penyakit jantung berdasarkan beberapa faktor kesehatan utama. "
    "Dengan memasukkan data seperti <b>usia, jenis kelamin, tekanan darah, kolesterol, detak jantung, hasil elektrokardiogram</b>, dan lainnya, pengguna dapat mengetahui tingkat risiko mereka terkena penyakit jantung.<br><br>"
    "Model <i>Machine Learning</i> yang digunakan telah dilatih menggunakan dataset medis terpercaya dengan ribuan data historis. Hal ini memastikan aplikasi mampu memberikan hasil prediksi yang andal. "
    "Aplikasi ini dirancang untuk membantu pengguna dalam mengidentifikasi potensi risiko kesehatan sejak dini, sehingga dapat mengambil langkah pencegahan yang tepat, "
    "seperti perubahan gaya hidup atau konsultasi medis lebih lanjut.<br><br>"
    "Sederhana, responsif, dan mudah digunakan, aplikasi ini adalah alat yang ideal untuk mendukung kesehatan jantung Anda."
    "</div>",
    unsafe_allow_html=True,
)
    st.write("Sumber data: https://github.com/Prem07a/Heart-Disease")
    st.write("Shenia Lusi Himatur Rosyida - 2024")

# Fungsi untuk halaman Dataset
def show_dataset():
    st.header("Dataset")
    st.dataframe(df)
    st.markdown("""
( 1 ) **age (Umur)**
   - Umur dalam tahun
  \n(
2 ) **aex (Jenis Kelamin)**
   - 1 = laki-laki; 0 = perempuan
  \n(
3 ) **cp (Jenis nyeri dada)**
   - 0 : Angina tipikal : nyeri dada berhubungan penurunan suplai darah ke jantung
   - 1: Angina atipikal: nyeri dada yang tidak berhubungan dengan jantung
   - 2: Nyeri non-angina: biasanya kejang esofagus (tidak berhubungan dengan jantung)
   - 3 : Tanpa gejala : nyeri dada tidak menunjukkan tanda-tanda penyakit
  \n(
4 ) **trestbps (tekanan darah istirahat)**
   - (dalam mm Hg saat masuk rumah sakit) di atas 130-140 biasanya menimbulkan kekhawatiran
  \n(
5 ) **Kol (kolestoral serum dalam mg/dl)**
   - serum = LDL + HDL + trigliserida di atas 200 memprihatinkan
  \n(
6 ) **fbs (gula darah puasa > 120 mg/dl)**
   - (1 = benar; 0 = salah) '>126' mg/dL menandakan diabetes
  \n(
7 ) **restecg (hasil elektrokardiografi istirahat)**
   -  0: Tidak ada yang perlu diperhatikan
   - 1 : Kelainan Gelombang ST-T dapat berkisar dari gejala ringan hingga masalah parah menandakan detak jantung tidak normal
   - 2: Kemungkinan atau pasti hipertrofi ventrikel kiri Ruang pemompaan utama jantung membesar
  \n(
8 ) **thalach (detak jantung maksimal tercapai)**
  \n(
9 ) **exang (angina akibat olahraga)**
   - (1 = ya; 0 = tidak)
  \n(
10 ) **oldpeak**
   - Depresi ST yang disebabkan oleh olahraga dibandingkan dengan istirahat terlihat pada stres jantung saat berolahraga. Jantung yang tidak sehat akan lebih stres
  \n(
11 ) **kemiringan – kemiringan puncak latihan segmen ST**
   - 0: Menanjak: detak jantung lebih baik dengan olahraga (jarang)
   - 1: Miring datar: perubahan minimal (tipikal jantung sehat)
   - 2 : Downslopins : tanda jantung tidak sehat
  \n(
12 ) **ca (jumlah pembuluh darah besar (0-3) yang diwarnai dengan flourosopy)**
   - pembuluh berwarna berarti dokter dapat melihat darah yang melewatinya semakin banyak pergerakan darah semakin baik (tidak menggumpal)
  \n(
13 ) **thal (akibat stres thalium)**
   - 1,3: biasa
   - 6: cacat tetap: dulu cacat tapi sekarang oke
   - 7: cacat reversibel: tidak ada pergerakan darah yang baik saat berolahraga
  \n(
14 ) **target (menderita penyakit atau tidak )**
   - (1=ya, 0=tidak) (= atribut yang diprediksi)
""")


# Fungsi untuk halaman Grafik
def show_grafik():
    st.header("Grafik")
    f, ax = plt.subplots(1, 2, figsize=(15, 8))
    
    # Pie chart
    df["target"].replace({0: "No Heart Disease", 1: "Heart Disease"}).value_counts().plot(
        kind="pie", colors=["salmon", "lightblue"], ax=ax[0], explode=[0, 0.1], 
        autopct='%1.1f%%', shadow=True)
    ax[0].set_ylabel('')
    
    # Bar chart
    df["target"].replace({0: "No Heart Disease", 1: "Heart Disease"}).value_counts().plot(
        kind="bar", ax=ax[1], color=["salmon", "lightblue"])
    ax[1].set_ylabel('')
    ax[1].set_xlabel('')
    
    # Menampilkan grafik di Streamlit
    st.pyplot(f)

def show_prediksi():
    st.header("Prediksi")
    st.title('Prediksi Penyakit Jantung')
    age = st.slider('Umur', 18, 100, 50)
    sex_options = ['Pria', 'Wanita']
    sex = st.selectbox('Jenis Kelamin', sex_options)
    sex_num = 1 if sex == 'Male' else 0 
    cp_options = ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Tanpa Gelaja']
    cp = st.selectbox('Jenis Nyeri Dada', cp_options)
    cp_num = cp_options.index(cp)
    trestbps = st.slider('Tekanan Darah Saat Istirahat', 90, 200, 120)
    chol = st.slider('Kolestrol', 100, 600, 250)
    fbs_options = ['Tidak', 'Ya']
    fbs = st.selectbox('Gula Darah Puasa > 120 mg/dl', fbs_options)
    fbs_num = fbs_options.index(fbs)
    restecg_options = ['Normal', 'ST-T Abnormality', 'Left Ventricular Hypertrophy']
    restecg = st.selectbox('Hasil Elektrokardiografi Istirahat', restecg_options)
    restecg_num = restecg_options.index(restecg)
    thalach = st.slider('Detak Jantung Maksimal Tercapai', 70, 220, 150)
    exang_options = ['Tidak', 'Ya']
    exang = st.selectbox('Angina Latihan yang Diinduksi', exang_options)
    exang_num = exang_options.index(exang)
    oldpeak = st.slider('Depresi ST Dipicu oleh Latihan Relatif terhadap Istirahat', 0.0, 6.2, 1.0)
    slope_options = ['Upsloping', 'Flat', 'Downsloping']
    slope = st.selectbox('Kemiringan Puncak Latihan Segmen ST', slope_options)
    slope_num = slope_options.index(slope)
    ca = st.slider('Jumlah Kapal Besar yang diwarnai dengan Fluoroskopi', 0, 4, 1)
    thal_options = ['Normal', 'Fixed Defect', 'Reversible Defect']
    thal = st.selectbox('Thalassemia', thal_options)
    thal_num = thal_options.index(thal)

    with open('mean_std_values.pkl', 'rb') as f:
        mean_std_values = pickle.load(f)


    if st.button('Predict'):
        user_input = pd.DataFrame(data={
            'age': [age],
            'sex': [sex_num],  
            'cp': [cp_num],
            'trestbps': [trestbps],
            'chol': [chol],
            'fbs': [fbs_num],
            'restecg': [restecg_num],
            'thalach': [thalach],
            'exang': [exang_num],
            'oldpeak': [oldpeak],
            'slope': [slope_num],
            'ca': [ca],
            'thal': [thal_num]
        })
        # Apply saved transformation to new data
        user_input = (user_input - mean_std_values['mean']) / mean_std_values['std']
        prediction = model.predict(user_input)
        prediction_proba = model.predict_proba(user_input)

        if prediction[0] == 1:
            bg_color = 'red'
            prediction_result = 'Positive'
        else:
            bg_color = 'green'
            prediction_result = 'Negative'
        
        confidence = prediction_proba[0][1] if prediction[0] == 1 else prediction_proba[0][0]

        st.markdown(f"<p style='background-color:{bg_color}; color:white; padding:10px;'>Prediction: {prediction_result}<br>Confidence: {((confidence*10000)//1)/100}%</p>", unsafe_allow_html=True)

if __name__ == '__show_prediksi__':
    show_prediksi()

add_selectbox = st.sidebar.selectbox(
    "PILIH MENU",
    ("Deskripsi", "Dataset", "Grafik", "Prediksi")
)

if add_selectbox == "Deskripsi":
    show_deskripsi()
elif add_selectbox == "Dataset":
    show_dataset()
elif add_selectbox == "Grafik":
    show_grafik()
elif add_selectbox == "Prediksi":
    show_prediksi()