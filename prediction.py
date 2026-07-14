import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# =====================================================
# Load Model
# =====================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "garbage_classification_mobilenetv2.keras",
        custom_objects={
            "preprocess_input": preprocess_input
        }
    )

    return model

# =====================================================
# Target Classes
# =====================================================

TARGET_CLASSES = [
    "cardboard",
    "metal",
    "paper",
    "plastic",
    "white-glass"
]

# =====================================================
# Display Names
# =====================================================

DISPLAY_NAMES = {
    "cardboard": "Cardboard",
    "metal": "Metal",
    "paper": "Paper",
    "plastic": "Plastic",
    "white-glass": "White Glass"
}


# =====================================================
# Prediction Page
# =====================================================

def show_prediction():

    st.title("🤖 Garbage Classification Prediction")

    st.markdown("""
    Upload gambar sampah kemudian klik **Predict**
    untuk mengetahui hasil klasifikasi menggunakan
    model CNN.
    """)

    st.divider()

    model = load_model()

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is None:
        st.info("Silakan upload gambar terlebih dahulu.")
        return

    # =====================================================
    # Preview Image
    # =====================================================

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Preview Image",
        use_container_width=True
    )

    width, height = image.size

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Nama File",
            uploaded_file.name
        )

    with col2:
        st.metric(
            "Format",
            uploaded_file.type.split("/")[-1].upper()
        )

    with col3:
        st.metric(
            "Resolusi",
            f"{width} × {height}"
        )

    st.divider()

    if st.button(
        "🔍 Predict",
        use_container_width=True
    ):

        # ===============================================
        # PREPROCESSING
        # ===============================================

        image = image.resize((224, 224))

        image_array = np.array(image)

        image_input = np.expand_dims(
            image_array,
            axis=0
        )

        # ===============================================
        # PREDICTION
        # ===============================================

        with st.spinner("Melakukan prediksi..."):

            prediction = model.predict(
                image_input,
                verbose=0
            )

        predicted_index = np.argmax(prediction)

        predicted_label = TARGET_CLASSES[predicted_index]

        confidence = float(
            np.max(prediction)
        )

        st.success("Prediksi berhasil dilakukan.")

        st.divider()

        # ===============================================
        # RESULT
        # ===============================================

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "♻️ Predicted Class",
                DISPLAY_NAMES[predicted_label]
            )

        with col2:

            st.metric(
                "🎯 Confidence Score",
                f"{confidence*100:.2f}%"
            )

        if confidence >= 0.90:
            st.success("✅ Model memiliki tingkat keyakinan yang sangat tinggi terhadap hasil prediksi.")

        elif confidence >= 0.70:
            st.info("ℹ️ Model memiliki tingkat keyakinan yang cukup baik terhadap hasil prediksi.")

        else:
            st.warning("⚠️ Tingkat keyakinan model masih rendah. Gunakan gambar yang lebih jelas atau objek yang lebih fokus.")

        st.divider()

        # ===============================================
        # Probability Table
        # ===============================================

        probability = np.round(
            prediction[0] * 100,
            2
        )

        result_df = pd.DataFrame({
            "Class": [DISPLAY_NAMES[label] for label in TARGET_CLASSES],
            "Probability (%)": probability
        })

        result_df = result_df.sort_values(
            by="Probability (%)",
            ascending=False
        ).reset_index(drop=True)

        st.subheader("📋 Probabilitas Setiap Kelas")

        st.write(
            "Berikut merupakan probabilitas prediksi untuk setiap kelas berdasarkan hasil inferensi model CNN."
        )

        st.dataframe(
            result_df.style.format({
                "Probability (%)": "{:.2f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        # ===============================================
        # Probability Chart
        # ===============================================

        st.subheader("📊 Probability Distribution")

        fig, ax = plt.subplots(figsize=(8,4))

        bars = ax.bar(
            result_df["Class"],
            result_df["Probability (%)"]
        )

        for bar in bars:

            height = bar.get_height()

            ax.text(
                bar.get_x()+bar.get_width()/2,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

        ax.set_ylabel("Probability (%)")
        ax.set_xlabel("Class")
        ax.set_title("Prediction Probability")

        ax.set_ylim(0, 100)

        ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.4
        )

        plt.xticks(rotation=20)

        st.pyplot(fig)

        st.info(
            """
        **Insight**

        Grafik menunjukkan distribusi probabilitas prediksi untuk seluruh kelas.
        Semakin tinggi nilai probabilitas pada suatu kelas, semakin besar tingkat keyakinan model bahwa gambar termasuk ke dalam kategori tersebut.
        """
        )