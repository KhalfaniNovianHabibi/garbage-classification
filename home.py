import streamlit as st


def show_home():

    # =====================================================
    # HEADER
    # =====================================================
    st.title("♻️ Garbage Classification using CNN")

    st.markdown("""
    ### Computer Vision Project

    Aplikasi ini menggunakan **Convolutional Neural Network (CNN)** untuk
    mengklasifikasikan gambar sampah ke dalam **5 kategori** sehingga dapat
    membantu proses pemilahan sampah secara otomatis.
    """)

    st.divider()

    # =====================================================
    # METRICS
    # =====================================================

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("📂 Dataset", "Garbage")
    col2.metric("🤖 Model", "CNN")
    col3.metric("🗂 Classes", "5")
    col4.metric("🛠 Framework", "TensorFlow")

    st.divider()

    # =====================================================
    # PROJECT INFORMATION
    # =====================================================

    left, right = st.columns([1.2, 1])

    with left:

        st.subheader("📌 Project Overview")

        st.info("""
    Project ini bertujuan untuk membangun model **Computer Vision**
    yang mampu mengenali jenis sampah menggunakan pendekatan
    **CNN Image Classification**.

    Model dilatih menggunakan dataset Garbage Classification
    dengan lima kategori sampah sehingga dapat membantu proses
    pemilahan sampah secara otomatis.
    """)

    with right:

        st.subheader("🎯 Project Objective")

        st.success("""
    Mengklasifikasikan gambar sampah menjadi:

    - Paper
    - Cardboard
    - Plastic
    - Metal
    - White Glass
    """)

    st.divider()

    # =====================================================
    # DATASET INFORMATION
    # =====================================================

    st.subheader("🗂 Dataset Information")

    c1, c2 = st.columns(2)

    with c1:

        st.markdown("""
    **Dataset**

    Garbage Classification Dataset

    **Framework**

    - TensorFlow
    - Streamlit
    - NumPy
    - Matplotlib
    """)

    with c2:

        st.markdown("""
    **Model**

    Convolutional Neural Network (CNN)

    **Number of Classes**

    5 Classes
    """)

    st.divider()

    # =====================================================
    # CLASSES
    # =====================================================

    st.subheader("♻️ Waste Categories")

    a, b, c, d, e = st.columns(5)

    a.success("📄\n\nPaper")
    b.success("📦\n\nCardboard")
    c.success("🧴\n\nPlastic")
    d.success("🥫\n\nMetal")
    e.success("🍾\n\nWhite Glass")

    st.divider()

    # =====================================================
    # WORKFLOW
    # =====================================================

    st.subheader("🔄 Workflow")

    step1, step2, step3, step4 = st.columns(4)

    step1.info("📂 Dataset")
    step2.info("🧹 Preprocessing")
    step3.info("🧠 CNN Model")
    step4.info("🎯 Prediction")

    st.markdown(
        """
    Dataset

    ⬇️

    Preprocessing

    ⬇️

    CNN Model

    ⬇️

    Prediction
    """
        )

    st.divider()

    # =====================================================
    # HOW TO USE
    # =====================================================

    st.subheader("🚀 How to Use")

    st.container()

    st.markdown("""
    1. Buka menu **Prediction** pada sidebar.

    2. Upload gambar dengan format:

    - JPG
    - JPEG
    - PNG

    3. Klik tombol **Predict**.

    4. Sistem akan menampilkan:

    - Predicted Class
    - Confidence Score
    - Probability setiap kelas
    """)

    st.divider()

    # =====================================================
    # DATASET SOURCE
    # =====================================================

    st.markdown("""
    Dataset yang digunakan pada project ini berasal dari **Kaggle**.

    **Garbage Classification Dataset**

    Dataset asli memiliki **12 kelas**, namun pada project ini hanya digunakan **5 kelas**, yaitu:

    - 📄 Paper
    - 📦 Cardboard
    - 🧴 Plastic
    - 🥫 Metal
    - 🍾 White Glass

    Pemilihan lima kelas dilakukan agar model fokus pada kategori sampah yang digunakan selama proses training, evaluasi, dan deployment.
    """)

    st.link_button(
        "🔗 Lihat Dataset di Kaggle",
        "https://www.kaggle.com/datasets/mostafaabla/garbage-classification"
    )

    # =====================================================
    # FOOTER
    # =====================================================

    st.caption(
        "Computer Vision | CNN Image Classification"
    )