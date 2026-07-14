import os
from collections import Counter

import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

# =====================================================
# Read Dataset Information
# =====================================================

@st.cache_data
def load_dataset(dataset_path):

    class_counts = Counter()

    image_sizes = []

    sample_images = {}

    classes = sorted(os.listdir(dataset_path))

    for cls in classes:

        class_path = os.path.join(dataset_path, cls)

        if not os.path.isdir(class_path):
            continue

        images = [
            img for img in os.listdir(class_path)
            if img.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        class_counts[cls] = len(images)

        if images:
            sample_images[cls] = os.path.join(class_path, images[0])

        # Ambil beberapa gambar untuk mengecek resolusi
        for img in images[:5]:

            img_path = os.path.join(class_path, img)

            try:
                with Image.open(img_path) as im:
                    image_sizes.append(im.size)

            except Exception:
                pass

    return class_counts, image_sizes, sample_images


# =====================================================
# Main Function
# =====================================================

def show_eda():

    st.title("📊 Exploratory Data Analysis")

    st.markdown("""
    Visualisasi berikut dibuat langsung dari dataset
    **Garbage Classification (5 Classes)** yang digunakan
    pada proses training model.
    """)

    dataset_path = "garbage_dataset_5class"

    class_counts, image_sizes, sample_images = load_dataset(dataset_path)

    total_images = sum(class_counts.values())

    st.divider()

    st.subheader("📈 Dataset Summary")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Total Images",
        total_images
    )

    c2.metric(
        "Total Classes",
        len(class_counts)
    )

    c3.metric(
        "Rata-rata Gambar/Kelas",
        round(total_images / len(class_counts))
    )


    # =====================================================
    # CLASS DISTRIBUTION
    # =====================================================

    st.divider()

    st.subheader("📊 Class Distribution")

    fig, ax = plt.subplots(figsize=(8, 5))

    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    bars = ax.bar(classes, counts)

    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Images")
    ax.set_title("Number of Images per Class")

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom"
        )

    plt.xticks(rotation=20)

    plt.xticks(rotation=20)

    plt.tight_layout()

    st.pyplot(fig)

    st.info(
    """
    **Insight**

    Distribusi jumlah gambar pada setiap kelas relatif seimbang.
    Hal ini membantu model CNN belajar secara lebih adil terhadap
    masing-masing kategori dan mengurangi potensi bias pada kelas tertentu.
    """
    )


    # =====================================================
    # PIE CHART
    # =====================================================

    st.divider()

    st.subheader("🥧 Class Percentage")

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.pie(
        counts,
        labels=classes,
        autopct="%1.1f%%",
        startangle=90
    )

    ax.axis("equal")
    ax.set_title("Percentage of Images per Class")

    st.pyplot(fig)

    st.info(
    """
    **Insight**

    Persentase gambar antar kelas cukup merata.
    Distribusi yang seimbang membuat proses training lebih stabil
    karena setiap kelas memiliki representasi yang hampir sama.
    """
    )


    # =====================================================
    # TABLE
    # =====================================================

    st.divider()

    st.subheader("📋 Number of Images per Class")

    table_data = {
        "Class": [
            cls.replace("-", " ").title()
            for cls in classes
        ],
        "Images": counts
    }

    st.dataframe(
        table_data,
        use_container_width=True,
        hide_index=True
    )


    # =====================================================
    # SAMPLE IMAGES
    # =====================================================

    st.divider()

    st.subheader("🖼 Sample Images")

    cols = st.columns(len(sample_images))

    for col, (cls, img_path) in zip(cols, sample_images.items()):

        with col:

            img = Image.open(img_path)

            st.image(
                img,
                caption=cls.replace("-", " ").title(),
                use_container_width=True
            )

    st.info(
    """
    **Insight**

    Setiap kelas memiliki karakteristik visual yang berbeda.
    Perbedaan warna, bentuk, tekstur, dan material menjadi informasi
    penting yang dipelajari oleh model CNN untuk membedakan setiap kategori.
    """
    )


    # =====================================================
    # WASTE CATEGORIES CHARACTERISTICS
    # =====================================================

    st.divider()

    st.subheader("♻️ Waste Categories Characteristics")

    characteristics = {

        "paper": {
            "karakteristik": [
                "Permukaan tipis dan datar",
                "Didominasi warna putih atau memiliki tulisan/cetakan",
                "Bahan mudah dilipat",
                "Tekstur bervariasi tergantung jenis kertas"
            ],
            "contoh": [
                "Kertas HVS",
                "Koran",
                "Majalah"
            ]
        },

        "cardboard": {
            "karakteristik": [
                "Bahan lebih tebal dibanding kertas",
                "Didominasi warna cokelat",
                "Memiliki tekstur bergelombang (corrugated)",
                "Bentuk umumnya berupa lembaran atau kotak"
            ],
            "contoh": [
                "Kardus",
                "Kotak kemasan",
                "Karton"
            ]
        },

        "plastic": {
            "karakteristik": [
                "Permukaan halus",
                "Memiliki berbagai warna",
                "Dapat mengilap atau transparan",
                "Bersifat lentur atau kaku tergantung jenis plastik"
            ],
            "contoh": [
                "Botol plastik",
                "Gelas plastik",
                "Kemasan makanan"
            ]
        },

        "metal": {
            "karakteristik": [
                "Permukaan mengilap dan memantulkan cahaya",
                "Bertekstur logam",
                "Material keras dan kaku",
                "Didominasi warna perak atau logam"
            ],
            "contoh": [
                "Kaleng minuman",
                "Kaleng makanan",
                "Wadah logam"
            ]
        },

        "white-glass": {
            "karakteristik": [
                "Bersifat transparan atau semi transparan",
                "Permukaan halus",
                "Memantulkan cahaya",
                "Mudah pecah dibanding material lainnya"
            ],
            "contoh": [
                "Botol kaca",
                "Toples kaca",
                "Gelas kaca"
            ]
        }

    }

    for cls in class_counts.keys():

        with st.expander(cls.replace("-", " ").title()):

            col1, col2 = st.columns([1, 2])

            with col1:
                st.image(
                    sample_images[cls],
                    use_container_width=True
                )

            with col2:

                st.markdown("#### Karakteristik")

                for item in characteristics[cls]["karakteristik"]:
                    st.write(f"• {item}")

                st.markdown("#### Contoh")

                for item in characteristics[cls]["contoh"]:
                    st.write(f"• {item}")


    # =====================================================
    # IMAGE RESOLUTION
    # =====================================================

    st.divider()

    st.subheader("📏 Image Resolution")

    widths = [w for w, h in image_sizes]
    heights = [h for w, h in image_sizes]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Rata-rata Lebar",
            f"{sum(widths)/len(widths):.0f} px"
        )

    with col2:
        st.metric(
            "Rata-rata Tinggi",
            f"{sum(heights)/len(heights):.0f} px"
        )

    with col3:
        st.metric(
            "Jumlah Sampel",
            len(image_sizes)
        )

    resolution_table = {
        "Statistik": [
            "Lebar Minimum",
            "Lebar Maksimum",
            "Tinggi Minimum",
            "Tinggi Maksimum"
        ],
        "Nilai": [
            min(widths),
            max(widths),
            min(heights),
            max(heights)
        ]
    }

    st.dataframe(
        resolution_table,
        use_container_width=True,
        hide_index=True
    )

    st.info("""
    **Insight**

    Resolusi gambar pada dataset bervariasi. Oleh karena itu seluruh gambar
    diubah menjadi ukuran **224 × 224 piksel** sebelum digunakan sebagai
    input model agar proses pelatihan berlangsung secara konsisten.
    """)

    # =====================================================
    # PREPROCESSING SUMMARY
    # =====================================================

    st.divider()

    st.subheader("🧹 Ringkasan Preprocessing")

    col1, col2 = st.columns(2)

    with col1:

        st.success("""
    ### Resize

    Semua gambar diubah menjadi ukuran:

    **224 × 224 piksel**
    """)

        st.success("""
    ### Normalisasi

    Nilai piksel diproses menggunakan
    preprocessing yang sama seperti
    yang digunakan pada model CNN.
    """)

    with col2:

        st.success("""
    ### Data Augmentation

    Augmentasi hanya diterapkan pada
    data training untuk meningkatkan
    kemampuan generalisasi model.
    """)

        st.success("""
    ### CNN Input

    Hasil preprocessing menjadi
    input utama model CNN
    untuk proses klasifikasi.
    """)

        st.info("""
    **Insight**

    Tahapan preprocessing bertujuan untuk membuat seluruh gambar memiliki
    format yang konsisten sehingga model dapat mempelajari pola visual
    dengan lebih optimal. Seluruh proses preprocessing yang digunakan pada
    deployment mengikuti proses yang sama seperti pada notebook training
    dan inference.
    """)
        
    
    st.divider()

st.caption(
    "Dataset : Garbage Classification"
)