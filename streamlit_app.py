import streamlit as st
import time

# LOADER EKRANI EN BAŞTA
if 'loader_shown' not in st.session_state:
    st.session_state['loader_shown'] = False

if not st.session_state['loader_shown']:
    start = time.time()
    st.markdown("""
    <style>
    .custom-loader-bg {
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background: #181c24;
        z-index: 9999;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .custom-loader-dot {
        width: 48px;
        height: 48px;
        border-radius: 50%;
        border: 6px solid #fff;
        border-top: 6px solid #43cea2;
        animation: spin 1s linear infinite;
        margin-bottom: 32px;
    }
    @keyframes spin {
        0% { transform: rotate(0deg);}
        100% { transform: rotate(360deg);}
    }
    .custom-loader-text {
        color: #fff;
        font-size: 2.5rem;
        font-weight: bold;
        letter-spacing: 2px;
        margin-bottom: 18px;
    }
    .custom-loader-author {
        color: #fff;
        font-size: 2rem;
        font-weight: bold;
        letter-spacing: 4px;
        margin-top: 24px;
    }
    </style>
    <div class="custom-loader-bg">
        <div class="custom-loader-dot"></div>
        <div class="custom-loader-text">YÜKLENİYOR</div>
        <div class="custom-loader-author">OSMAN DEMİR</div>
    </div>
    """, unsafe_allow_html=True)
    # En az 5 saniye loader gözüksün
    time.sleep(5)
    st.session_state['loader_shown'] = True
    st.rerun()  # YENİ SÜRÜMDE ÇALIŞIR!

else:
    # Tüm importlar burada tekrar edilmeli!
    import torch
    import numpy as np
    from PIL import Image
    from torchvision import transforms
    from train import HayvanSiniflandirmaModeli
    from translate import translate
    import os
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # ... geri kalan kodlar (tüm uygulama) buraya gelecek ...
    # Yani loader_shown True ise, ana uygulama kodu çalışacak
    # Şu anki kodunun geri kalanını buraya taşı
    st.set_page_config(
        page_title="Hayvanı Tahmin Et",
        page_icon="🦁",
        layout="wide"
    )

    st.markdown("""
    <style>
        .header {
            background: linear-gradient(135deg, #43cea2 0%, #185a9d 100%);
            color: white;
            padding: 2rem;
            border-radius: 0 0 2rem 2rem;
            margin-bottom: 2rem;
            text-align: center;
            position: relative;
        }
        .header-names {
            position: absolute;
            top: 50%;
            left: 0; right: 0;
            width: 100%;
            display: flex;
            justify-content: space-between;
            transform: translateY(-50%);
            z-index: 2;
            pointer-events: none;
        }
        .header-name {
            font-size: 2.1rem;
            font-weight: bold;
            color: #111 !important;
            letter-spacing: 2px;
            user-select: none;
        }
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }
        .header p {
            font-size: 1.2rem;
            opacity: 0.9;
        }
        .social-links {
            margin-top: 1rem;
            display: flex;
            justify-content: center;
            gap: 1.5rem;
        }
        .social-links a {
            color: white;
            text-decoration: none;
            font-size: 1.2rem;
        }
        .result-box-dark {
            background-color: #181c24;
            color: #fff;
            padding: 1.7rem 1.5rem 1.2rem 1.5rem;
            border-radius: 1rem;
            margin-top: 1rem;
            box-shadow: 0 2px 16px 0 rgba(0,0,0,0.10);
            font-size: 1.15rem;
            font-weight: 500;
            letter-spacing: 0.5px;
            border: 2px solid #23272f;
            min-height: 400px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }
        .result-box-dark strong {
            color: #fff;
            font-weight: 700;
        }
        .stButton>button {
            width: 100%;
            background-color: #43cea2;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            font-size: 1rem;
            margin-bottom: 0.5rem;
        }
        .stButton>button.temizle-btn {
            background-color: #b71c1c;
            color: #fff;
            border: none;
            margin-top: 0.5rem;
        }
        .stButton>button.temizle-btn:hover {
            background-color: #d32f2f;
        }
    </style>
    """, unsafe_allow_html=True)

    def model_yukle():
        try:
            checkpoint = torch.load('hayvan_model.pth', map_location=torch.device('cpu'))
            model = HayvanSiniflandirmaModeli(len(checkpoint['class_names']))
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            return model, checkpoint['class_names']
        except Exception as e:
            logger.error(f"Model yükleme hatası: {str(e)}")
            st.error(f"Model yüklenirken bir hata oluştu: {str(e)}")
            return None, None

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def tahmin_yap(image):
        try:
            model, class_names = model_yukle()
            if model is None:
                return None, None
            image_pil = Image.fromarray(image.astype('uint8'), 'RGB')
            image_tensor = transform(image_pil).unsqueeze(0)
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, predicted = torch.max(probabilities, 0)
                sinif = class_names[predicted.item()]
                return sinif, confidence.item()*100
        except Exception as e:
            logger.error(f"Tahmin hatası: {str(e)}")
            st.error(f"Tahmin yapılırken bir hata oluştu: {str(e)}")
            return None, None

    # State management
    if 'result' not in st.session_state:
        st.session_state['result'] = ""
    if 'uploaded_image' not in st.session_state:
        st.session_state['uploaded_image'] = None
    if 'uploader_key' not in st.session_state:
        st.session_state['uploader_key'] = 0

    st.markdown("""
    <div class="header">
        <div class="header-names">
            <div class="header-name">OSMAN</div>
            <div class="header-name">DEMİR</div>
        </div>
        <h1>Hayvanı Tahmin Et</h1>
        <p>Yapay Zeka ile Hayvan Tanıma Sistemi</p>
        <div class="social-links">
            <a href="https://osmandemir2533.github.io/" target="_blank">🌐 Web</a>
            <a href="https://www.linkedin.com/in/osmandemir2533/" target="_blank">💼 LinkedIn</a>
            <a href="https://github.com/osmandemir2533" target="_blank">📦 GitHub</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1,1.3], gap="large")

    with col1:
        st.markdown("### 📸 Hayvan Görüntüsü Yükle")
        uploaded_file = st.file_uploader(
            "Bir hayvan fotoğrafı seçin",
            type=["jpg", "jpeg", "png"],
            key=f"fileuploader_{st.session_state['uploader_key']}"
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.session_state['uploaded_image'] = image
        if st.session_state['uploaded_image'] is not None:
            st.image(st.session_state['uploaded_image'], caption="Yüklenen Görüntü", use_container_width=True)
        btn_col1, btn_col2 = st.columns([1,1])
        with btn_col1:
            if st.button("🔍 Analiz Et"):
                if st.session_state['uploaded_image'] is not None:
                    with st.spinner("Analiz yapılıyor..."):
                        sinif, confidence = tahmin_yap(np.array(st.session_state['uploaded_image']))
                        if sinif is not None:
                            st.session_state['result'] = f"""
                            <p style='font-size:2.1rem;'><span style='font-size:2.3rem;'>🎯</span> <strong>Ana Tahmin:</strong> {sinif}</p>
                            <p style='font-size:1.7rem;'><span style='font-size:2.1rem;'>💫</span> <strong>Güven Skoru:</strong> {confidence:.2f}%</p>
                            <p style='font-size:1.3rem;'><span style='font-size:1.7rem;'>📏</span> <strong>Görüntü Boyutu:</strong> 128x128</p>
                            """
                        else:
                            st.session_state['result'] = ""
        with btn_col2:
            if st.button("🗑️ Temizle", key="temizle", help="Yüklenen resmi ve sonucu temizle", type="primary"):
                st.session_state['uploaded_image'] = None
                st.session_state['result'] = ""
                st.session_state['uploader_key'] += 1
                st.session_state.pop(f"fileuploader_{st.session_state['uploader_key']-1}", None)
                st.rerun()

    with col2:
        st.markdown("### 📊 Tahmin Sonuçları")
        kutu_icerik = st.session_state['result'] if st.session_state['result'] else "<span style='font-size: 2rem; color: #888;'>Sonuçlar</span>"
        st.markdown(
            f"""<div class='result-box-dark'>{kutu_icerik}</div>""",
            unsafe_allow_html=True
        )

    # Bilgi bölümü
    st.markdown("""
    ### 🚀 Nasıl Kullanılır?
    - Sol tarafa bir hayvan fotoğrafı yükleyin
    - "Analiz Et" butonuna tıklayın
    - Tahmin sonuçlarını görüntüleyin
    - "Temizle" butonu ile yeni bir görüntü yükleyebilirsiniz

    ### 🦁 Desteklenen Hayvanlar
    - Köpek (Dog)
    - At (Horse)
    - Fil (Elephant)
    - Kelebek (Butterfly)
    - Tavuk (Chicken)
    - Kedi (Cat)
    - İnek (Cow)
    - Koyun (Sheep)
    - Sincap (Squirrel)
    - Örümcek (Spider)

    ### 💡 İpuçları
    - Net ve iyi aydınlatılmış fotoğraflar daha iyi sonuçlar verir
    - Hayvanın tamamının göründüğü fotoğraflar tercih edilir
    - Arka planın sade olması tahmin doğruluğunu artırır
    """)