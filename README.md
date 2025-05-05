---
title: Animals10Classify
emoji: 🦁
colorFrom: indigo
colorTo: green
sdk: streamlit
sdk_version: "1.31.1"
app_file: streamlit_app.py
pinned: false
---

# Animals10Classify

Yapay zeka ile hayvan türü tahmini yapan Streamlit uygulaması.

## 📋 Özellikler

- Meyve görüntülerini sınıflandırma
- Basit ve kullanıcı dostu web arayüzü
- Yüksek doğruluk oranı
- Gerçek zamanlı tahmin

## 🛠️ Kurulum

1. Gerekli paketleri yükleyin:

```bash
pip install -r requirements.txt
```

2. Modeli eğitin:

```bash
python train.py
```

3. Web arayüzünü başlatın:

```bash
python app.py
```

## 📊 Model Detayları

- Model Mimarisi: CNN (Convolutional Neural Network)
- Giriş Boyutu: 128x128x3
- Çıkış Sınıfları: Meyve kategorileri
- Eğitim Veri Seti: Fruits 360 Dataset

## 🎯 Kullanım

1. Web tarayıcınızda `http://localhost:7860` adresine gidin
2. "Görüntü Yükle" butonuna tıklayın
3. Bir meyve fotoğrafı seçin
4. "Submit" butonuna tıklayın
5. Sonucu görüntüleyin

## 📈 Performans Metrikleri

- Doğruluk (Accuracy)
- Kesinlik (Precision)
- Duyarlılık (Recall)

## 👥 Katkıda Bulunma

1. Bu depoyu fork edin
2. Yeni bir branch oluşturun (`git checkout -b feature/yeniOzellik`)
3. Değişikliklerinizi commit edin (`git commit -am 'Yeni özellik eklendi'`)
4. Branch'inizi push edin (`git push origin feature/yeniOzellik`)
5. Pull Request oluşturun
