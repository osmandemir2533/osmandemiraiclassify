---
title: Animal10AIClassify
emoji: 🐾
colorFrom: teal
colorTo: blue
sdk: streamlit
sdk_version: "1.10.0"
app_file: streamlit_app.py
pinned: true
---

# Animal10 AI Classify

Bu proje, derin öğrenme kullanarak çeşitli hayvan türlerini sınıflandıran bir **Hayvan Sınıflandırma Modeli** ve **Streamlit tabanlı web uygulaması** içerir. Kullanıcılar, yükledikleri fotoğraflardaki hayvanları tanımak için bu aracı kullanabilirler.

---

## 🧑‍💻 Proje Özeti

Bu proje, **PyTorch** tabanlı bir **Convolutional Neural Network (CNN)** modeli kullanarak **10 farklı hayvan sınıfını** sınıflandırmak amacıyla geliştirilmiştir. Eğitilen model, kullanıcıların yüklediği hayvan görüntülerini analiz eder ve doğru tahmin ile birlikte güven skorunu gösterir.

Model, **Animals-10 Dataset**'inden faydalanarak eğitilmiş ve **Streamlit** ile bir arayüz oluşturulmuştur. Kullanıcılar yükledikleri görselin türünü hızlı bir şekilde öğrenebilirler.

---

## 🛠️ Kurulum

1. Gerekli tüm paketleri yükleyin:

```bash
pip install -r requirements.txt
```

2. Modeli eğitin:

```bash
python train.py
```

3. Web arayüzünü başlatın:

```bash
streamlit run streamlit_app.py
```

## 📦 Gereksinimler

Bu proje, aşağıdaki Python kütüphanelerini kullanmaktadır. İlgili kütüphaneleri yüklemek için `requirements.txt` dosyasını kullanabilirsiniz.

### Gereksinimler:

- **streamlit**: Web uygulaması oluşturmak için kullanılan Python kütüphanesidir. Kullanıcı dostu bir arayüzle hızlıca uygulama geliştirilmesini sağlar.
  
  ```bash
  streamlit
  ```

- **torch**: PyTorch, derin öğrenme modelleme için kullanılan güçlü bir kütüphanedir. Modelin eğitimi ve tahmin süreçlerinde bu kütüphane kullanılmıştır.
  
  ```bash
  torch
  ```

- **torchvision**: PyTorch ile birlikte gelen görüntü işleme araçları ve önceden eğitilmiş modelleri içerir. Görüntüleri işlemek ve modelin eğitimi için kullanılır.
  
  ```bash
  torchvision
  ```

- **numpy**: Sayısal işlemler ve veri işleme için kullanılan temel bir kütüphanedir. Modelin eğitim ve test süreçlerinde veri işlemlerinde kullanılmıştır.
  
  ```bash
  numpy
  ```

- **pillow**: Görüntü işleme kütüphanesidir. Resimleri açmak, düzenlemek ve kaydetmek için kullanılır.
  
  ```bash
  pillow
  ```

- **scikit-learn**: Makine öğrenmesi için kullanılan bir Python kütüphanesidir. Modelin değerlendirilmesi ve performans metriklerinin hesaplanması için kullanılır.
  
  ```bash
  scikit-learn
  ```


## 🐍 Veri Seti

Proje, **Animals-10 Dataset** kullanılarak geliştirilmiştir. Bu veri seti, 10 farklı hayvan türünü içeren etiketli görsellerden oluşmaktadır. Her bir sınıf, dengeli bir şekilde dağıtılmış 1000'den fazla görselden oluşmaktadır. 

**Veri Seti Linki:**  
[Animals-10 Dataset](https://www.kaggle.com/datasets/alessiocorrado99/animals10)

### Veri Seti Detayları:
- **Sınıflar:**
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

Veri setini kullanarak, her bir hayvan türünü doğru şekilde sınıflandırmayı amaçlayan bir derin öğrenme modeli geliştirilmiştir.

• **Yükleyeceğiniz fotoğraflarda mümkünse arka plan sade ve net olmalıdır.**

• **Ayrıca, fotoğrafların netliği ve kaliteli olması önemlidir. Fotoğrafın çözünürlüğü ne kadar yüksek olursa, modelin tahmin etme oranı o kadar yüksek olacaktır.**

• **Aksi takdirde, düşük kaliteli veya karmaşık arka planlara sahip fotoğraflarda tahmin doğruluğu düşebilir.**



---

## 📊 Model Detayları

Bu projede kullanılan model, derin öğrenme alanındaki Convolutional Neural Network (CNN) (Eğik Sinir Ağı) yapısını temel alır. Modelin amacı, verilen bir hayvan görselini doğru bir şekilde sınıflandırmaktır. **Eğitilen model ve proje ile ilgili daha fazla bilgi almak veya sorularınızı iletmek için aşağıdaki iletişim kanallarını kullanabilirsiniz.**

### Model Yapısı:
- **Özellik Çıkartıcı (Feature Extractor)**:
    - 3 adet **Convolutional Layer** (Evrişim Katmanı), her biri farklı filtre boyutları ile, hayvan görsellerinden özellikleri çıkarır.
    - Her evrişim katmanından sonra **ReLU** aktivasyon fonksiyonu ve **MaxPooling** kullanılır.
  
- **Sınıflandırıcı (Classifier)**:
    - Çıkartılan özellikler, **Fully Connected (FC) Layer** (Tam Bağlantılı Katmanlar) ile sınıflara dönüştürülür.
    - Sonuç olarak, her sınıf için bir olasılık değeri döndürülür.

- **Dropout ve Early Stopping**:
    - Modelin aşırı uyum sağlamaması için **dropout** tekniği kullanılmıştır.
    - Eğitim sırasında **erken durdurma** (early stopping) uygulanarak, overfitting önlenmeye çalışılmıştır.

### Kullanılan Hiperparametreler:
- **Öğrenme Oranı (Learning Rate)**: 0.001
- **Epoch Sayısı**: 30
- **Batch Size**: 32
- **Dropout Oranı**: 0.5
- **Optimizer**: Adam

### Eğitim Süreci:
Eğitim sırasında her epoch sonunda modelin doğruluğu ve kaybı (loss) kaydedilmiştir. Modelin eğitim kaybı ve doğrulama kaybı (validation loss) her epoch sonunda takip edilmiştir.

Eğitim sonucu model %75 doğruluk oranına ulaşmıştır.

---

## 🖥️ Uygulama

Bu proje, **Streamlit** kullanarak bir web uygulaması olarak geliştirilmiştir. Kullanıcılar, uygulama üzerinden hayvan görselleri yükleyebilir ve model tarafından yapılan tahminleri anında görebilirler.

### Uygulama Arayüzü
Uygulamanın temel özellikleri şu şekildedir:
1. **Görsel Yükleme**: Kullanıcı, bilgisayarından bir hayvan görseli seçebilir.
2. **Modeli Çalıştırma**: Yüklenen görseli model aracılığıyla sınıflandırmak için "Analiz Et" butonuna tıklanır.
3. **Sonuç Gösterimi**: Model, tahmin edilen sınıf ve sınıfın güven skoru ile sonuçları kullanıcıya gösterir.
4. **Temizleme**: Kullanıcılar, yükledikleri görseli ve sonuçları temizlemek için "Temizle" butonunu kullanabilirler.

### Uygulama Adımları
1. **Görsel Yükleme**: Uygulamanın sol kısmındaki "Hayvan Görüntüsü Yükle" alanından bilgisayarınızdan bir hayvan fotoğrafı yükleyin.
2. **Tahmin Yapma**: "Analiz Et" butonuna tıklayın. Uygulama, görseli modelinize gönderecek ve tahmin sonuçlarını bekleyin.
3. **Sonuçları Görme**: Sağ panelde, yüklediğiniz fotoğrafın tahmin edilen sınıfı ve modelin güven skoru görüntülenecektir.
4. **Temizleme**: Görseli ve sonucu temizlemek için "Temizle" butonuna tıklayın.

### Kullanıcı Arayüzü
Uygulama, kullanımı oldukça basit ve etkileşimli bir arayüze sahiptir. Gerekli talimatlar, arayüz üzerinde net bir şekilde sağlanmıştır.

---

## 🚀 Deploy

Proje, **Render** platformunda deploy edilmiştir. Web uygulamasına aşağıdaki bağlantıdan ulaşabilirsiniz:

[osmandemiraiclassify.onrender.com](https://osmandemiraiclassify.onrender.com)

Uygulama, kullanıcıların görsellerini yükleyip anında model tahminleri almasını sağlayacak şekilde canlı olarak çalışmaktadır. Herhangi bir hata ya da performans sorunu ile karşılaşırsanız, proje üzerinde güncellemeler yapılarak çözüm sağlanacaktır.

### Render'da Deploy Ayarları

- **Build Command**:  
  `pip install -r requirements.txt`

- **Start Command**:  
  `streamlit run streamlit_app.py --server.port=8502`

---

## 📚 Katkıda Bulunma

Bu projeye katkıda bulunmak isterseniz, aşağıdaki adımları takip edebilirsiniz:
1. Projeyi **fork** edin.
2. Yeni bir **branch** oluşturun (`git checkout -b feature-name`).
3. Yapmak istediğiniz değişiklikleri ekleyin ve commit işlemi yapın.
4. **Push** işlemi gerçekleştirin (`git push origin feature-name`).
5. **Pull request** oluşturun.

Proje ile ilgili herhangi bir hata, iyileştirme veya öneriniz varsa, **issue** açarak bizimle paylaşabilirsiniz.

---
---

Bu projede yaptığım çalışmalarla ilgili başka sorularınız varsa, bana her zaman ulaşabilirsiniz!  


[![LinkedIn](https://img.icons8.com/ios-filled/50/0A66C2/linkedin.png)](https://www.linkedin.com/in/osmandemir2533/)  &nbsp;&nbsp; 
[![Website](https://img.icons8.com/ios-filled/50/8e44ad/domain.png)](https://osmandemir2533.github.io/)

