import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
from translate import translate

# Hayvan sınıflandırma için CNN modeli
class HayvanSiniflandirmaModeli(nn.Module):
    def __init__(self, num_classes):
        super(HayvanSiniflandirmaModeli, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Görüntüleri ön işleme için gerekli dönüşümleri tanımladım
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Tüm görüntüleri 128x128 boyutuna getiriyorum
    transforms.ToTensor(),  # Görüntüleri PyTorch tensörlerine çeviriyorum
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet değerleriyle normalize ediyorum
])

# Veri setini yüklemek için bir fonksiyon yazdım
def veri_seti_yukle(data_path):
    # Veri setini ImageFolder ile yüklüyorum. Bu otomatik olarak sınıfları algılayacak
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    
    # Veriyi eğitim ve doğrulama için %80-%20 oranında bölüyorum
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Veri yükleyicileri oluşturuyorum. Batch size 32 seçtim
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Sınıf isimlerini Türkçe'ye çevir
    class_names = [translate[cls] for cls in dataset.classes]
    
    return train_loader, val_loader, len(dataset.classes), class_names

# Model eğitimi için ana fonksiyonu yazdım
def model_egit(model, train_loader, val_loader, num_epochs=10, start_epoch=0, checkpoint_path=None):
    # GPU varsa onu kullanacağım, yoksa CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Kayıp fonksiyonu olarak CrossEntropyLoss kullanıyorum
    criterion = nn.CrossEntropyLoss()
    # Optimizer olarak Adam'ı seçtim, öğrenme oranı 0.001
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Eğitim ve validasyon metriklerini saklamak için listeler
    train_losses = []
    val_losses = []
    accuracies = []
    
    # Eğer checkpoint varsa, modeli ve optimizer'ı yükle
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        accuracies = checkpoint['accuracies']
        print(f"Checkpoint'ten yüklendi. Epoch {start_epoch}'den devam ediliyor.")
    
    # Eğitim döngüsü
    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()  # Eğitim moduna geçiyorum
        running_loss = 0.0
        
        # Eğitim verilerini işliyorum
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()  # Gradyanları sıfırlıyorum
            outputs = model(inputs)  # İleri yayılım
            loss = criterion(outputs, labels)  # Kaybı hesaplıyorum
            loss.backward()  # Geri yayılım
            optimizer.step()  # Ağırlıkları güncelliyorum
            running_loss += loss.item()
            
        # Validasyon aşaması
        model.eval()  # Değerlendirme moduna geçiyorum
        val_loss = 0.0
        correct = 0
        total = 0
        
        # Validasyon verilerini işliyorum
        with torch.no_grad():  # Gradyan hesaplamıyorum
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        # Metrikleri kaydet
        train_loss = running_loss/len(train_loader)
        val_loss = val_loss/len(val_loader)
        accuracy = 100.*correct/total
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        accuracies.append(accuracy)
                
        # Sonuçları yazdırıyorum
        print(f'Epoch {epoch+1}/{start_epoch + num_epochs}:')
        print(f'Eğitim Kaybı: {train_loss:.3f}')
        print(f'Validasyon Kaybı: {val_loss:.3f}')
        print(f'Doğruluk: {accuracy:.2f}%\n')
        
        # Overfitting kontrolü
        if len(val_losses) > 5 and val_losses[-1] > val_losses[-2] and val_losses[-2] > val_losses[-3]:
            print("Overfitting tespit edildi! Eğitim durduruluyor...")
            break

    return model

if __name__ == "__main__":
    # Ana çalıştırma bloğu
    data_path = "archive/raw-img"  # Yeni veri seti yolu
    
    # Veri setini yüklüyorum
    train_loader, val_loader, num_classes, class_names = veri_seti_yukle(data_path)
    
    # Modeli oluşturup eğitiyorum
    model = HayvanSiniflandirmaModeli(num_classes)
    
    # Eski modeli sil
    if os.path.exists('hayvan_model.pth'):
        os.remove('hayvan_model.pth')
        print("Eski model silindi.")
    
    # Yeni modeli eğit
    print("Yeni model eğitimi başlıyor...")
    model = model_egit(model, train_loader, val_loader, num_epochs=30)  # Epoch sayısını 30'a çıkardık
    
    # Final modeli kaydet
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names
    }, 'hayvan_model.pth')

    print("Model eğitimi tamamlandı ve kaydedildi.") 