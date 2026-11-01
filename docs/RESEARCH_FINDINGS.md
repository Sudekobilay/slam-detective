# Araştırma bulguları

[Dokümantasyona dön](README.md)

Bu belge gönderilen güncel ZIP içindeki `results/exp_42_20261101T161014994377Z/summary.json` ile doğrulanan sonuçları özetler. Bu kayıtta 480 çalıştırma ve gerekli süreler tamamlanmış; korunan kaynaklar ve kalibrasyon değişmemiştir. İşlem süreleri donanıma bağlıdır.

## 1. Hareket, belirli koşulda kaynak teşhisini değiştirdi

EXP-42: koridor, normal rota yay, ön LiDAR +5°.

| Politika | Doğru kaynak | Belirsiz | Davranış |
| --- | --- | --- | --- |
| Pasif | 0/10 | 10/10 | Yay devam |
| Sabit yay | 0/10 | 10/10 | Yay devam |
| Sabit düz | 10/10 | 0/10 | Düz harekete geçiş |
| Adaptif | 10/10 | 0/10 | Düz seçildi |

Bu koşulda kısa hareket değişikliği, kaynak doğrulaması için gereken ardışık kanıtı sağladı. Sabit düz de aynı başarıyı verdiğinden adaptif yöntem en iyi sabit politikadan üstün değildir.

## 2. Sonuç bütün koşullara genellenmiyor

Diğer üç +5° ortam/normal-rota hücresinde bütün politikalar 10/10 doğru kaynak doğrulaması verdi. Geometrik puanlarda adaptif her zaman en yüksek değeri sağlamadı. Dolayısıyla ana sonuç, belirli bir geometrinin teşhiste hareketi önemli kılmasıdır; her alarmda manevra yapmanın yararlı olduğu değildir.

+2° koşullarında hiçbir politika alarm üretmedi. Dondurulmuş göreli açı eşiği 2.5° idi. Küçük hata duyarlılığı açık bir sınırlamadır; sonuca bakılarak eşik düşürülmedi.

Hatasız dört ortam/rota hücresinde her politika için yanlış alarm 0/10'du. Politikalar aynı sağlıklı kayıtları paylaştığından bu, 160 bağımsız sağlıklı kayıt değildir. Temel sağlıklı koşul sayısı 40'tır.

## 3. Harita ve ölçüm seçimi teşhis kanıtını etkiliyor

Önceki deneylerde yalnız eşleşen çizgilerin NIS değerini izlemek bazı hataları kaçırdı. Ham tarama hizalama izlemesi bu nedenle eklendi. Hatalı ölçümlerin haritaya işlenmesi sonraki model farklarını değiştirebildiğinden koruma ve donmuş referans deneyleri ayrı tutuldu. EXP-30 ile EXP-42'nin kalibrasyon ve ortamları farklıdır; oranlar doğrudan aynı test gibi birleştirilmez.

## 4. Algoritma çeşitliliği genel üstünlük göstermedi

EXP-38 nominal örneklerinde EKF daha düşük RMSE verdi. FastSLAM 2.0 her koşulda FastSLAM 1.0'dan iyi değildi. Parçacık tohumları sensör tekrarı değildir. Çok LiDAR eklemek de her ayarda daha düşük hata garantilemedi.

## 5. GPS koruması ve gerçek kayıt ayrı kanıtlar

GPS sapmalı koşullarda korumalı füzyon incelendi. Nominal GPS eklemek çok iyi odometri bulunan kısa örneklerde hatayı artırabilir. Gerçek Freiburg kaydı yeniden oynatıldı; bağımsız gerçek konum bulunmadığından doğruluk RMSE'si üretilmedi. Gerçek kayıtta yeni teşhis hareketi uygulanmadı.

## Sonucun sınırı

Deneyler çalışan bir araştırma prototipi ve koşula bağlı hareket katkısı gösterir. Evrensel adaptif üstünlük, gerçek robot güvenliği veya literatürde ilk yöntem iddiası desteklenmez. Ayrıntılı sayısal tablolar deney çıktılarında; kapsam [LIMITATIONS](LIMITATIONS.md) belgesindedir.
