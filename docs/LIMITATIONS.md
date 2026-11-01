# Kapsam ve sınırlamalar

[Dokümantasyona dön](README.md)

Sınırları belirtmek başarısızlığı gizlememek ve sonuçların hangi koşullarda geçerli olduğunu açıklamak içindir.

| Konu | Yapılan | Henüz gösterilmeyen / kapsam dışı |
| --- | --- | --- |
| Tek LiDAR | Haritalama ve yeniden oynatım | Mevcut çift-sensör kanıtıyla kaynak ayrımı |
| GPS | Sapma/kesinti, izleme ve korumalı füzyon | Yalnız GPS hatasında teşhis hareketi seçimi |
| Dört LiDAR + FastSLAM | Kestirim ve aktif akış bağlantısı | İncelenen +5° örnekte alarm başarısı; alarm oluşmadı |
| EXP-42 | İki ek ortam, EKF, dört LiDAR, tek ön yaw hatası | Bütün sensör kaynakları, GPS, FastSLAM ve gerçek robot genellemesi |
| Küçük hata | +2° koşulları raporlandı | EXP-42'de bu hataların yakalanması; alarm yoktu |
| Hareket seçimi | Düz/yay, tahmini haritada geometrik ayrım | Formel bilgi kazancı, bütün sabit politikalara üstünlük |
| Güvenlik | Gerçek geometrili deney ön kontrolü ve tahmini haritada açıklık | Bilinmeyen çevrede çarpışmama garantisi |
| IMU | Düzlemsel gyro | Tam ivmeölçer/6 eksen IMU modeli |
| Gerçek veri | LiDAR ve kayıtlı odometri | Gerçek GPS/IMU/çoklu LiDAR ve aktif hareket uygulaması |
| Doğruluk | Simülasyonda gerçek konuma göre RMSE | Bağımsız gerçek konumsuz kayıtta ATE/RMSE |

## İstatistiksel sınırlar

EXP-42'de hücre başına 10 tohum vardır. Dört politika aynı temel koşulu paylaştığından 480 yürütme 480 bağımsız kayıt değildir. Sonuçlar sınırlı ortam, başlangıç ve sensör modeli içindir. İstatistiksel anlamlılık veya nüfus geneli başarı garantisi sunulmaz.

Geometrik model farkı olasılık değildir. Desteksiz sonuç sıfır maliyet sayılmaz. Çizgi sayısı harita doğruluğu veya netliği yerine geçirilmez. İşlem süresi standart donanım benchmark'ı değildir.

## Algoritmik sınırlar

Çizgi çıkarımı ve açgözlü eşleştirme hata yapabilir. EKF yerel doğrusallaştırma kullanır. FastSLAM uyarlamaları sınırlı parçacık sayısı, sezgisel yeni özellik ağırlıkları ve çizgi modeli içerir. Büyük ölçekli döngü kapama veya üretim düzeyi gezinme sistemi geliştirildiği iddia edilmez.

## Araştırma sınırı

Bağımsız kod geliştirmek, literatürde yöntem yeniliğini tek başına kanıtlamaz. Sistematik literatür karşılaştırması ayrı iştir. Sonuçların güçlü tarafı açık protokol ve olumlu/olumsuz bulguların birlikte sunulmasıdır.
