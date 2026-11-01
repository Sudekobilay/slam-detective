# Konum ve harita kestirim algoritmaları

[Dokümantasyona dön](README.md)

## Ortak giriş: ham LiDAR'dan çizgiye

LiDAR noktaları çizgi çıkarımından geçirilir. Uygun destek ve uzunluk koşullarını sağlayan parçalar çizgi gözlemi olur. Eşleştirme, bu gözlemleri mevcut tahmini harita çizgileriyle ilişkilendirir. Desteksiz, belirsiz veya çatışmalı eşleştirmeler ayrı kaydedilir.

Simülatörün bildiği hazır duvar kimlikleri filtreye verilmez. Bununla birlikte çıkarım ve eşleştirme kusursuz değildir; hatalı ilişkilendirme konumu ve haritayı etkileyebilir. ICP deneyleri ayrı göreli tarama doğrulamasıdır; bütün SLAM sisteminin yalnız ICP ile çalıştığı söylenmez.

## EKF-SLAM

Robot durumu x, y, yön ve gyro bias bileşenlerini içerir; harita çizgileriyle birlikte ortak durum ve kovaryans tutulur. Encoder/gyro hareket tahminini ilerletir. Eşleşen çizgiler ölçüm düzeltmesi yapar; uygun yeni çizgiler duruma eklenir. Robot ile çizgilerin çapraz kovaryansları korunur.

Genel ölçüm düzeltmesinde:

$$r=z-h(\hat x),\quad S=HPH^T+R,\quad K=PH^TS^{-1}.$$

Burada yenilik r, beklenen ölçüm ile gözlem arasındaki farktır. Yön/çizgi açısı farkları açısal normalizasyona ihtiyaç duyar. Bu doğrusal olmayan modelde EKF yerel doğrusallaştırmadır; veri eşleştirme hatasını kendiliğinden çözmez.

## FastSLAM 1.0 uyarlaması

Her parçacık robot hareketi ve kendi çizgi haritasını taşır. Hareket önselinden örneklenir, ölçüm uyumuyla ağırlıklandırılır, gerekli durumda yeniden örneklenir. Harita belirsizliği parçacık koşullu çizgi kestirimleriyle temsil edilir.

## FastSLAM 2.0 uyarlaması

Öneri dağılımı yalnız hareket önselinden gelmez: uygun GPS veya seçilen eşleşmiş çizgi ile yerel Gaussian öneri koşullandırılabilir. Örnekleme sonrasında önsel/öneri oranı önem ağırlığına eklenir; ölçüm olabilirliği çift sayılmaz. Uygun ölçüm yoksa ölçümle koşullandırılmış öneri beklenmez.

Bunlar çizgi özelliklerine uyarlanmış deneysel çekirdeklerdir. Özgün makalelerin bütün ayrıntılarını, ağaç tabanlı karmaşıklık avantajını veya aynı performansı yeniden ürettikleri iddia edilmez. Yeni özellik ağırlıklarında sezgisel tercihler vardır.

## Karşılaştırma nasıl yapıldı?

EXP-38 aynı nominal ölçüm kayıtlarını üç çekirdeğe verir. Tek/çift/dört LiDAR ve GPS açık/kapalı koşulları kullanılır. EKF bir kez; parçacıklı yöntemler 901–903 parçacık tohumlarıyla çalışır. Parçacık tohumu, bağımsız sensör kaydı değildir.

İncelenen kayıtlarda EKF daha düşük konum RMSE'si verdi. FastSLAM 2.0 her koşulda FastSLAM 1.0'dan iyi çıkmadı. Bu, bütün ortamlar ve ayarlar için algoritma sıralaması değildir. Çizgi sayısı da doğrudan harita doğruluğu ölçüsü değildir.

## Kuramsal başvurular

- [FastSLAM teknik raporu](https://robots.stanford.edu/papers/montemerlo.fastslam-tr.pdf)
- [FastSLAM 2.0 makalesi](https://robots.stanford.edu/papers/Montemerlo03a.pdf)

Uygulama sınırları için [LIMITATIONS](LIMITATIONS.md), komutlar için [deney rehberi](experiment_guide.md).
