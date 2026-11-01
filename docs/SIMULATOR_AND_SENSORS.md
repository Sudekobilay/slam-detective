# Simülatör ve sensör modelleri

[Dokümantasyona dön](README.md)

## Hazır simülatör yerine ne kullanıyoruz?

Python ile yazılmış 2B deney ortamı kullanılır. Duvarlar doğru parçaları, engeller dikdörtgen sınırlarıdır. Robotun kinematik gerçek konumu ayrı tutulur. LiDAR ışınları bu geometriyle kesiştirilir. ROS, Gazebo veya Nav2 gerekmez.

Diferansiyel sürüşte tekerlek hızlarından gövde hızları:

$$v=(r_R\omega_R+r_L\omega_L)/2,\qquad \omega=(r_R\omega_R-r_L\omega_L)/b.$$

Burada tekerlek yarıçapları metre, açısal hızlar rad/s, tekerlek açıklığı metre cinsindendir. Kinematik model düz ve sabit eğrilikli hareketleri üretir; fizik motoru veya ayrıntılı lastik-zemin modeli değildir.

## Sensörler

| Sensör | Ölçüm / ayar | İncelenen kusur |
| --- | --- | --- |
| Encoder | Tekerlek sayımları, çözünürlük, yarıçap varsayımı | Kuantalama ve tekerlek yarıçapı uyuşmazlığı |
| Gyro | Düzlemsel açısal hız, gürültü ve bias | Yön entegrasyonunda sapma |
| GPS | Gürültülü x/y, kullanılabilirlik | Konum sapması ve belirli zaman aralığında kesinti |
| LiDAR | Işın menzili/noktalar; montaj, FOV, menzil, ışın sayısı | Fiziksel montaj yönü ile bilinen montajın uyuşmaması |

Gyro modeli tam 6 eksenli IMU değildir. LiDAR örneklerinin ideal geometrisi, gerçek optik etki ve hareketli nesne çeşitliliğini kapsamaz. Ana sonuçlar gürültülü gerçek LiDAR altında doğrulanmış sayılmaz.

## Çoklu LiDAR

`config/sensors/` altındaki JSON düzenleri sensör kimliği, robot üzerindeki x/y konumu, yaw, görüş açısı, ışın sayısı ve menzili belirtir. Tek, çift ve dört LiDAR örneği vardır. Bunlar her karşılaştırmada eşit toplam ışın bütçesi anlamına gelmez.

Montaj dönüşümü sensör noktasını robot gövdesine, robot pozu da dünya koordinatına taşır. Montaj hatası deneyinde fiziksel tarama hatalı montajla üretilir; kestiriciye nominal montaj verilir. Böylece hata etiketi algoritmaya doğrudan bildirilmez.

## Ortamlar

- İlk geliştirme ortamı: 10×8 m sınır ve iki dikdörtgen engel.
- EXP-42 koridoru: uzun, engel içermeyen dikdörtgen koridor.
- EXP-42 ikinci ortamı: iki asimetrik engelli oda.

Son iki ortam kalibrasyon ortamından ayrıdır. Her adayın gerçek geometride çarpışmasız olması deney ön kontrolüdür. Çevrim içi seçicide yalnız tahmini çizgilerle açıklık kontrolü vardır; bilinmeyen alan için güvenlik garantisi yoktur.

## Ortak zaman kaydı

İlk ortak akışta encoder/gyro 10 Hz, GPS 5 Hz, LiDAR başına 2 Hz kullanıldı. Ölçümler kaynak kimliği ve zaman damgasıyla saklanır. GPS kesinti paketi konum ölçümü değildir. Saat kayması, taşıma gecikmesi ve tarama içi hareket bozulması modellenmiş sayılmaz.

Ölçüm akışı ile `ground_truth.csv` ayrıdır. Gerçek durum yalnız üretim ve çevrimdışı değerlendirmede kullanılır.
