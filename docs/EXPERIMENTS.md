# Deneyler ve kanıt planı

[Dokümantasyona dön](README.md)

Deney numaraları geliştirme tarihçesini korur. Bir numaranın çok sayıda zaman damgalı klasörü olması normaldir: her klasör ayrı çalıştırmadır. Sonradan eklenen deney önceki çıktıları değiştirmez.

## Geliştirme aşamaları

| Deneyler | Ne kuruldu / sınandı? | Araştırmadaki işlev |
| --- | --- | --- |
| EXP-01–10 | Robot, ortam, çarpışma, LiDAR montajı, encoder, gyro, GPS, ortak kayıt | Ölçüm üretiminin doğrulanması |
| EXP-11–17 | Odometri, GPS EKF, hata enjeksiyonu, ICP, çizgi çıkarımı/eşleştirme | Ham ölçümden kestirime hazırlık |
| EXP-18–23 | Çizgi EKF-SLAM, GPS koruması, kaynak inceleme, LiDAR güncelleme kilidi | Haritalama ve tutarsızlık altyapısı |
| EXP-24–29 | Düz/yay dalları, donmuş referans, çevrim içi politika ve pencere kontrolleri | Eşleştirilmiş hareket karşılaştırması |
| EXP-30–33 | Ham tarama izleyicisi, dört politika ve hipoteze bağlı seçim | Ana teşhis döngüsü |
| EXP-34–35 | Kaynak çeşitleri ve GPS izolasyonu | Alarm ile kaynak doğruluğunu ayırma |
| EXP-36–38 | JSON düzenler, FastSLAM çekirdekleri ve ortak kayıt karşılaştırması | Sensör/algoritma çeşitliliği |
| EXP-39–40 | Bütünleşik ve yapılandırılabilir aktif akış | Ana kullanım yolu |
| EXP-41 | Gerçek FLASER/odometri yeniden oynatımı | Giriş ve haritalama uygulanabilirliği |
| EXP-42 | İki ek ortamda sabit kalibrasyonla dört politika | Ana sorunun ek geometrilerde değerlendirilmesi |

## Sunum için seçilecek deneyler

1. EXP-40 sağlıklı/hatalı dört LiDAR örneği: sistemin karar döngüsünü gösterir.
2. EXP-42: hareketin teşhise etkisini koşullar ve başarısızlıklarla birlikte inceler.
3. EXP-38: algoritma çeşitliliğini gösterir; aktif teşhis üstünlüğünü kanıtlamaz.
4. EXP-35: GPS korumasını gösterir; GPS için aktif hareket seçimi değildir.
5. EXP-41: gerçek veri bağlantısını gösterir; gerçek robotta aktif deney değildir.

## Eşleştirilmiş karşılaştırma

Başlangıç, sensör/hata koşulu ve tohum eşleştirilir. İlk alarma kadar ölçüm ve gerçek hareket geçmişi aynı olmalıdır. Politikalar ayrıldıktan sonra ölçümlerin aynı kalması beklenmez; farklı hareketler farklı çevre görür. Bu ayrışma kontrol edilen deneyin parçasıdır.

## EXP-42 tam tasarım

2 ortam × 2 normal rota × 3 hata büyüklüğü × 4 politika × 10 tohum = **480 politika çalıştırması**. Dört politika 120 eşleştirilmiş temel koşulu paylaşır. Eski ortamın sağlıklı 11–15 tohumları kalibrasyon; yeni ortamların 1201–1210 tohumları değerlendirmedir. GPS kapalı, çekirdek EKF, düzen dört LiDAR'dır.

Tam kaynaklar: `protocol.json`, `calibration.json`, `trials.csv`, `group_summary.csv`, `paired_summary.csv`, `summary.json`. [Ayrıntılı protokol ve komutlar](heldout_environments.md).

## Metrikleri karıştırmayın

Alarm oranı, doğru kaynak doğrulaması, tam destekli pencere, model maliyet farkı ve konum RMSE'si farklıdır. Düşük konum hatası tek başına doğru kaynak teşhisi değildir. İki adayın geometrik farkı da doğruluk olasılığı değildir.
