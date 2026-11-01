# Proje tanımı ve araştırma sorusu

[Dokümantasyona dön](README.md)

## Problem

Mobil robot, encoder ve gyro ile hareketini tahmin eder; LiDAR ile çevre çizgilerini gözlemler; GPS varsa konum ölçümünden yararlanır. Ölçümlerin uyuşmaması bir sorun belirtisidir. Ancak bu belirti tek başına hangi sensörün veya tahminin hatalı olduğunu açıklamaz.

Örneğin LiDAR'ın montaj yönündeki bir hata ile robotun yön tahminindeki hata, bazı görüntülerde benzer geometrik kaymalar oluşturabilir. Hatalı ölçümler haritaya işlendiğinde harita da hatayı öğrenebilir; sonraki ölçümler bozulmuş haritayla uyumlu görünmeye başlayabilir.

## Amaç

SLAM-Detective, ölçüm tutarsızlığını izleyen, tanımlı hata modellerini karşılaştıran ve bu modelleri ayırmaya yardımcı olması beklenen kısa bir hareket seçen bağımsız Python deney ortamıdır. Hazır robot simülatörü kullanılmaz.

> Robot, normal hareketinde ayırt edemediği bir sensör hatasını, uygun bir teşhis hareketinden sonra daha iyi ayırt edebilir mi?

## Araştırma katkısının kapsamı

1. Ham LiDAR ölçümlerini çizgi çıkarımı ve veri eşleştirme üzerinden SLAM'e bağlamak.
2. Alarm, kaynak teşhisi ve hareket seçimini ayrı aşamalar olarak değerlendirmek.
3. Tahmini harita üzerinde rakip hata modellerinin gelecekteki ölçümlerini karşılaştırmak.
4. Adaptif seçimi pasif ve iki sabit hareket politikasıyla eşleştirilmiş koşullarda sınamak.
5. Başarısız, belirsiz ve desteksiz durumları sonuçlardan çıkarmamak.

Bu, proje içindeki tasarım ve deney katkısıdır. Literatürde ilk yöntem olduğu veya yeni bir SLAM algoritması geliştirildiği iddia edilmez.

## Uygulanan sistem

| Bileşen | Uygulama |
| --- | --- |
| Ortam | 2B duvarlar ve dikdörtgen engeller, çarpışma kontrolü |
| Robot | Diferansiyel sürüş kinematiği |
| Sensörler | Yapılandırılabilir LiDAR, encoder, düzlemsel gyro, GPS |
| Haritalama | Çizgi EKF-SLAM, çizgi temelli FastSLAM 1.0/2.0 uyarlamaları |
| Teşhis | Ham tarama tutarlılığı ve geometrik hipotez karşılaştırması |
| Karar | Kısa düz ve yay adayları arasında seçim |
| Gerçek kayıt | CARMEN FLASER + kayıtlı odometri yeniden oynatımı |

## Tamamlanan ana kanıt

EXP-42'de dört LiDAR ve EKF ile iki yeni geometri incelendi. Koridorda normal rota yay iken +5° ön LiDAR hatasında pasif yöntem 10 denemenin tamamında belirsiz kaldı; adaptif yöntem düz hareket seçerek 10/10 doğru kaynak doğrulaması sağladı. Sabit düz de 10/10 verdi. Bu nedenle sonuç, hareketin katkısını destekler; adaptif yöntemin bütün sabit politikalardan üstünlüğünü göstermez.

Sayısal kapsam ve olumsuz sonuçlar [bulgular](RESEARCH_FINDINGS.md) ve [sınırlar](LIMITATIONS.md) belgelerinde birlikte verilir.
