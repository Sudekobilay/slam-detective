# SLAM-Detective — Dokümantasyon

**Çoklu Sensörlü SLAM Sistemlerinde Hareket Seçimiyle Hata Teşhisinin İncelenmesi**

Bu klasör, projenin neden geliştirildiğini, bileşenlerin nasıl çalıştığını ve deneylerin ne gösterdiğini açıklar. Ana araştırma sorusu: **Robot, uygun bir kısa hareket seçerek sensör hatasının kaynağını daha iyi ayırt edebilir mi?**

## Önerilen okuma sırası

| Belge | Yanıtladığı soru |
| --- | --- |
| [Proje tanımı](PROJECT_OVERVIEW.md) | Hangi problemi araştırıyoruz, katkımız ne? |
| [Başlangıç](GETTING_STARTED.md) | Projeyi nasıl çalıştırırım? |
| [Mimari](ARCHITECTURE.md) | Hangi klasör ne yapar, veriler nasıl ilerler? |
| [Simülatör ve sensörler](SIMULATOR_AND_SENSORS.md) | Ölçümleri ve hataları nasıl üretiyoruz? |
| [SLAM algoritmaları](ALGORITHMS.md) | EKF ve FastSLAM çekirdekleri nasıl çalışır? |
| [Hata teşhisi ve hareket seçimi](ACTIVE_DIAGNOSIS.md) | Robot neye göre alarm verir ve hareket seçer? |
| [Deney planı](EXPERIMENTS.md) | Hangi deney hangi iddiayı sınar? |
| [Araştırma bulguları](RESEARCH_FINDINGS.md) | Sonuçlar neyi destekliyor? |
| [Sınırlar](LIMITATIONS.md) | Hangi sonuçları iddia etmiyoruz? |
| [Yeniden üretilebilirlik](REPRODUCIBILITY.md) | Sonuçları nasıl kontrol ederim? |

## Ayrıntılı başvuru belgeleri

- [Deney ve komut rehberi](experiment_guide.md): önceki deneylerin ayrıntılı açıklamaları.
- [Gerçek veri](real_data.md): Freiburg 079 kaynağı, biçim, indirme ve kapsam.
- [Ortak sonuç analizi](results_analysis.md): seçilmiş deneylerden tablo, grafik ve PDF üretimi.
- [Ek ortam değerlendirmesi](heldout_environments.md): EXP-42 protokolü, çalıştırma ve bulgular.

İlk okumada EXP-01'den EXP-42'ye kadar her şeyi çalıştırmanız gerekmez. Ana çevrim içi gösterim EXP-40, ana ek ortam değerlendirmesi EXP-42'dir. Deneysel araştırma prototipidir; genel hata teşhisi veya gerçek robot güvenliği garantisi vermez.

## Belge düzeni

Yukarıdaki konu belgeleri projenin güncel açıklamasıdır. Ayrıntılı deney rehberleri geçmiş geliştirme adımlarını da korur. Terminal çıktıları ve dosya özetleri açıklama metni yerine kanıt/kayıt olarak saklanır. Dosyaları ayırma talimatı için teslim klasöründeki `YERLESTIRME.md` belgesine bakın.


