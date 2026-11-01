# Yeniden üretilebilirlik ve kayıt düzeni

[Dokümantasyona dön](README.md)

## Çalıştırmadan önce

Proje kökünde sanal ortamı ve testleri doğrulayın:

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Gönderilen güncel ZIP ayrı bir çalışma kopyasında Python 3.12 ile kontrol edildi: 319 test geçti (13.987 s). Kullanıcının geliştirme sürümü Python 3.13.3’tür. Bu sayı yalnız kontrol edilen ZIP sürümüne aittir. Eski `records/test_sonucu.txt` içindeki 181 testlik çıktı tarihsel kayıttır. Test başarısı deneysel üstünlük anlamına gelmez.

## Sabitlenmesi gerekenler

- Kod sürümü/commit veya dosya özetleri.
- Kullanılan sensör JSON'u ve algoritma ayarları.
- Sensör, kalibrasyon ve parçacık tohumları; bunlar aynı şey değildir.
- Hata türü, büyüklüğü, başlangıç zamanı, normal rota ve politika.
- Kalibrasyon kayıtları, eşikler ve değerlendirme protokolü.
- Python, NumPy ve diğer bağımlılık sürümleri; süre için donanım bilgisi.

## EXP-42 yeniden üretimi

```powershell
$env:OPENBLAS_NUM_THREADS = "1"
.\.venv\Scripts\python.exe -m experiments.exp_42_heldout_environments --save-recordings
.\.venv\Scripts\python.exe -m analytics.heldout_environment_results --run "results\EXP42_KLASORU"
```

Tam koşul 480 politika yürütmesidir. `--save-recordings` yalnız saklanan ham kayıtları etkiler. Eşikler eski ortamın sağlıklı kayıtlarından elde edilir; test hatalarına bakılarak değiştirilmez. Çıktıdaki ortak geçmiş, tamamlanma ve kaynak koruma kontrolleri incelenir.

| Kayıt | İşlev |
| --- | --- |
| `protocol.json` | Deneme öncesi sabitlenen yerel tasarım |
| `calibration.json` | Kullanılan eşikler ve kalibrasyon bilgisi |
| `used_config.json` | Sensör/çalıştırma ayarları |
| `trials.csv` | Başarısız ve belirsiz dahil bütün politika sonuçları |
| `group_summary.csv` | Hücre bazında sonuç sayıları |
| `paired_summary.csv` | Aynı temel koşullardaki politika farkları |
| `summary.json` | Tamamlanma ve koruma denetimleri |

## Sonuç klasörleri

Her çalıştırma yeni `results/exp_XX_ZAMAN/` klasörü üretir. Önceki çıktıların güncellendiğini varsaymayın. Ana karşılaştırma için seçilen kayıtların açık yol listesini saklayın. En iyi sonucu veren tek klasörü seçmek yerine protokolün bütün koşullarını raporlayın.

`analytics.summarize_results` için kaynakları `selected_sources.json` ile tekrar sabitleyebilirsiniz. EXP-42 kendi ayrı analiz betiğini kullanır; iki aracın protokolleri karıştırılmaz.

## Gerçek veri ve paylaşım

`data/raw/` gitignore içinde kalır. Veri dosyasını depoya eklemek yerine kaynak, indirme ve SHA-256 açıklaması tutulur. Bağımsız gerçek konum yoksa kayıtlı odometri gerçek konum yerine kullanılmaz. Ayrıntı: [gerçek veri](real_data.md).

## Eski notlar

Terminal dökümleri ve denetim özetleri kaynak kanıt olarak korunabilir. Bunlar ana kullanım belgesinin yerine geçmez. Belgeleri ayırma işlemi sonuç dosyalarının içeriğini, eski kodu veya deney numaralarını değiştirmemelidir.
