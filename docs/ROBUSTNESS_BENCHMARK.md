# Robustness ve politika gösterimi - EXP-43

Bu geliştirme 1 Kasım 2026 tarihli gönderilmiş proje ZIP'i üzerinde uygulanıp doğrulanmıştır. Önceki deney ve kestirim çekirdekleri değiştirilmez. Yeni dosyalar eski modülleri kullanır; bunların yerine geçmez.

## Eklenen çalışma

- Aynı koşulda dört politikanın kayıttan yan yana GIF gösterimi.
- Üç farklı konum/yön başlangıcı. Kestiriciye aynı doğru başlangıç öncülü verilir; başlangıç belirsizliği çözülmüş değildir.
- İdeal ölçüm ve 1 cm standart sapmalı Gaussian radyal menzil gürültüsü + %1 bağımsız ışın kaybı.
- 0°, +2°, +5° ön LiDAR montaj koşulları. JSON değiştirilerek -10°..+10° aralığında farklı açılar seçilebilir; varsayılan çalışmada negatif açılar değerlendirilmiş sayılmaz.
- Eşleştirilmiş doğru kaynak, yanlış alarm, belirsizlik, teşhis gecikmesi, ortak ufuk mesafesi ve mutlak dönüş karşılaştırması.

## Sabit protokol

2 ortam × 2 normal rota × 3 başlangıç × 2 ölçüm profili × 3 hata × 3 tohum × 4 politika = **864 politika çalıştırması**.

216 temel koşul, aynı alarm öncesi geçmişi paylaşan dört politikayla değerlendirilir. Üç gürültü tohumu betimsel tekrar sağlar; güçlü istatistiksel genelleme iddiası değildir.

Alarm ve kaynak doğrulama eşikleri, EXP-42'nin özgün ortamındaki sağlıklı/ideal kayıtlarla belirlenir ve dondurulur. Test ortamındaki sonuçlara göre eşik, sensör kovaryansı veya aday puanı ayarlanmaz. Bu nedenle gürültülü koşul bir **dağılım değişimi stres testidir**; gürültüye göre kalibre edilmiş bir yöntemin başarısı diye sunulmaz. Filtrede gürültü kovaryansı uyarlaması eklenmemiştir.

Gerçek ortam ve hata etiketi yalnız ölçüm üretimi ve çevrimdışı değerlendirmededir. Başlangıç pozu yalnız t=0 öncülü olarak verilir; çalışma boyunca gerçek poz geri beslemesi yoktur.

## Dosyaları yerleştirme

| Proje içindeki yol | İşlev |
| --- | --- |
| simulator/lidar_corruption.py | Tohum/sensör/zamanla anahtarlanan menzil gürültüsü ve dropout |
| config/benchmarks/robustness.json | Başlangıçlar, gürültü, hatalar ve değerlendirme tohumları |
| experiments/exp_43_robustness.py | Kalibrasyon, dört politika, eşleştirme ve ölçüm kayıtları |
| analytics/robustness_report.py | Başarı tablosu ve eşleştirilmiş hareket maliyeti grafikleri |
| visualization/policy_comparison.py | Kaydedilmiş dört politikanın GIF gösterimi |
| tests/test_robustness_extension.py | Gürültü, zaman/sensör eşleştirme, başlangıç ve ölçüt kontrolleri |
| docs/ROBUSTNESS_BENCHMARK.md | Bu açıklama |

`main.py` değiştirilmez. Yeni dosyalar doğrudan modül komutlarıyla çalıştırılır. `.gitignore` içinde `results/` bulunması sayısal kayıtların GitHub'a otomatik eklenmesini önler.

## Çalıştırma

Proje kökünde PowerShell:

```powershell
$env:OPENBLAS_NUM_THREADS = "1"
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_robustness_extension.py -v
.\.venv\Scripts\python.exe -m experiments.exp_43_robustness --smoke
```

Kısa kontrol 24 çalıştırmadır (koridor, nominal başlangıç, normal yay, bir tohum). Tam değerlendirme değildir.

Tam protokol:

```powershell
.\.venv\Scripts\python.exe -m experiments.exp_43_robustness
```

Ham sensör kayıtlarını da saklamak için `--save-recordings` eklenebilir. Ham kayıtlar olmadan da yörüngeler, komutlar, ajan geçmişi ve hipotezler saklanır.

Tamamlanan sonuç klasörünün adı kullanılır:

```powershell
.\.venv\Scripts\python.exe -m analytics.robustness_report --run "results\EXP43_KLASORU"
.\.venv\Scripts\python.exe -m visualization.policy_comparison --run "results\EXP43_KLASORU"
```

Gösterimin varsayılan koşulu koridor / nominal / ideal / yay / ilk tohum / +5°'dir. Bu eski ana sonuçla karşılaştırılabilir örnektir; bütün koşulların başarısını temsil etmez.

Gürültülü gösterim:

```powershell
.\.venv\Scripts\python.exe -m visualization.policy_comparison --run "results\EXP43_KLASORU" --noise noisy
```

Başka başlangıç için `--start shifted` veya `--start rotated`; başka rota için `--normal straight`; başka hata için `--yaw 2`; başka tohum için `--seed 2202` kullanılabilir. Koşulun ilgili sonuç klasöründe gerçekten bulunması gerekir.

## Ölçüler

Doğru kaynak: EXP-42'nin aynı model ve aynı açı işaretini iki ardışık destekli post-alarm taramada doğrulama kuralı. Alarm yokluğu, erken alarm, eksik pencere, belirsizlik ve yanlış kaynak paydadan çıkarılmaz.

Teşhis gecikmesi: doğrulama zamanından alarm/hata başlangıcının çıkarılması. Doğrulama yoksa null bırakılır; sıfır saniye kabul edilmez.

Hareket maliyeti: bütün yöntemlerde ortak 0-4 saniye gerçek yol mesafesi ve mutlak dönüş. Adaptif eksi karşılaştırıcı farkları aynı temel koşulda hesaplanır. Eşit hızlı ve eşit süreli adaylarda mesafe farkı sıfır olabilir; bu bir hata değildir. Değişen komut süresi ayrıca kaydedilir.

Maliyet seçicide optimize edilmez; sonradan ölçülür. Daha yüksek geometrik aday puanı daha düşük hareket maliyeti garantisi vermez. Başarısız yöntem daha az hareket etmiş olsa bile başarı/maliyet üstünlüğü iddia edilmez.

## Tekrarlanabilirlik

Gürültü çekilişleri tohum, sensör kimliği, zaman ve ışın indeksi ile eşleştirilir. Politikaların farklı ilerlemesi, ortak zamanlardaki rastgele sayı tüketimini kaydırmaz. Aynı koşul ve tohum tekrarlandığında aynı ölçümler elde edilir. Noktalar bozulmuş menzillerden yeniden hesaplanır; ideal noktalar kestiriciye sızdırılmaz.

Her koşul için bütün aday dalları gerçek geometride ön kontrolden geçer. Çevrim içi seçici yine yalnız kendi tahmini haritasını kullanır. Gerçek dünyada güvenli rota garantisi yoktur.

Grafik/gösterim araçları kaynak dosya SHA-256 değerlerini önce/sonra kontrol eder. GIF, kayıtlı komut ve ölçümlerden oluşan yörüngeleri gösterir; sonradan yeni hareket seçmez. Alarm ve kaynak sonucu zamana göre açıklanır; gelecekteki sonuç önceki karelerde gösterilmez.

## GitHub sunumu

Yeni ana sayfa bölümü için kendi sonuç grafikleri `docs/assets/` içine sabit adlarla kopyalanabilir. `success.png` bütün koşulları; `paired_costs.png` maliyeti; `policy_comparison.gif` belirli bir örnek koşulu gösterir. Örnek GIF'in açıklamasına koşul ve tohum yazılmalıdır. Sayısal sonuçlar görülmeden başarı/üstünlük rozeti eklenmez.

## Uyumluluk ve bağımlılıklar

NumPy ve Matplotlib önceki projede vardır; GIF için Pillow gerekir:

```powershell
.\.venv\Scripts\python.exe -m pip install pillow
```

İlk geliştirme ortamı Python 3.13.3'tür. Bu eklentinin yerel kontrolü Python 3.12 üzerinde yapıldı; farklı sürümlerde sayısal toleranslar ve bağımlılık kurulumu ayrıca kontrol edilmelidir.

Grafikler ortam ve başlangıçları birlikte özetler. Ayrıntılı hücreler `stratified_summary.csv` içinde saklanır. Bu dosyada her hücrenin tekrar sayısı, bütün hata/başarı sınıfları ve destekli gecikme/maliyet ortalamaları vardır. Grafik toplaması, aynı ortamda çok sayıda bağımsız tekrar yapılmış gibi sunulmaz.
