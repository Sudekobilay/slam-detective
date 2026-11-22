# Mimari düzenleme — 0.2.0

Bu sürüm algoritma eşiklerini veya araştırma hedefini değiştirmez. Ölçüm sınırı, modül bağımlılıkları ve çalıştırma/kayıt düzeni yenilenmiştir.

## Yeni sınırlar

| Alan | Sorumluluk |
| --- | --- |
| domain | Paylaşılan poz/geometri, ölçüm değerleri, paket doğrulama, backend ve karar sözleşmeleri |
| configuration | Sensör düzeni ve benchmark şeması; protokol sabitleri |
| simulator | Dünya/sensör üretimi; başlangıç ve ortam kurucudan verilir |
| estimation | Üç backend, kendine ait geometri ve GPS referans dağılımı |
| application | Bileşen kurulumu, sağlıklı kalibrasyon, ortak koşu döngüsü ve CLI |
| evaluation | Kaynak doğrulama, eşleştirme, yörünge ve hareket maliyeti |
| storage | Atomik dosyalar, çalışma manifesti ve isteğe bağlı kayıtçı |
| experiments | Tarihsel deney komutları ve protokol girişleri |
| analytics / visualization | Ortak sonuçları okuyup sunar; numaralı deneylerden işlev almaz |

Eski importlar ve komutlar uyumluluk için korunur. Tarihsel deneylerin bütün uzun çizim/main fonksiyonları yeniden yazılmamıştır; güncel genişletilmiş akış daha küçük sorumluluklara ayrılmıştır.

## Ölçüm işleme sözleşmesi

Bir anda gelen bütün paketler, filtre/izleyici/geçmiş değiştirilmeden doğrulanır. Bilinmeyen tür/kimlik, eksik alan, uyumsuz zaman, geçersiz sayısal değer ve LiDAR boyut/nokta uyuşmazlığı reddedilir. Doğrulama hatası düzeltilip aynı grup tekrar gönderilebilir.

Doğrulamadan sonraki beklenmeyen sayısal veya callback hatası instance'ı FAILED yapar; komut sıfırlanır ve sonraki işleme kapatılır. Otomatik tam rollback garantisi yoktur; yeni instance gerekir. Keyfi callback'in iç mutasyonları veya dış nesnelerin eşzamanlı değişmesi desteklenmez.

Zaman sözleşmesi eşzamanlı encoder/gyro çiftidir. Gecikmeli/asenkron ölçüm füzyonu eklenmemiştir.

## Kurulum ve komutlar

Proje kökünde Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe main.py --help
.\.venv\Scripts\python.exe main.py robustness --smoke
```

Kurulum sonrası `slam-detective` komutu da aynı girişe yönlenir:

```powershell
slam-detective robustness --smoke --output .\results\architecture_check
slam-detective report --run .\results\architecture_check
slam-detective policy-demo --run .\results\architecture_check
```

Çıktı klasörü yeni olmalıdır. Aynı klasöre kör devam/üzerine yazma yapılmaz. Tam çalışma için --smoke kaldırılır ve başka çıktı klasörü seçilir. Varsayılan çıktı çağrılan çalışma dizininin results klasöründedir; paket kurulum dizinine yazılmaz. Referans JSON ayarları kurulan pakete dahildir; --config ile dış dosya verilebilir.

Eski `python -m experiments.exp_43_robustness` ve `python -m analytics.robustness_report` komutları da korunur. main.py global sys.argv değiştirmez.

## Çalışma durumları

Yeni genişletilmiş akışın run_manifest.json dosyası RUNNING, COMPLETED, PARTIAL veya FAILED durumunu taşır. Kaynak/config hash'leri, bağımlılık/Python sürümleri ve bulunabilirse git commit/dirty bilgisi kaydedilir. ZIP kurulumunda commit bulunmayabilir; kaynak hash'leri yine kaydedilir.

Manifest atomik değiştirilir. Deneme dosyaları yazıldıktan sonra case_manifest.json yazılır. Ctrl+C manifesti PARTIAL yapar. Otomatik resume eklenmemiştir. Dosya sistemi yazmayı reddederse son durumun kaydedilmesi garanti değildir.

Eksik ufuk varsayılan olarak başarısız CLI çıkışı üretir. --allow-partial açıkça seçilirse çıkış başarılı olabilir ama manifest PARTIAL kalır. Raporlama kısmi kayıtları ancak --allow-partial ile kabul eder. GIF tamamlanmış kayıt ister. Başarısız bilimsel sonuçlar sessizce paydadan çıkarılmaz.

Eski kayıtlar manifest olmadan okunabilir; onların kayıp çalışma kimliği sonradan uydurulmaz.

## Kayıt ve bellek

Varsayılan MemoryRecorder mevcut araştırma davranışını korur. ActivePipeline'a JsonlRecorder verilirse dört denetleyici geçmişinin tamamı diske, yalnız son N satırı RAM'e yazılır. Simülatör, backend ve guard geçmişleri ayrı ve deney boyutundadır. Bu değişiklik tüm sistemi sınırsız süre çalıştıran bir streaming sistem yapmaz. Kalibrasyon/benchmark için varsayılan tam geçmiş kullanılmalıdır.

## Sürüm ve doğrulama

Bu mimari sürüm yeni bir bilimsel başarı iddiası getirmez. CI dosyası Windows/Linux ve Python 3.12/3.13 için hazırlanmıştır; GitHub'a push edilmeden hosted CI çalışmış sayılmaz. Yerel paket kurulumu, mevcut bağımlılıkların tekrar kullanıldığı ayrı ortamda sınanmıştır; tamamen internetsiz/sıfır bağımlılıklı temiz makine testi değildir.
