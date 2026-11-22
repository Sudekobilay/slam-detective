# Kurulum ve ilk çalıştırma

> Güncel mimari ve yeni ortak CLI için [0.2.0 düzenleme notlarına](ARCHITECTURE_REFACTOR.md) bakın. Aşağıdaki tarihsel açıklamalar önceki deney akışlarını da içerir.

[Dokümantasyona dön](README.md)

Komutları VS Code terminalinde proje kökünde çalıştırın. Kullanıcı geliştirme ortamı Windows PowerShell ve Python 3.13.3'tür. Mevcut `.venv` varsa yeniden oluşturmayın.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy matplotlib pillow
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe main.py --help
```

NumPy sayısal işlemleri, Matplotlib grafikleri, Pillow GIF çıktısını sağlar. PDF sonuç kitapçığı istenirse ayrıca `reportlab` gerekir.

## 1. Sensör düzenlerini hazırlama

Örnek JSON dosyaları henüz yoksa:

```powershell
.\.venv\Scripts\python.exe main.py layouts
```

`single.json`, `dual.json`, `quad.json` örneklerini yalnız yoklarsa oluşturur. Mevcut dosyalar değiştirilmez. Bu komut aynı zamanda temel yerleşim deneyini yürütür.

## 2. Hatasız kontrol

```powershell
.\.venv\Scripts\python.exe main.py active --config config/sensors/quad.json --fault healthy --no-gps
```

Varsayılan örnekte alarm ve seçilen hareket `None` olur. Bu, yalnız o denemede doğrulanmış tutarsızlık olmadığını ifade eder.

## 3. Hatalı sensör ve hareket seçimi

```powershell
.\.venv\Scripts\python.exe main.py active --config config/sensors/quad.json --fault-sensor left --no-gps
```

EXP-40 çalışır. Varsayılan EKF/quad örneğinde hata t=2 s'de başlar; kontrol edilmiş örnekte alarm 2.5 s'de oluşur ve yay seçilir. Farklı ayarların aynı sonucu vermesi zorunlu değildir.

## 4. Kayıttan hareketli gösterim

```powershell
.\.venv\Scripts\python.exe main.py active-demo --run "results\EXP40_KLASORU"
```

Yer tutucuyu terminalde yazdırılan gerçek klasör adıyla değiştirin. Bu, kayıtlı deneyi gösterir; başka politika altında yeni hareket sonucu üretmez.

## 5. Ana araştırma değerlendirmesi

```powershell
$env:OPENBLAS_NUM_THREADS = "1"
.\.venv\Scripts\python.exe -m experiments.exp_42_heldout_environments --save-recordings
.\.venv\Scripts\python.exe -m analytics.heldout_environment_results --run "results\EXP42_KLASORU"
```

Tam protokol 480 politika çalıştırmasıdır. Kısa bağlantı kontrolü `--seeds 1201` ile yapılabilir; bu tam değerlendirme değildir. Ayrıntılar: [EXP-42](heldout_environments.md).

## Diğer görevler

| Komut | Amaç |
| --- | --- |
| `python main.py compare` | Üç SLAM çekirdeği, EXP-38 |
| `python main.py gps` | GPS tutarlılığı ve korumalı füzyon, EXP-35 |
| `python main.py real --download` | Gerçek kaydı indirip yeniden oynatma, EXP-41 |
| `python -m analytics.summarize_results --pdf` | Seçilmiş sonuçlardan kitapçık; reportlab gerekir |

Tablodaki `python` yerine sanal ortamın Python yolunu kullanın. Her deneyin çalışma süresi ve kapsamı farklıdır. Tüm eski deneyleri yeniden çalıştırmak ilk kullanım koşulu değildir.
