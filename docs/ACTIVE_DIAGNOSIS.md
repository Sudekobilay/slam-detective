# Hata teşhisi ve hareket seçimi

[Dokümantasyona dön](README.md)

## Üç farklı soru

| Aşama | Soru | Çıktı |
| --- | --- | --- |
| Alarm | Ölçümler arasında destekli tutarsızlık var mı? | Alarm zamanı veya doğrulanmamış durum |
| Kaynak teşhisi | Tanımlı hata modellerinden hangisi ölçümü daha iyi açıklıyor? | Aday, belirsiz veya doğrulanan kaynak |
| Hareket seçimi | Hangi kısa hareket rakip modelleri daha çok ayırabilir? | Düz/yay ya da uygun aday yok |

Birinci aşamanın başarılı olması diğer ikisinin başarılı olduğu anlamına gelmez.

## Alarm ve harita koruması

Çizgi yeniliği üzerinden NIS hesaplanabilir:

$$\operatorname{NIS}=r^TS^{-1}r.$$

Yalnız kabul edilen eşleşmeleri izlemek, reddedilmiş ölçümlerdeki hatayı kaçırabilir. Bu nedenle ham tarama hizalamasındaki göreli açı ve uyum da izlenir. Yapılandırılabilir izleyici destekli sensör çiftlerini ayrı takip eder; aynı çiftte tutarlı işaretli sapmanın devamı alarmı doğrular.

Şüphede LiDAR düzeltmeleri ve yeni harita çizgisi ekleme askıya alınabilir. Robot hareket tahminini sürdürür. Amaç, hatalı ölçümün haritaya işlenip teşhis kanıtını değiştirmesini azaltmaktır. Bu koruma tek başına kaynağı belirlemez.

## Rakip hata modelleri

Her destekli LiDAR için montaj yaw hatası modeli ve ortak robot yön sapması modeli karşılaştırılır. Ham noktaların aynı tahmini sonlu çizgi haritasına uyumu, açı taramasıyla değerlendirilir. Tek-hata modelleri bütün olası kusurları kapsamaz.

Uyum maliyeti ve modeller arası fark geometrik ölçülerdir; posterior olasılık değildir. Destek yetersizse veya fark küçükse sistem belirsiz kalabilir. Tek LiDAR'da mevcut sensörler arası ayrım yöntemi desteklenmez.

## Adaptif hareket

En iyi iki model seçilir. Her model altında robotun kısa düz ve yay hareketinden sonra görmesi beklenen menziller, mevcut tahmini haritada hesaplanır. Ortak ışın desteği ve bilinen haritada açıklık koşulları sağlanan adaylar karşılaştırılır. Ayrımı yüksek aday seçilir.

Puan, modellerin öngördüğü menziller arasındaki kırpılmış karesel farktan türetilir. Belirsizlik yayılımını içeren formel bilgi kazancı değildir. Normal rota zaten seçilen hareketi kullanıyorsa seçim komut değişikliği yaratmaz.

## Dört politika

- **Pasif:** normal komutları sürdürür; alarm ve harita koruması yine vardır.
- **Sabit düz:** alarm sonrası aynı düz hareketi uygular.
- **Sabit yay:** alarm sonrası aynı yay hareketini uygular.
- **Adaptif:** mevcut model ve haritaya göre aday seçer.

## EXP-42'de kaynak doğrulaması

Bu deney, hareket puanından ayrı bir birincil sonuç kullanır: alarm sonrası iki saniyede aynı kaynak ve aynı açı işareti iki ardışık destekli taramada aday olmalıdır. Eşikler yalnız eski ortamın sağlıklı kayıtlarından dondurulur. İlk doğrulanan yanlış kaynak sonraki iyi kareyle düzeltilmiş başarı olarak sayılmaz.

Alarm yokluğu, erken alarm, eksik pencere, belirsizlik ve yanlış kaynak başarı sayılmaz. Bu okuma EXP-42 protokolüne aittir; bütün eski deneylerin çıktılarının aynı anlamda teşhis oranı olduğu varsayılmaz.

## GPS ayrımı

GPS tutarlılığı ayrı GPS'siz referansla kontrol edilir ve şüpheli ölçümler füzyondan dışlanabilir. Yalnız GPS hatası için yeni hareketin yararını hesaplayan öteleme modeli uygulanmamıştır. Bu durumda adaptif akış desteklenen aday olmadığını bildirip durabilir.
