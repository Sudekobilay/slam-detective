"""Yüksek bireysel LiDAR NIS değerinde kilitlenen deneysel güncelleme kapısı.

Sensör arızası teşhis etmez. Bir kez tetiklenince deney sonuna kadar bütün
LiDAR düzeltmeleri ve yeni çizgi eklemeleri askıya alınır. Hareket tahmini
ve istenmişse GPS güncellemeleri devam eder.
"""

import math


class LidarUpdateHold:
    def __init__(self, nis_threshold=9.21034):
        if not math.isfinite(nis_threshold) or nis_threshold <= 0:
            raise ValueError('NIS eşiği pozitif ve sonlu olmalı.')
        self.nis_threshold = float(nis_threshold)
        self.active = False
        self.trigger_time_s = None
        self.trigger_evidence = []
        self._last_time_s = None

    def observe(self, time_s, evidence):
        if (
            not math.isfinite(time_s)
            or time_s < 0
            or (self._last_time_s is not None and time_s <= self._last_time_s)
        ):
            raise ValueError('Tarama zamanı sonlu, pozitif sıralı ve tekil olmalı.')
        validated = []
        for item in evidence:
            nis = float(item['nis'])
            if not math.isfinite(nis) or nis < 0:
                raise ValueError('NIS negatif olmayan ve sonlu bir sayı olmalı.')
            validated.append({
                'sensor_id': item['sensor_id'],
                'observation_index': item['observation_index'],
                'nis': nis,
            })
        # Geçersiz girdi, durumun kısmen değişmesine yol açmaz.
        self._last_time_s = time_s
        if self.active:
            return False
        exceeded = [item for item in validated if item['nis'] > self.nis_threshold]
        if not exceeded:
            return False
        self.active = True
        self.trigger_time_s = time_s
        self.trigger_evidence = exceeded
        return True
