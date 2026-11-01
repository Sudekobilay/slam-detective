"""EXP-42'nin bütün koşullarını değiştirmeden özetler; yeni deney çalıştırmaz."""
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiments.exp_42_heldout_environments import POLICIES, ENVIRONMENTS
from simulator.environment import Environment, Rectangle


NAMES={'passive':'Pasif','fixed_straight':'Sabit düz','fixed_arc':'Sabit yay','adaptive':'Adaptif'}
COLORS={'passive':'#627080','fixed_straight':'#cb7c32','fixed_arc':'#347f79','adaptive':'#345e98'}
ENV_NAMES={'corridor':'Koridor','offset_obstacles':'Asimetrik engeller'}


def markdown_table(headers,rows):
    def value(item):
        if item is None:
            return '-'
        if isinstance(item,float):
            return f'{item:.6g}'
        return str(item).replace('|','/')
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']
                      + ['| '+' | '.join(value(v) for v in row)+' |' for row in rows])


def csv_data(path):
    with path.open(newline='',encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def draw_routes(run,output,protocol):
    seed=protocol['evaluation_seeds'][0]
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for column,environment in enumerate(ENVIRONMENTS):
        geometry=protocol['environment_geometry'][environment]
        world=Environment(Rectangle(**geometry['boundary']),tuple(Rectangle(**r) for r in geometry['obstacles']))
        tracks={p:csv_data(run/environment/'arc'/f'seed_{seed}'/'yaw_5'/p/'ground_truth.csv') for p in POLICIES}
        for row in (0,1):
            axis=axes[row,column]
            for wall in world.walls:
                axis.plot([wall.start[0],wall.end[0]],[wall.start[1],wall.end[1]],color='black',lw=1.)
            for p,track in tracks.items():
                axis.plot([float(r['x_m']) for r in track],[float(r['y_m']) for r in track],color=COLORS[p],
                          ls='--' if p=='adaptive' else '-',lw=2.,label=NAMES[p])
            if row==0:
                b=world.boundary
                axis.set(xlim=(b.x_min-.2,b.x_max+.2),ylim=(b.y_min-.2,b.y_max+.2),title=ENV_NAMES[environment])
            else:
                points=[r for track in tracks.values() for r in track]
                axis.set(xlim=(min(float(r['x_m']) for r in points)-.2,max(float(r['x_m']) for r in points)+.2),
                         ylim=(min(float(r['y_m']) for r in points)-.2,max(float(r['y_m']) for r in points)+.2),title='Aynı başlangıçtan gerçek rotalar - yakın görünüm')
            axis.set_aspect('equal',adjustable='box')
            axis.set(xlabel='x (m)',ylabel='y (m)')
            axis.grid(alpha=.2)
            if row==1:axis.legend(fontsize=8)
    fig.suptitle(f'EXP-42: iki ek ortam | normal rota yay | ön LiDAR +5° | seed {seed}\nİlk doğrulama tohumu görselleştirilir; nicel sonuçlar bütün tohumları içerir.',fontsize=11)
    fig.savefig(output/'01_environments_and_routes.png',dpi=180,bbox_inches='tight');plt.close(fig)


def draw_primary(groups,output):
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for axis,(environment,normal) in zip(axes.flat,((e,n) for e in ENVIRONMENTS for n in ('straight','arc'))):
        for index,policy in enumerate(POLICIES):
            rows=[next(r for r in groups if r['environment']==environment and r['normal_motion']==normal
                       and r['fault_yaw_deg']==yaw and r['policy']==policy) for yaw in (2.,5.)]
            bars=axis.bar(np.arange(2)+(index-1.5)*.19,[r['correct_source_rate'] for r in rows],width=.18,color=COLORS[policy],label=NAMES[policy])
            for bar,r in zip(bars,rows):
                axis.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.025,f"{r['correct_source']}/{r['trials']}",ha='center',fontsize=8)
        axis.set_xticks([0,1],['+2°','+5°'])
        axis.set(ylim=(0,1.2),ylabel='Doğru kaynak doğrulanan deneme oranı',title=f"{ENV_NAMES[environment]} | normal rota {'düz' if normal=='straight' else 'yay'}")
        axis.set_yticks([0,.5,1],['0%','50%','100%']);axis.grid(axis='y',alpha=.2)
    axes[0,0].legend(fontsize=8,ncol=2)
    fig.suptitle('EXP-42: önceden sabitlenen teşhis kuralıyla bütün hatalı denemeler\nAlarm yok / belirsiz / yanlış / eksik pencere, başarılı kaynak teşhisi sayılmaz.',fontsize=11)
    fig.savefig(output/'02_correct_source_rates.png',dpi=180,bbox_inches='tight');plt.close(fig)


def draw_secondary(trials,output):
    fig,axis=plt.subplots(figsize=(11,4.5),constrained_layout=True)
    conditions=[(e,n) for e in ENVIRONMENTS for n in ('straight','arc')]
    for index,other in enumerate(('passive','fixed_straight','fixed_arc')):
        means,deviations,counts=[],[],[]
        for environment,normal in conditions:
            rows=[r for r in trials if r['environment']==environment and r['normal_motion']==normal and float(r['fault_yaw_deg'])==5.]
            by_policy={p:{r['seed']:r for r in rows if r['policy']==p} for p in ('adaptive',other)}
            values=[]
            for seed in set(by_policy['adaptive'])&set(by_policy[other]):
                a,b=by_policy['adaptive'][seed],by_policy[other][seed]
                if a['mean_true_source_margin_m2'] and b['mean_true_source_margin_m2']:
                    values.append((float(a['mean_true_source_margin_m2'])-float(b['mean_true_source_margin_m2']))*1e4)
            counts.append(len(values));means.append(statistics.mean(values) if values else float('nan'))
            deviations.append(statistics.stdev(values) if len(values)>1 else 0.)
        bars=axis.bar(np.arange(4)+(index-1)*.23,means,width=.22,yerr=deviations,capsize=3,
                      color=COLORS[other],label='Adaptif eksi '+NAMES[other].lower())
        for bar,n in zip(bars,counts):
            if np.isfinite(bar.get_height()):axis.annotate(f'n={n}',(bar.get_x()+bar.get_width()/2,bar.get_height()),xytext=(0,5 if bar.get_height()>=0 else -12),textcoords='offset points',ha='center',fontsize=8)
    axis.axhline(0,color='black',lw=.8)
    axis.set_xticks(range(4),[ENV_NAMES[e]+'\nNormal '+('düz' if n=='straight' else 'yay') for e,n in conditions])
    axis.set_ylabel('Eşleştirilmiş ortalama model farkı değişimi (×10⁻⁴ m²)')
    axis.legend(fontsize=8);axis.grid(axis='y',alpha=.2);axis.margins(y=.3)
    axis.set_title('İkincil ölçüt: +5° koşullarında geometrik puan farkı\nHata çizgileri örnek standart sapma; güven aralığı veya teşhis olasılığı değildir.',fontsize=10)
    fig.savefig(output/'03_paired_geometric_differences.png',dpi=180,bbox_inches='tight');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    source_files=[args.run/name for name in ('summary.json','protocol.json','calibration.json','trials.csv','paired_summary.csv')]
    summary=json.loads((args.run/'summary.json').read_text(encoding='utf-8-sig'))
    if summary.get('experiment_id')!='exp_42' or not summary.get('protected_sources_unchanged') or not summary.get('calibration_unchanged'):
        raise ValueError('Doğrulanmış EXP-42 kaydı gerekli.')
    protocol=json.loads((args.run/'protocol.json').read_text(encoding='utf-8-sig'))
    for environment in ENVIRONMENTS:
        for policy in POLICIES:
            source_files.append(args.run/environment/'arc'/f"seed_{protocol['evaluation_seeds'][0]}"/'yaw_5'/policy/'ground_truth.csv')
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    trials=csv_data(args.run/'trials.csv')
    groups=summary['groups']
    if len(trials)!=protocol['trial_count'] or sum(r['trials'] for r in groups)!=len(trials):
        raise ValueError('Deneme toplamları uyuşmuyor.')
    output=args.run/('review_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir(exist_ok=False)
    draw_routes(args.run,output,protocol);draw_primary(groups,output);draw_secondary(trials,output)
    calibration=json.loads((args.run/'calibration.json').read_text(encoding='utf-8-sig'))
    sections=['# EXP-42 - İki ek ortamda hareket seçimi doğrulaması',
              f"Toplam {len(trials)} politika çalıştırması. Dört politika aynı başlangıç ölçümlerini paylaşır; bunlar bağımsız {len(trials)} sensör kaydı değildir. Her ortam/rota/hata/politika hücresinde {len(protocol['evaluation_seeds'])} farklı sensör tohumu. Kaynak: {args.run.name}.",
              'Kalibrasyon yalnız eski ortamın sağlıklı kayıtlarından geldi ve test boyunca değişmedi. İki yeni ortamda eşik ayarı yapılmadı. Tüm yöntemlerin alarm öncesi ölçümleri ve gerçek rotaları eşleştirildi.',
              '## 1. Birincil ölçüt',
              'Alarm sonrası iki saniyede aynı kaynak ve aynı açı işareti iki ardışık destekli taramada aday olursa kaynak doğrulanır. Birincil oran bütün hatalı denemeleri içerir; alarm yokluğu ve belirsizlik başarı değildir.',
              markdown_table(['Ortam','Normal rota','Hata (°)','Politika','Doğru/n','Alarm/n','Belirsiz','Yanlış kaynak','Alarm yok'],
                [[r['environment'],r['normal_motion'],r['fault_yaw_deg'],NAMES[r['policy']],f"{r['correct_source']}/{r['trials']}",f"{r['alarm_trials']}/{r['trials']}",r['uncertain'],r['wrong_source'],r['no_alarm']] for r in groups if r['fault_yaw_deg']!=0]),
              '![Doğru kaynak oranları](02_correct_source_rates.png)',
              '## 2. Hatasız kontroller',
              markdown_table(['Ortam','Normal rota','Politika','Yanlış alarm/n','Tamamlanan/n'],
                [[r['environment'],r['normal_motion'],NAMES[r['policy']],f"{r['false_alarm']}/{r['trials']}",f"{r['completed_trials']}/{r['trials']}"] for r in groups if r['fault_yaw_deg']==0]),
              '## 3. Eşleştirilmiş doğru kaynak sonuçları',
              'Adaptif ile her sabit/pasif karşılaştırma, aynı ortam/rota/hata/tohum üzerinde yapılır. Aşağıdaki sayılar yalnız +5° koşuludur; +2° sonuçları ilk tabloda korunmuştur.',
              markdown_table(['Ortam','Normal rota','Karşılaştırılan','Eş çift','Yalnız adaptif doğru','Yalnız diğer doğru','İkisi doğru','İkisi de doğrulamadı'],
                [[r['environment'],r['normal_motion'],NAMES[r['comparator']],r['paired_trials'],r['adaptive_only_correct'],r['comparator_only_correct'],r['both_correct'],r['neither_correct']] for r in summary['paired'] if r['fault_yaw_deg']==5.]),
              '## 4. İkincil geometrik puan',
              'Pozitif fark, bu puanın adaptifte daha yüksek olduğunu gösterir; daha fazla doğru kaynak teşhisi olduğu anlamına gelmez. +5° grafiği iki yöntemde de dört karelik destek varsa hesaplanır. Desteksiz eşleşmeler sıfır yapılmaz.',
              '![Eşleştirilmiş geometrik fark](03_paired_geometric_differences.png)',
              '## 5. Gerçek yolların örneği',
              'İlk değerlendirme tohumu, yay normal rota ve +5° hata görselleştirilir. Bütün nicel sonuçlar tüm tohumları içerir. Çakışan çizgiler aynı fiziksel hareketin uygulandığını gösterebilir.',
              '![Ortam ve yollar](01_environments_and_routes.png)',
              '## 6. Sabit kalibrasyon ve sınırlar',
              f"Göreli açı eşiği: {calibration['alignment']['config']['relative_yaw_threshold_deg']} derece. Hipotez eşikleri: {calibration['diagnosis_thresholds']}. En büyük sensör RMS sınırı: {calibration['diagnosis_max_sensor_rms_m']:.6f} m.",
              'Bu sonuçlar iki ek geometri, bir başlangıç pozu, ön LiDAR montaj hatası, ideal LiDAR ve +2/+5 derece için geçerlidir. GPS kapalı ve çekirdek EKF’dir. Evrensel üstünlük, istatistiksel anlamlılık, gerçek robot veya bütün hata türleri başarısı iddia edilmez.',
              'Sabit düz/sabit yay yöntemleri de karşılaştırmada tutulur. Yalnız pasiften iyi çıkan bir hücreyi seçerek genel adaptif üstünlük ilan edilmez. Eski EXP-33 puanları farklı protokolden olduğu için aynı tabloda başarı sıralamasına katılmaz.',
              f"Gerekli sürelerin tamamı bitmiş: {summary['all_horizons_completed']}. Eski kaynaklar korunmuş: {summary['protected_sources_unchanged']}. Kalibrasyon korunmuş: {summary['calibration_unchanged']}."]
    (output/'heldout_results.md').write_text('\n\n'.join(sections)+'\n',encoding='utf-8')
    unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in before.items())
    (output/'sources.json').write_text(json.dumps({'source_run':str(args.run),'source_sha256':before,'source_files_unchanged':unchanged},indent=2),encoding='utf-8')
    if not unchanged:raise RuntimeError('Kaynak sonuç dosyaları değişti.')
    print('Sonuç dosyaları korunmuş: PASS')
    print(f'Rapor klasörü: {output}')


if __name__=='__main__':
    main()
