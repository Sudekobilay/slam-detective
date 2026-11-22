"""Read EXP-43 results, retain failures, plot success and paired movement costs."""
import argparse
from collections import defaultdict, Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

POLICIES=('passive','fixed_straight','fixed_arc','adaptive')


def load_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def group_rows(rows):
    groups=defaultdict(list)
    for r in rows:groups[r['noise_id'],r['policy'],float(r['fault_yaw_deg'])].append(r)
    result=[]
    for (noise,policy,angle),items in sorted(groups.items()):
        counts=Counter(r['outcome'] for r in items)
        result.append({'noise_id':noise,'policy':policy,'fault_yaw_deg':angle,'trials':len(items),
                       'correct_source':counts['correct_source'],'false_alarm':counts['false_alarm'],
                       'uncertain':counts['uncertain'],'no_alarm':counts['no_alarm'],
                       'wrong_source':counts['wrong_source'],'premature_alarm':counts['premature_alarm'],
                       'incomplete_window':counts['incomplete_window'],
                       'healthy_no_alarm':counts['healthy_no_alarm']})
    return result


def make_report(run):
    run=Path(run).resolve();summary=json.loads((run/'summary.json').read_text())
    required=('all_pairings_passed','protected_sources_unchanged','calibration_unchanged')
    if summary.get('experiment_id')!='exp_43' or not all(summary.get(k) for k in required):
        raise ValueError('Validated EXP-43 run required.')
    names=('summary.json','protocol.json','calibration.json','trials.csv','paired.csv')
    before={n:hashlib.sha256((run/n).read_bytes()).hexdigest() for n in names}
    rows=load_csv(run/'trials.csv');paired=load_csv(run/'paired.csv')
    if len(rows)!=summary['trial_count']:raise ValueError('Trial count mismatch.')
    groups=group_rows(rows);profiles=sorted({r['noise_id'] for r in rows})
    strata=defaultdict(list)
    keys=('environment','start_id','noise_id','normal_motion','fault_yaw_deg','policy')
    for row in rows:strata[tuple(row[k] for k in keys)].append(row)
    detailed=[]
    for key,items in sorted(strata.items()):
        counts=Counter(r['outcome'] for r in items)
        item={**dict(zip(keys,key)),'trials':len(items),
              **{o:counts[o] for o in ('correct_source','healthy_no_alarm','false_alarm','no_alarm','uncertain','wrong_source','premature_alarm','incomplete_window')}}
        for metric in ('common_distance_m','common_absolute_turn_rad','common_position_rmse_m','confirmation_delay_from_fault_s'):
            vals=[float(r[metric]) for r in items if r[metric]]
            item['mean_'+metric]=statistics.mean(vals) if vals else None
            item['supported_'+metric]=len(vals)
        detailed.append(item)
    angles=sorted({float(r['fault_yaw_deg']) for r in rows if float(r['fault_yaw_deg'])!=0})
    out=run/('review_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));out.mkdir()
    with (out/'stratified_summary.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(detailed[0]));writer.writeheader();writer.writerows(detailed)
    fig,axes=plt.subplots(len(profiles),max(1,len(angles)),figsize=(5*max(1,len(angles)),3.6*len(profiles)),squeeze=False,constrained_layout=True)
    for i,noise in enumerate(profiles):
        for j,angle in enumerate(angles):
            ax=axes[i,j];selected=[next(g for g in groups if g['noise_id']==noise and g['policy']==p and g['fault_yaw_deg']==angle) for p in POLICIES]
            values=[100*g['correct_source']/g['trials'] for g in selected]
            bars=ax.bar(range(4),values,color=['#94a3b8','#d99b42','#5eaaa1','#2563eb'])
            for b,g in zip(bars,selected):ax.text(b.get_x()+b.get_width()/2,b.get_height()+2,f"{g['correct_source']}/{g['trials']}",ha='center',fontsize=9)
            ax.set(ylim=(0,115),title=f'{noise} | {angle:+g}°',ylabel='Doğru kaynak (%)')
            ax.set_xticks(range(4),['Pasif','Sabit düz','Sabit yay','Adaptif']);ax.grid(axis='y',alpha=.2)
    fig.suptitle('EXP-43 | Başlangıçlar ve ortamlar birlikte; bütün hatalı denemeler paydada')
    fig.savefig(out/'success.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    metrics=[('delta_common_distance_m','Ortak 0-4 s mesafe farkı (m)'),('delta_common_absolute_turn_rad','Ortak 0-4 s mutlak dönüş farkı (rad)')]
    for ax,(key,label) in zip(axes,metrics):
        for i,comparator in enumerate(('passive','fixed_straight','fixed_arc')):
            vals=[float(r[key]) for r in paired if r['comparator']==comparator and r[key] and float(r['fault_yaw_deg'])!=0]
            ax.scatter([i]*len(vals),vals,alpha=.18,s=14,color='#2563eb')
            if vals:ax.scatter(i,statistics.mean(vals),marker='D',s=65,color='#0f766e',zorder=3)
        ax.axhline(0,color='gray',linewidth=.8);ax.set_xticks(range(3),['Pasif','Sabit düz','Sabit yay']);ax.set_ylabel(label);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Adaptif eksi karşılaştırıcı | Nokta: eş çift; elmas: ortalama; güven aralığı değil')
    fig.savefig(out/'paired_costs.png',dpi=160);plt.close(fig)
    lines=['# EXP-43 - Robustness ve hareket maliyeti','',f"{len(rows)} politika çalıştırması; {summary['paired_conditions']} eşleştirilmiş koşul.",
           '',f"Kısa kontrol modu: {json.loads((run/'protocol.json').read_text())['smoke_only']}",
           f"Gerekli süreler tamamlandı: {summary['all_horizons_completed']}",'',
           'Ayrıntılı ortam/başlangıç/rota/politika hücreleri stratified_summary.csv içindedir; pooled grafik hücre başına üç tohum sonucu gibi yorumlanmaz.', '', '## Bütün sonuçlar','', '| Gürültü | Politika | Hata (°) | n | Doğru | Belirsiz | Alarm yok | Erken | Yanlış kaynak | Yanlış alarm |',
           '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for g in groups:lines.append('| '+' | '.join(str(g[k]) for k in ('noise_id','policy','fault_yaw_deg','trials','correct_source','uncertain','no_alarm','premature_alarm','wrong_source','false_alarm'))+' |')
    lines+=['','![Doğru kaynak](success.png)','','![Eşleştirilmiş maliyet](paired_costs.png)','',
     '## Ölçüm sınırları','',
     '- Sağlıklı satırlarda doğru kaynak oranı tanımlanmaz; yanlış alarm ve healthy_no_alarm sayıları kullanılır.',
     '- Farklı ortam ve başlangıçların toplamı aynı koşuldaki bağımsız tekrar sayısı değildir; ayrıntılar trials.csv içindedir.',
     '- Kalibrasyon özgün ortamın ideal LiDAR kayıtlarından dondurulmuştur. Gürültü eklendiğinde eşikler/filtre kovaryansları yeniden ayarlanmamıştır; bu bir dağılım değişimi stres testidir.',
     '- Başlangıç pozu kestiriciye doğru öncül olarak verilir; başlangıç belirsizliği çözülmüş değildir.',
     '- Hareket maliyeti ortak 0-4 s ufkunda hesaplanır. Eşit hızlı manevralarda ek mesafe sıfır olabilir; duruştan/teşhisten sonraki seyahat uydurulmaz.',
     '- Teşhis süresi yalnız doğrulanmış denemelerde tanımlıdır; trials.csv içindeki confirmation_delay alanlarındadır. Kaçırılan sonuçlarda sıfır yazılmaz.',
     '- Negatif hareket farkı daha az hareketi belirtir; tek başına daha iyi teşhis değildir. Başarı ve maliyet birlikte okunur.',
     '- Seçicinin puanı değiştirilmedi; maliyet sonradan ölçüldü, maliyetle optimize edilmiş politika uygulanmış gibi sunulmaz.']
    (out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    after={n:hashlib.sha256((run/n).read_bytes()).hexdigest() for n in names}
    if before!=after:raise RuntimeError('Source results changed.')
    (out/'sources.json').write_text(json.dumps({'run':str(run),'sha256':before,'source_unchanged':True,'groups':groups},indent=2))
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    print('Report:',make_report(parser.parse_args().run))

if __name__=='__main__':main()
