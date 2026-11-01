"""Seçilmiş deney sonuçlarından ortak tablolar, grafikler ve isteğe bağlı PDF.

Yeni simülasyon çalıştırmaz. Farklı protokollerin skorlarını birleştirmez.
Kaynak seçimi ve dosya özetleri sources.json içinde saklanır.
"""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


BATCHES = {'algorithms':38, 'alarms':30, 'policies':33, 'gps':35, 'sources':34, 'real':41}
POLICIES = ('passive','fixed_straight','fixed_arc','adaptive')
POLICY_NAMES = {'passive':'Pasif','fixed_straight':'Sabit düz','fixed_arc':'Sabit yay','adaptive':'Adaptif'}
ALGOS = ('ekf','fastslam1','fastslam2')
ALGO_NAMES = {'ekf':'EKF-SLAM','fastslam1':'FastSLAM 1.0','fastslam2':'FastSLAM 2.0'}
ROUTES = {'straight':'Düz normal rota','arc':'Yay normal rota'}
COLORS = ['#23395d','#cb7c32','#347f79','#895a91']


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def number(value):
    if value is None or value == '':
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('Sonlu olmayan sonuç.')
    return result


def yes(value):
    if value not in (True, False, 'True', 'False'):
        raise ValueError(f'Geçersiz bool: {value}')
    return value is True or value == 'True'


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def spread(values):
    values = [v for v in values if v is not None]
    return statistics.stdev(values) if len(values) > 1 else None


def read_trials(folder):
    with (Path(folder)/'trials.csv').open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def key_groups(rows, fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in fields)].append(row)
    return groups


def check_summary(folder, experiment):
    data = load_json(Path(folder)/'summary.json')
    if data.get('experiment_id') != f'exp_{experiment}':
        raise ValueError('Deney numarası uyuşmuyor.')
    if experiment == 33 and (not data.get('all_horizons_completed') or not data.get('pairing_checks')
                             or not all(all(c.values()) for c in data['pairing_checks'].values())):
        raise ValueError('EXP-33 süre veya ortak geçmiş kontrolleri tamamlanmamış.')
    if experiment == 34 and not data.get('all_isolation_checks_passed'):
        raise ValueError('EXP-34 izolasyon kontrolü başarısız.')
    if experiment == 35 and not data.get('all_gps_isolation_passed'):
        raise ValueError('EXP-35 izolasyon kontrolü başarısız.')
    if experiment == 40 and not data.get('required_horizon_completed'):
        raise ValueError('EXP-40 gerekli süreyi tamamlamamış.')
    if experiment == 41 and not data.get('source_file_unchanged'):
        raise ValueError('EXP-41 kaynak koruma kontrolü başarısız.')
    return data


def resolve_sources(results, selection=None):
    """En yeni geçerli toplu kayıt; EXP-40'ta her farklı dört-LiDAR koşulu."""
    results = Path(results)
    selected, rejected = {}, []
    if selection is not None:
        for name, value in selection.items():
            if name not in (*BATCHES, 'active'):
                raise ValueError(f'Bilinmeyen kaynak alanı: {name}')
            paths = value if name == 'active' else [value]
            if not isinstance(paths, list):
                raise ValueError('active alanı klasör listesi olmalı.')
            for value in paths:
                folder = Path(value)
                check_summary(folder, 40 if name == 'active' else BATCHES[name])
                selected.setdefault(name, []).append(folder)
        return selected, rejected
    for name, experiment in BATCHES.items():
        for path in sorted(results.glob(f'exp_{experiment}_*/summary.json'), reverse=True):
            try:
                check_summary(path.parent, experiment)
                if experiment != 41:
                    read_trials(path.parent)
                selected[name] = [path.parent]
                break
            except (ValueError, OSError, KeyError) as error:
                rejected.append({'folder':str(path.parent), 'reason':str(error)})
    seen = set()
    for path in sorted(results.glob('exp_40_*/summary.json'), reverse=True):
        try:
            summary = check_summary(path.parent, 40)
            layout = load_json(path.parent/'used_config.json')
            metadata = load_json(path.parent/'simulation_metadata.json')
            if len(layout['lidars']) != 4:
                continue
            with (path.parent/'commands.csv').open(newline='',encoding='utf-8-sig') as handle:
                first = next(csv.DictReader(handle))
            identity = json.dumps([summary['algorithm'],layout,metadata['layout_fault'],
                                   metadata['world_config']['gyro_seed'],
                                   first['left_rate_rad_s'],first['right_rate_rad_s']], sort_keys=True)
            if identity not in seen:
                seen.add(identity)
                selected.setdefault('active', []).append(path.parent)
        except (ValueError, OSError, KeyError, StopIteration) as error:
            rejected.append({'folder':str(path.parent), 'reason':str(error)})
    return selected, rejected


def algorithm_rows(rows):
    # Aynı koşulun algoritmalarına farklı ham ölçüm kaydı verilmişse birleştirme.
    for _, group in key_groups(rows, ('layout','gps_enabled')).items():
        if len({r['source_sha256'] for r in group}) != 1:
            raise ValueError('EXP-38 içinde ortak ölçüm özeti uyuşmuyor.')
    result = []
    for (layout, gps, algorithm), group in sorted(key_groups(rows, ('layout','gps_enabled','algorithm')).items()):
        seeds = [r.get('particle_seed') for r in group]
        if len(seeds) != len(set(seeds)):
            raise ValueError('Tekrarlanan algoritma/tohum kaydı.')
        rmse = [number(r['position_rmse_m']) for r in group]
        seconds = [number(r['replay_seconds']) for r in group]
        result.append({'layout':layout, 'gps':yes(gps), 'algorithm':algorithm, 'n':len(group),
                       'mean_rmse_m':mean(rmse), 'sd_rmse_m':spread(rmse),
                       'mean_seconds':mean(seconds), 'sd_seconds':spread(seconds)})
    return result


def policy_rows(rows):
    result, pairs = [], []
    identities = [(r['normal_motion'], r['seed'], r['fault_yaw_deg'], r['policy']) for r in rows]
    if len(identities) != len(set(identities)):
        raise ValueError('Tekrarlanan politika denemesi.')
    for (normal, yaw, policy), group in sorted(key_groups(rows, ('normal_motion','fault_yaw_deg','policy')).items()):
        supported = [r for r in group if yes(r['diagnostic_window_completed']) and number(r['window_frames']) == 4
                     and number(r['supported_frames']) == 4 and number(r['mean_front_margin_m2']) is not None]
        result.append({'route':normal, 'yaw':float(yaw), 'policy':policy, 'n':len(group),
                       'alarms':sum(yes(r['alarm']) for r in group), 'full_support':len(supported),
                       'margin_m2':mean([number(r['mean_front_margin_m2']) for r in supported]),
                       'sd_margin_m2':spread([number(r['mean_front_margin_m2']) for r in supported]),
                       'rmse_m':mean([number(r['common_position_rmse_m']) for r in group]),
                       'changed':sum((number(r['changed_command_duration_s']) or 0)>1e-9 for r in group),
                       'distance_m':mean([number(r['diagnostic_distance_m']) for r in group])})
    for (normal, yaw), group in sorted(key_groups(rows, ('normal_motion','fault_yaw_deg')).items()):
        if float(yaw) == 0:
            continue
        by_policy = {p:{r['seed']:r for r in group if r['policy']==p} for p in ('adaptive','passive')}
        differences = []
        for seed in sorted(set(by_policy['adaptive']) & set(by_policy['passive'])):
            a, b = by_policy['adaptive'][seed], by_policy['passive'][seed]
            if all(yes(r['diagnostic_window_completed']) and number(r['supported_frames']) == number(r['window_frames']) == 4
                   and number(r['mean_front_margin_m2']) is not None for r in (a,b)):
                differences.append(number(a['mean_front_margin_m2']) - number(b['mean_front_margin_m2']))
        pairs.append({'route':normal, 'yaw':float(yaw), 'paired_n':len(differences),
                      'mean_delta_m2':mean(differences), 'sd_delta_m2':spread(differences),
                      'positive':sum(d>1e-12 for d in differences), 'tied':sum(abs(d)<=1e-12 for d in differences)})
    return result, pairs


def gps_rows(rows):
    result = []
    for (route, case), group in sorted(key_groups(rows, ('normal_motion','case')).items()):
        result.append({'route':route, 'case':case, 'n':len(group),
                       'geometry_alarms':sum(yes(r['geometric_alarm']) for r in group),
                       'gps_alarms':sum(yes(r['gps_confirmed']) for r in group),
                       **{name:mean([number(r[name+'_rmse_m']) for r in group])
                          for name in ('no_gps','plain_gps','guarded_gps')}})
    return result


def outcome_rows(rows):
    result = []
    mapped = [dict(r, magnitude=abs(float(r['angle_deg']))) for r in rows]
    for (source, magnitude), group in sorted(key_groups(mapped, ('source','magnitude')).items()):
        counts = {k:sum(r['outcome']==k for r in group) for k in ('correct','uncertain','missed_fault','wrong_source','false_fault')}
        if sum(counts.values()) != len(group):
            raise ValueError('Bilinmeyen teşhis sonuç etiketi.')
        result.append({'source':source, 'magnitude':magnitude, 'n':len(group), **counts})
    return result


def active_row(folder):
    summary = check_summary(folder, 40)
    metadata = load_json(folder/'simulation_metadata.json')
    layout = load_json(folder/'used_config.json')
    fault = metadata['layout_fault']
    return {'run':folder.name, 'algorithm':summary['algorithm'], 'lidars':len(layout['lidars']),
            'gps':layout['gps_enabled'], 'fault':fault['type'], 'sensor':fault['sensor'],
            'yaw':fault['yaw_deg'], 'alarm_s':summary['trigger_time_s'],
            'motion':summary['selected_motion'], 'completed':summary['required_horizon_completed']}


def compact(value):
    if value is None:
        return '-'
    if isinstance(value, bool):
        return 'Evet' if value else 'Hayır'
    if isinstance(value, float):
        return f'{value:.3e}' if 0<abs(value)<.001 else f'{value:.4f}'.rstrip('0').rstrip('.')
    return str(value)


def markdown_table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']
                     + ['| '+' | '.join(compact(v).replace('|','/') for v in row)+' |' for row in rows])


def chart_save(figure, path):
    figure.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(figure)


def plot_algorithms(rows, path, timing=False):
    fig, axes = plt.subplots(1,2, figsize=(11,4), constrained_layout=True)
    layouts = [name for name in ('single','dual','quad') if any(r['layout']==name for r in rows)]
    for axis, gps in zip(axes, (False,True)):
        for index, algorithm in enumerate(ALGOS):
            chosen = [next(r for r in rows if r['layout']==layout and r['gps']==gps and r['algorithm']==algorithm) for layout in layouts]
            metric, deviation = ('mean_seconds','sd_seconds') if timing else ('mean_rmse_m','sd_rmse_m')
            values = [r[metric] for r in chosen]
            axis.bar(np.arange(len(layouts))+(index-1)*.24, values, width=.23,
                     yerr=[r[deviation] or 0 for r in chosen], capsize=3, color=COLORS[index], label=ALGO_NAMES[algorithm])
        axis.set_xticks(range(len(layouts)), ['1 LiDAR' if v=='single' else '2 LiDAR' if v=='dual' else '4 LiDAR' for v in layouts])
        axis.set_title('GPS açık' if gps else 'GPS kapalı')
        axis.set_ylabel('Yerel yeniden oynatma süresi (s)' if timing else 'Ortalama konum RMSE (m)')
        axis.grid(axis='y', alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('EXP-38: aynı koşul içinde ortak kayıt; EKF n=1, FastSLAM n=3 parçacık tohumu\nHata çubukları örnek standart sapmadır; güven aralığı değildir.', fontsize=10)
    chart_save(fig, path)


def plot_alarms(rows, path):
    fig, axes = plt.subplots(1,2, figsize=(11,4), constrained_layout=True)
    for axis, normal in zip(axes, ('straight','arc')):
        for index, monitor in enumerate(('matched_nis','scan_consensus')):
            chosen = sorted([r for r in rows if r['normal_motion']==normal and r['monitor']==monitor], key=lambda r:r['fault_yaw_deg'])
            x = np.arange(len(chosen))+(index-.5)*.30
            bars=axis.bar(x,[r['alarm_trials']/r['trials'] for r in chosen],width=.28,color=COLORS[index],label='Eşleşen NIS' if index==0 else 'Ek ham tarama izleyicisi')
            for bar,row in zip(bars,chosen):
                axis.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.035,f"{row['alarm_trials']}/{row['trials']}",ha='center',fontsize=9)
        axis.set_xticks(range(len(chosen)),[f"{r['fault_yaw_deg']:g}°" for r in chosen])
        axis.set(ylim=(0,1.2),title=ROUTES[normal],xlabel='Ön LiDAR montaj hatası',ylabel='Alarm üreten deneme oranı')
        axis.set_yticks([0,.25,.5,.75,1.],['0%','25%','50%','75%','100%'])
        axis.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('EXP-30: ayrı doğrulama tohumları (held_out); her koşulda n=5\n0° çubukları yanlış alarm kontrolüdür. Alarm, doğru kaynak teşhisi değildir.',fontsize=10)
    chart_save(fig,path)


def plot_policies(rows, path):
    fig, axes = plt.subplots(2,2,figsize=(11,6),constrained_layout=True)
    for axis, (route,yaw) in zip(axes.flat, ((r,y) for r in ('straight','arc') for y in (2.,5.))):
        chosen = [next(r for r in rows if r['route']==route and r['yaw']==yaw and r['policy']==p) for p in POLICIES]
        values = [r['margin_m2']*1e4 if r['margin_m2'] is not None else float('nan') for r in chosen]
        bars=axis.bar(range(4),values,yerr=[(r['sd_margin_m2'] or 0)*1e4 for r in chosen],capsize=3,color=COLORS)
        for bar,row in zip(bars,chosen):
            axis.text(bar.get_x()+bar.get_width()/2,(bar.get_height() if row['margin_m2'] is not None else 0),f" n={row['full_support']}/{row['n']}",ha='center',va='bottom',fontsize=8)
        axis.set_xticks(range(4),[POLICY_NAMES[p] for p in POLICIES])
        axis.set_title(f'{ROUTES[route]}, +{yaw:g}°')
        axis.set_ylabel('Geometrik model farkı (×10⁻⁴ m²)')
        axis.margins(y=.25)
        axis.grid(axis='y',alpha=.2)
    fig.suptitle('EXP-33: tam destekli teşhis pencerelerinde ortalama ön-LiDAR model farkı\nÇubuklar ortalama, hata çizgileri örnek SS; olasılık veya doğru teşhis oranı değildir.',fontsize=10)
    chart_save(fig,path)


def plot_gps(rows,path):
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    cases=('healthy','front_fault','rear_fault','gps_positive','gps_negative','front_and_gps')
    labels=('Hatasız','Ön LiDAR','Arka LiDAR','GPS +','GPS -','Ön + GPS')
    for axis,route in zip(axes,('straight','arc')):
        chosen=[next(r for r in rows if r['route']==route and r['case']==case) for case in cases]
        for index,(metric,label) in enumerate((('no_gps','GPS yok'),('plain_gps','Korumasız GPS'),('guarded_gps','Korumalı GPS'))):
            axis.bar(np.arange(6)+(index-1)*.24,[r[metric]*1000 for r in chosen],width=.23,color=COLORS[index],label=label)
        axis.set_xticks(range(6),labels,rotation=25,ha='right')
        axis.set(title=ROUTES[route],ylabel='Ortalama konum RMSE (mm)')
        axis.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('EXP-35: her rota/senaryoda n=5 sensör tohumu; aynı kayıt üzerinde üç kestirici\nLiDAR izleme/koruma ortak; farklılık GPS güncellemesindedir.',fontsize=10)
    chart_save(fig,path)


def plot_outcomes(rows,path):
    chosen=[r for r in rows if r['source']!='healthy']
    names={'front_yaw':'Ön LiDAR','rear_yaw':'Arka LiDAR','heading_offset':'Kontrollü yön'}
    fig,axis=plt.subplots(figsize=(10,4.2),constrained_layout=True)
    left=np.zeros(len(chosen))
    for index,(key,label) in enumerate((('correct','Doğru'),('uncertain','Belirsiz'),('missed_fault','Kaçırılan'),('wrong_source','Yanlış kaynak'),('false_fault','Yanlış hata'))):
        values=np.array([r[key] for r in chosen])
        axis.barh(range(len(chosen)),values,left=left,color=plt.get_cmap('tab10')(index),label=label)
        for y,(width,start) in enumerate(zip(values,left)):
            if width:axis.text(start+width/2,y,str(width),ha='center',va='center',fontsize=9)
        left+=values
    axis.set_yticks(range(len(chosen)),[f"{names[r['source']]} | ±{r['magnitude']:g}° | n={r['n']}" for r in chosen])
    axis.set_xlabel('Deneme sayısı; iki rota ve iki hata işareti birlikte')
    axis.legend(ncol=3,fontsize=8,loc='upper center',bbox_to_anchor=(.5,1.23))
    axis.grid(axis='x',alpha=.2)
    chart_save(fig,path)


def build_sections(selected, output):
    sections, computed = [], {}
    if 'algorithms' in selected:
        rows=algorithm_rows(read_trials(selected['algorithms'][0]));computed['algorithms']=rows
        plot_algorithms(rows,output/'01_algorithms_rmse.png')
        plot_algorithms(rows,output/'02_algorithms_time.png',timing=True)
        sections.append({
            'title':'1A. Konum kestirimi - EXP-38',
            'note':'Her yerleşim/GPS koşulunda ortak sensör kaydı. EKF n=1; FastSLAM n=3 parçacık tohumu. Ortalama ± örnek standart sapma kullanılır; n=1 için sapma tanımsızdır. Üç parçacık tohumu bağımsız üç sensör deneyi değildir. Süreler bu çalıştırmanın bilgisayarına aittir.',
            'headers':['Düzen','GPS','Algoritma','n','RMSE (m)','SS (m)','Süre (s)'],
            'rows':[[r['layout'],r['gps'],ALGO_NAMES[r['algorithm']],r['n'],r['mean_rmse_m'],r['sd_rmse_m'],r['mean_seconds']] for r in rows],
            'charts':['01_algorithms_rmse.png'],
            'interpretation':'Bu bölümün sonuçları nominal, ideal LiDAR ve seçilen parçacık ayarları içindir. Başka protokollerin hata değerleriyle bir genel başarı sıralaması oluşturulmaz. Işın bütçeleri yerleşimler arasında tam eşit değildir.'})
        sections.append({
            'title':'1B. Yerel işlem süreleri - EXP-38',
            'note':'Aynı kayıtlarda ölçülen yerel yeniden oynatma süresi. EKF n=1; FastSLAM n=3 parçacık tohumu. SS örnek standart sapmadır, güven aralığı değildir. Donanım ve sayısal kütüphane ayarları süreyi etkiler.',
            'headers':['Düzen','GPS','Algoritma','n','Ort. süre (s)','SS (s)'],
            'rows':[[r['layout'],r['gps'],ALGO_NAMES[r['algorithm']],r['n'],r['mean_seconds'],r['sd_seconds']] for r in rows],
            'charts':['02_algorithms_time.png'],
            'interpretation':'Bu değerler gerçek robot gerçek zaman garantisi, işlem başına gecikme veya FPS ölçümü değildir. Başka bilgisayarda elde edilmiş sürelerle tek sıralamada birleştirilmez.'})
    if 'alarms' in selected:
        folder=selected['alarms'][0]
        rows=[r for r in check_summary(folder,30)['groups'] if r['split']=='held_out']
        trials=[r for r in read_trials(folder) if r['split']=='held_out']
        for r in rows:
            group=[t for t in trials if t['normal_motion']==r['normal_motion'] and t['monitor']==r['monitor'] and float(t['fault_yaw_deg'])==r['fault_yaw_deg']]
            if len(group)!=r['trials'] or sum(yes(t['alarm']) for t in group)!=r['alarm_trials']:
                raise ValueError('EXP-30 özet ve tekil denemeler uyuşmuyor.')
        computed['alarms']=rows
        plot_alarms(rows,output/'03_alarm_monitors.png')
        sections.append({
            'title':'2. Alarm izleyicileri - EXP-30',
            'note':'Yalnız held_out doğrulama bölümü; her rota/hata/izleyicide n=5. İki izleyici aynı sabit kaydı alır. Bu hareket seçimi veya doğru kaynak teşhisi karşılaştırması değildir.',
            'headers':['Rota','Hata (°)','İzleyici','Alarm/n','Erken','Gecikme (s)'],
            'rows':[[r['normal_motion'],r['fault_yaw_deg'],'NIS' if r['monitor']=='matched_nis' else 'Ham tarama',f"{r['alarm_trials']}/{r['trials']}",r['premature_alarms'],r['mean_detection_delay_s']] for r in rows],
            'charts':['03_alarm_monitors.png'],
            'interpretation':'Gecikme yalnız hata sonrası alarm üreten denemeler üzerinden hesaplanır; alarm yokken sıfır kabul edilmez. 0° koşulu yanlış alarm kontrolüdür. Bu küçük örneklemden genel hata yakalama garantisi çıkarılmaz.'})
    if 'policies' in selected:
        rows,pairs=policy_rows(read_trials(selected['policies'][0]));computed['policies']=rows;computed['paired_policies']=pairs
        plot_policies(rows,output/'04_motion_policies.png')
        faulty=[r for r in rows if r['yaw']!=0]
        sections.append({
            'title':'3. Teşhis hareketi politikaları - EXP-33',
            'note':'Her rota/hata/politikada n=5 sensör tohumu. Hipotez puanı yalnız dört tarama anının tamamında yeterli destek bulunan pencerelerde hesaplanır. Eksik destekli denemeler sıfır puan değildir.',
            'headers':['Rota','Hata (°)','Politika','Destek/n','Fark (m²)','Komut değişen/n'],
            'rows':[[r['route'],r['yaw'],POLICY_NAMES[r['policy']],f"{r['full_support']}/{r['n']}",r['margin_m2'],f"{r['changed']}/{r['n']}"] for r in faulty],
            'charts':['04_motion_policies.png'],
            'interpretation':'Geometrik model farkı olasılık veya doğru teşhis yüzdesi değildir. Adaptif seçim normal rotayla aynı hareketi seçebilir. Bu durumda seçim vardır ama uygulanan komut değişmemiştir. Tüm koşullarda adaptif üstünlük gösterilmiş kabul edilmez.'})
        sections.append({
            'title':'4. Eşleştirilmiş hareket farkı - EXP-33',
            'note':'Aynı rota, hata büyüklüğü ve sensör tohumu için adaptif eksi pasif model farkı. Her iki pencerede tam destek koşulu birlikte aranır. Pozitif değer yalnız bu geometrik puanın arttığını gösterir; istatistiksel anlamlılık testi yapılmamıştır.',
            'headers':['Rota','Hata (°)','Eş çift n','Ort. fark (m²)','SS (m²)','Pozitif','Eşit'],
            'rows':[[r['route'],r['yaw'],r['paired_n'],r['mean_delta_m2'],r['sd_delta_m2'],r['positive'],r['tied']] for r in pairs],
            'charts':[],
            'interpretation':'Sabit yay politikası aynı karşılaştırmalarda adaptiften daha yüksek ortalama puan verebilir. Başarı, yalnız pasiften yüksek olmak üzerinden veya sonradan iyi örnekler seçilerek tanımlanmaz.'})
        healthy=[r for r in rows if r['yaw']==0]
        sections[-1]['headers2']=['Hatasız rota','Politika','Alarm/n']
        sections[-1]['rows2']=[[r['route'],POLICY_NAMES[r['policy']],f"{r['alarms']}/{r['n']}"] for r in healthy]
    if 'gps' in selected:
        rows=gps_rows(read_trials(selected['gps'][0]));computed['gps']=rows
        plot_gps(rows,output/'05_gps_protection.png')
        sections.append({
            'title':'5. GPS koruması - EXP-35',
            'note':'Her rota/senaryoda n=5; üç kestirici aynı kaydı kullanır. RMSE tabloda ve grafikte mm cinsindedir. LiDAR izleme/koruma düzeni ortaktır. GPS yok, korumasız GPS ve korumalı GPS güncellemesi karşılaştırılır.',
            'headers':['Rota','Senaryo','n','GPS yok (mm)','Korumasız (mm)','Korumalı (mm)','GPS alarm/n'],
            'rows':[[r['route'],r['case'],r['n'],r['no_gps']*1000,r['plain_gps']*1000,r['guarded_gps']*1000,f"{r['gps_alarms']}/{r['n']}"] for r in rows],
            'charts':['05_gps_protection.png'],
            'interpretation':'GPS sapması koşullarında koruma, sapmalı güncellemelerin etkisini azaltabilir. Hatasız senaryoda gürültülü GPS eklemek bu kısa, düşük odometri hatalı simülasyonda RMSE’yi artırabilir; GPS her zaman faydalı sonucu çıkarılmaz.'})
    if 'sources' in selected:
        rows=outcome_rows(read_trials(selected['sources'][0]));computed['fault_sources']=rows
        plot_outcomes(rows,output/'06_fault_source_outcomes.png')
        sections.append({
            'title':'6. Hata kaynağı sonuçları - EXP-34',
            'note':'İki rota ve pozitif/negatif işaretler, aynı kaynak ve mutlak hata büyüklüğü içinde toplanır. Hatalı her satır n=20; hatasız kontrol n=10. Referans t>=2 s için önceden belirlenmiş biçimde dondurulur; EXP-33 çevrim içi protokolüyle aynı değildir.',
            'headers':['Kaynak','|Açı| (°)','n','Doğru','Belirsiz','Kaçırılan','Yanlış kaynak','Yanlış hata'],
            'rows':[[r[k] for k in ('source','magnitude','n','correct','uncertain','missed_fault','wrong_source','false_fault')] for r in rows],
            'charts':['06_fault_source_outcomes.png'],
            'interpretation':'heading_offset, karşılaştırma tahminine uygulanan kontrollü yön sapmasıdır; gerçek IMU bias/drift teşhisi diye sunulmaz. Küçük hatalarda belirsiz sonuçların raporlanması tasarımın parçasıdır. Alarm ve doğru kaynak teşhisi ayrı ölçülerdir.'})
    active=[active_row(p) for p in selected.get('active',[])]
    if active:
        computed['active']=active
        sections.append({
            'title':'7. Son sistem kontrolleri - EXP-40',
            'note':'Her satır tek çalıştırmadır; başarı oranı veya politikalar arası üstünlük deneyi değildir. Yalnız seçilen dört LiDAR kayıtları gösterilir. Otomatik seçim aynı koşulun en yeni tamamlanmış kaydını alır; farklı koşulları birleştirmez.',
            'headers':['Çekirdek','Hata','Sensör','GPS','Açı (°)','Alarm (s)','Hareket','Tamamlandı'],
            'rows':[[r[k] for k in ('algorithm','fault','sensor','gps','yaw','alarm_s','motion','completed')] for r in active],
            'charts':[],
            'interpretation':'Alarm yokluğu sağlıklı teşhisi değildir; özellikle destek veya eşik yetersizliği olabilir. İncelenen dört LiDAR FastSLAM koşullarında alarm üretilememişti; bu kayıtlar seçili klasörlerde bulunmuyorsa burada sayısal satır üretilmez.'})
    if 'real' in selected:
        r=check_summary(selected['real'][0],41);computed['real']=r
        sections.append({
            'title':'8. Gerçek veri bağlantısı - EXP-41',
            'note':'Tek gerçek LiDAR ve kayda ekli odometri. Yeni hareket uygulanmaz; gerçek IMU/GPS veya çoklu LiDAR değerlendirmesi değildir.',
            'headers':['Ölçüt','Değer'],
            'rows':[['Algoritma',r['algorithm']],['Seçilen tarama',r['selected_scans']],['Kayıt süresi (s)',r['duration_s']],
                    ['Harita çizgisi',r['map_line_count']],['Yerel işlem süresi (s)',r['processing_total_s']],
                    ['Kaynak korunmuş',r['source_file_unchanged']],['ATE/RMSE doğruluğu','Hesaplanmadı; bağımsız referans yok']],
            'charts':[],
            'interpretation':'Kayıtlı odometri gerçek konum değildir. Harita çizgisi sayısı doğruluk puanı değildir. Seçilen kayıt ön eki ve tarama seyrekliği tam veri seti benchmark’ı yerine geçmez.'})
    return sections,computed


def write_pdf(sections, sources, path):
    # PDF isteğe bağlıdır; kullanıcı ortamında reportlab gerektirir.
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.pagesizes import A4
    from matplotlib import font_manager
    from xml.sax.saxutils import escape
    for name,weight in (('Lab','normal'),('LabBold','bold')):
        pdfmetrics.registerFont(TTFont(name,font_manager.findfont(font_manager.FontProperties(family='DejaVu Sans',weight=weight))))
    body=ParagraphStyle('body',fontName='Lab',fontSize=9,leading=13,spaceAfter=9)
    title=ParagraphStyle('title',fontName='LabBold',fontSize=17,leading=22,spaceAfter=14,textColor=colors.HexColor('#23395d'))
    small=ParagraphStyle('small',fontName='Lab',fontSize=7.2,leading=10,spaceAfter=6)
    cell=ParagraphStyle('cell',fontName='Lab',fontSize=7.2,leading=9.5)
    head=ParagraphStyle('head',parent=cell,fontName='LabBold',textColor=colors.white)
    def p(text,style=body):
        return Paragraph(escape(str(text)),style)
    width=A4[0]-76
    def table(headers,rows):
        data=[[p(h,head) for h in headers]]+[[p(compact(v),cell) for v in row] for row in rows]
        result=Table(data,colWidths=[width/len(headers)]*len(headers),repeatRows=1,hAlign='LEFT')
        result.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#23395d')),
                                   ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f2f5f8'),colors.white]),
                                   ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),
                                   ('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
        return result
    story=[p('SLAM-Detective',title),p('Seçilmiş deney sonuçları',title),
           p('Sudenaz Kobilay - Kocaeli Üniversitesi, Yazılım Mühendisliği'),
           p('Bu kitapçık seçilmiş kayıtların tablo ve grafik özetidir. Ana akademik rapor değildir. Bu çıktı, çalıştırıldığı proje kopyasındaki dosyalardan üretilmiştir; başka bilgisayardaki çalıştırmaların dosyalarıymış gibi sunulmaz.'),
           p('Karşılaştırma kuralları',title),
           p('Algoritmalar aynı protokol içinde karşılaştırılır. Alarm oranı, hipotez desteği ve doğru kaynak teşhisi ayrıdır. Eksik sonuçlar sıfır sayılmaz. Tek çalıştırmalı bütünleştirme kontrollerinden başarı oranı çıkarılmaz.'),
           p('Eksik bölümler: '+(', '.join(name for name in (*BATCHES,'active') if name not in sources) or 'Yok')),
           p('Kaynak seçimi',title),
           p('Toplu deneylerde en yeni geçerli sonuç; EXP-40 için her farklı dört-LiDAR koşulunun en yeni tamamlanmış kaydı kullanılır. Açık bir seçim dosyasıyla bu liste sabitlenebilir. Kaynak dosya özetleri sources.json içinde bulunur.'),
           table(['Bölüm','Kullanılan klasör'],[[name,Path(folder).name] for name,folders in sources.items() for folder in folders]),
           Spacer(1,10),p('Tarih: '+datetime.now().astimezone().isoformat(timespec='seconds'),small)]
    for section in sections:
        story.extend([PageBreak(),p(section['title'],title),p(section['note']),table(section['headers'],section['rows']),Spacer(1,10)])
        if 'rows2' in section:
            story.extend([p('Hatasız kontroller',body),table(section['headers2'],section['rows2']),Spacer(1,10)])
        for chart in section['charts']:
            image=Image(str(path.parent/chart));ratio=image.imageHeight/image.imageWidth
            story.extend([Image(str(path.parent/chart),width=width,height=width*ratio),Spacer(1,8)])
        story.append(p(section['interpretation']))
    def footer(canvas,doc):
        canvas.setFont('Lab',8);canvas.setFillColor(colors.HexColor('#627080'))
        canvas.drawString(38,24,'SLAM-Detective | Seçilmiş sonuçlar')
        canvas.drawRightString(A4[0]-38,24,str(doc.page))
    SimpleDocTemplate(str(path),pagesize=A4,leftMargin=38,rightMargin=38,topMargin=38,bottomMargin=40).build(story,onFirstPage=footer,onLaterPages=footer)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=Path('results'))
    parser.add_argument('--selection',type=Path,help='İsteğe bağlı: bölüm adı -> klasör yolu JSON dosyası.')
    parser.add_argument('--pdf',action='store_true',help='PDF de üret; reportlab gerekir.')
    args=parser.parse_args()
    selection=load_json(args.selection) if args.selection else None
    selected,rejected=resolve_sources(args.results,selection)
    if not selected:
        parser.error('Kullanılabilir sonuç kaydı bulunamadı.')
    if args.pdf:
        try:
            import reportlab
        except ImportError:
            parser.error('PDF için: python -m pip install reportlab. Alternatif: --pdf olmadan çalıştırın.')
    source_files={}
    for folders in selected.values():
        for folder in folders:
            for name in ('summary.json','trials.csv','calibration.json','used_config.json','simulation_metadata.json','commands.csv'):
                path=folder/name
                if path.exists():source_files[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output=args.results/f'analysis_{stamp}'
    output.mkdir(parents=True,exist_ok=False)
    sections,computed=build_sections(selected,output)
    source_paths={name:[str(p) for p in paths] for name,paths in selected.items()}
    missing=[name for name in (*BATCHES,'active') if name not in selected]
    text=['# SLAM-Detective - Seçilmiş deney sonuçları',
          'Bu çıktı çalıştırıldığı proje kopyasının kayıtlarından üretilmiştir. Kaynaklar sources.json içindedir.',
          'Eksik bölüm: '+(', '.join(missing) if missing else 'Yok')]
    for s in sections:
        text.extend(['\n## '+s['title'],s['note'],markdown_table(s['headers'],s['rows'])])
        if 'rows2' in s:text.extend(['### Hatasız kontroller',markdown_table(s['headers2'],s['rows2'])])
        text.extend([f'![{s["title"]}]({name})' for name in s['charts']])
        text.append(s['interpretation'])
    (output/'results_summary.md').write_text('\n\n'.join(text)+'\n',encoding='utf-8')
    (output/'tables.json').write_text(json.dumps(computed,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    (output/'selected_sources.json').write_text(json.dumps({name:paths if name=='active' else paths[0] for name,paths in source_paths.items()},ensure_ascii=False,indent=2),encoding='utf-8')
    if args.pdf:write_pdf(sections,source_paths,output/'selected_results.pdf')
    unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest for p,digest in source_files.items())
    manifest={'selection_rule':'latest_valid_batch_and_latest_distinct_four_lidar_case' if selection is None else 'explicit_selection',
              'selected':source_paths,'missing_sections':missing,'rejected':rejected,
              'source_sha256':source_files,'source_files_unchanged':unchanged,
              'analysis_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'runtime_versions':{'numpy':np.__version__,'matplotlib':matplotlib.__version__}}
    (output/'sources.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    if not unchanged:raise RuntimeError('Kaynak sonuç dosyası değişti.')
    for name,paths in source_paths.items():print(name+': '+', '.join(Path(p).name for p in paths))
    print('Kaynak sonuç dosyaları korunmuş: PASS')
    print(f'Grafik sayısı: {sum(len(s["charts"]) for s in sections)}')
    print(f'Sonuç klasörü: {output}')
    if missing:print('Eksik bölümler: '+', '.join(missing))


if __name__=='__main__':
    main()
