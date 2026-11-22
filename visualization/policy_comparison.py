"""Four saved EXP-43 policies at the same clock; no new simulation or selection."""
from storage.runs import validate_run
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from datetime import datetime,timezone
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
from matplotlib.patches import Rectangle
import numpy as np
from evaluation.diagnosis import diagnosis_readout

POLICIES=('passive','fixed_straight','fixed_arc','adaptive')
LABELS=('Pasif','Sabit düz','Sabit yay','Adaptif')


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def frame_status(history,summary,time_s,calibration):
    trigger=summary['trigger_time_s']
    if trigger is None or time_s<trigger-1e-9:return 'Henüz doğrulanmış alarm yok','Normal hareket'
    readout=diagnosis_readout(history,trigger,min(time_s,trigger+2.),
                            calibration['diagnosis_thresholds'],calibration['diagnosis_max_sensor_rms_m'])
    model=readout['confirmed_model']
    motion=summary['selected_motion'] or 'normal rotaya devam'
    label=f'Kaynak adayı doğrulandı: {model}' if model else 'Kaynak henüz doğrulanmadı'
    if time_s>=trigger+2.-1e-9:label+=' | pencere sonu'
    return label,motion


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--environment',default='corridor');parser.add_argument('--start',default='nominal')
    parser.add_argument('--noise',default='ideal');parser.add_argument('--normal',default='arc')
    parser.add_argument('--seed',type=int);parser.add_argument('--yaw',type=float,default=5.)
    parser.add_argument('--fps',type=int,default=8)
    args=parser.parse_args(argv)
    if not 1<=args.fps<=20:parser.error('fps must be 1..20')
    validate_run(args.run, experiment_id='exp_43')
    run=args.run.resolve();protocol=json.loads((run/'protocol.json').read_text(encoding='utf-8-sig'))
    seed=args.seed if args.seed is not None else protocol['seeds'][0]
    base=(run/args.environment/args.start/args.noise/args.normal/f'seed_{seed}'/f'yaw_{args.yaw:g}').resolve()
    if not base.is_relative_to(run):raise ValueError('Case must belong to run.')
    calibration=json.loads((run/'calibration.json').read_text(encoding='utf-8-sig'));geometry=protocol['environment_geometry'][args.environment]
    trials=[];sources=[run/'protocol.json',run/'calibration.json']
    for policy in POLICIES:
        folder=base/policy
        files=[folder/n for n in ('ground_truth.csv','estimate.csv','summary.json','raw_hypotheses.jsonl')]
        sources.extend(files)
        truth=read_csv(files[0]);estimate=read_csv(files[1]);summary=json.loads(files[2].read_text(encoding='utf-8-sig'))
        raw=[json.loads(l) for l in files[3].read_text(encoding='utf-8-sig').splitlines() if l.strip()]
        trials.append((truth,estimate,summary,raw))
    before={str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    max_time=max(float(t[0][-1]['time_s']) for t in trials)
    frames=sorted(set([0.,max_time]+[float(t[0][i]['time_s']) for t in trials for i in range(len(t[0]))]))
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    alltruth=[r for t in trials for r in t[0]];xs=[float(r['x_m']) for r in alltruth];ys=[float(r['y_m']) for r in alltruth]
    xlim=(min(xs)-.45,max(xs)+.45);ylim=(min(ys)-.45,max(ys)+.45)
    out=run/('demo_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));out.mkdir()
    def update(index):
        time_s=frames[index]
        for ax,label,(truth,estimate,summary,raw) in zip(axes.flat,LABELS,trials):
            ax.clear()
            for rect in [geometry['boundary'],*geometry['obstacles']]:
                ax.add_patch(Rectangle((rect['x_min'],rect['y_min']),rect['x_max']-rect['x_min'],rect['y_max']-rect['y_min'],fill=False,edgecolor='#94a3b8'))
            selected=[r for r in truth if float(r['time_s'])<=time_s+1e-9]
            estimated=[r for r in estimate if float(r['time_s'])<=time_s+1e-9]
            ax.plot([float(r['x_m']) for r in selected],[float(r['y_m']) for r in selected],color='#d89b40',label='Gerçek rota (çizim)')
            ax.plot([float(r['x_m']) for r in estimated],[float(r['y_m']) for r in estimated],'--',color='#2563eb',label='Kestirim')
            last=selected[-1];angle=float(last['theta_rad']);x,y=float(last['x_m']),float(last['y_m'])
            ax.plot(x,y,'o',color='#0f766e');ax.arrow(x,y,.12*math.cos(angle),.12*math.sin(angle),width=.005,color='#0f766e')
            state,motion=frame_status(raw,summary,time_s,calibration)
            ax.set_title(f'{label} | t={time_s:.1f} s\n{state}\nHareket: {motion}',fontsize=9)
            ax.set(xlim=xlim,ylim=ylim,xlabel='x (m)',ylabel='y (m)');ax.set_aspect('equal');ax.grid(alpha=.2)
            ax.legend(loc='lower right',fontsize=7)
        fig.suptitle(f'EXP-43 | {args.environment} / {args.start} / {args.noise} / {args.normal} / {args.yaw:+g}° / seed {seed}\nKayıttan gösterim; gerçek rota yalnız çizim ve çevrimdışı değerlendirme içindir.',fontsize=10)
    animation=FuncAnimation(fig,update,frames=len(frames),interval=1000/args.fps)
    animation.save(out/'policy_comparison.gif',writer=PillowWriter(fps=args.fps))
    update(len(frames)-1);fig.savefig(out/'final_frame.png',dpi=150);plt.close(fig)
    if before!={str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}:raise RuntimeError('Sources changed.')
    (out/'sources.json').write_text(json.dumps({'source_hashes':before,'case':str(base.relative_to(run)),'source_unchanged':True,'new_simulation':False},indent=2))
    print('Demo:',out)

if __name__=='__main__':main()
