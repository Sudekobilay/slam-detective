"""SLAM-Detective ortak komut girişi. Örnek: python main.py --help"""
import argparse
import importlib
import sys


TASKS = {
    'compare':'experiments.exp_38_algorithms',
    'diagnose':'experiments.exp_33_hypothesis_motion',
    'gps':'experiments.exp_35_joint_gps',
    'layouts':'experiments.exp_36_sensor_layouts',
    'faults':'experiments.exp_34_fault_sources',
    'demo':'visualization.motion_demo',
    'active-demo':'visualization.active_demo',
    'real':'experiments.exp_41_real_data',
    'active':'experiments.exp_40_configurable_active',
}


def main():
    parser = argparse.ArgumentParser(
        description='SLAM-Detective: karşılaştırma, teşhis, GPS ve hareketli gösterim.',
        epilog='Alt komut yardımı: python main.py compare --help',
    )
    parser.add_argument('task',choices=TASKS,help='Çalıştırılacak görev')
    parser.add_argument('arguments',nargs=argparse.REMAINDER,help='Göreve aktarılacak seçenekler')
    args = parser.parse_args()
    if args.task not in ('compare','layouts','demo','active','real','active-demo') and args.arguments:
        if args.arguments in (['--help'],['-h']):
            print(f'{args.task}: mevcut deney varsayılan ayarlarla çalışır; ek seçenek kabul etmez.')
            return
        parser.error(f'{args.task} görevi ek seçenek kabul etmiyor.')
    previous = sys.argv
    try:
        sys.argv = [TASKS[args.task],*args.arguments]
        importlib.import_module(TASKS[args.task]).main()
    finally:
        sys.argv = previous


if __name__=='__main__':
    main()
