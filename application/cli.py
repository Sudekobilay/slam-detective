"""One command entry, with explicit argument forwarding and no sys.argv mutation."""
import argparse
import importlib

TASKS = {
    'compare': 'experiments.exp_38_algorithms',
    'diagnose': 'experiments.exp_33_hypothesis_motion',
    'gps': 'experiments.exp_35_joint_gps',
    'layouts': 'experiments.exp_36_sensor_layouts',
    'faults': 'experiments.exp_34_fault_sources',
    'demo': 'visualization.motion_demo',
    'active-demo': 'visualization.active_demo',
    'real': 'experiments.exp_41_real_data',
    'active': 'experiments.exp_40_configurable_active',
    'robustness': 'application.robustness',
    'report': 'analytics.robustness_report',
    'policy-demo': 'visualization.policy_comparison',
}
FIXED_TASKS = {'diagnose', 'gps', 'faults'}


def main(argv=None):
    parser = argparse.ArgumentParser(description='SLAM-Detective: simulation, diagnosis and evaluation.')
    parser.add_argument('task', choices=TASKS)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.task in FIXED_TASKS and args.arguments:
        if args.arguments in (['--help'], ['-h']):
            print(f'{args.task}: historical fixed protocol; no additional options.')
            return
        parser.error(f'{args.task} does not accept extra arguments.')
    try:
        return importlib.import_module(TASKS[args.task]).main(args.arguments)
    except (FileNotFoundError, ValueError) as error:
        parser.exit(2, f'Error: {error}\n')
