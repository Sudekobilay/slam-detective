"""Backward-compatible EXP-43 entrypoint; implementation lives in application."""
from application.robustness import read_config, pose_of, RobustnessWorld, preflight, run_trial, path_cost, trial_row, paired_rows, main, execute_protocol

if __name__ == "__main__":
    main()
