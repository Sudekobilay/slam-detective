"""Frozen healthy calibration shared by held-out protocols."""
from application.layout_service import calibrate_layout, run_layout
from estimation.diagnostic_motion import CANDIDATES
from configuration.protocols import CALIBRATION_SEEDS

def calibrate_readout(data):
    """Yalnız özgün ortamın sağlıklı kayıtlarından alarm ve okuma eşikleri."""
    config, alignment = calibrate_layout(data,'ekf',40)
    improvements, gaps, rms = [],[],[]
    for normal in CANDIDATES:
        for seed in CALIBRATION_SEEDS:
            agent,_,_,_,reason = run_layout(data,config,normal=normal,fault='healthy',seed=seed)
            if reason is not None:
                raise RuntimeError('Sağlıklı kalibrasyon tamamlanamadı.')
            for frame in agent.raw_history:
                if frame['status'] != 'evaluated':
                    continue
                best,runner = frame['models'][:2]
                improvements.append(best['improvement_m2'])
                gaps.append(runner['cost_m2']-best['cost_m2'])
                rms.append(best['max_sensor_rms_m'])
    if not rms:
        raise RuntimeError('Hipotez okuma kalibrasyonu yetersiz.')
    thresholds = {'minimum_improvement_m2':2*max(improvements)+1e-6,
                  'minimum_model_gap_m2':2*max(gaps)+1e-6}
    max_rms = max(rms)+.01
    return config,thresholds,max_rms,{
        'alignment':alignment, 'diagnosis_thresholds':thresholds,
        'diagnosis_max_sensor_rms_m':max_rms, 'healthy_raw_frames':len(rms),
        'calibration_environment':'original_demo_only',
        'note':'Engineering margins derived only from healthy original-environment records; frozen before held-out trials.',
    }
