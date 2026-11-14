"""Synthetic demo profile; all values illustrative, not astronaut health measurements."""
def synthetic_profile(index=1):
    return {'subject_id':f'DEMO-{index:03d}','data_type':'SYNTHETIC_DEMO','heart_rate_bpm':68+index*2,
            'systolic_bp_mmHg':118+index*2,'diastolic_bp_mmHg':77,
            'spo2_pct':98,'skin_temperature_c':33.2,'bone_density_g_cm2':0.99,
            'bone_baseline_g_cm2':1.02,'sleep_hours':7.4,'activity_level':'illustrative',
            'note':'These are made-up interface demonstration values, not a physiological prediction or medical recommendation.'}
