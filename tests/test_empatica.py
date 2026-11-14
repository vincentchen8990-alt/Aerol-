import numpy as np
from gravity_ai.data.empatica import read_e4_csv,session_features

def make_signal(path,fs,n,values):
    path.write_text('2024-01-01T00:00:00+00:00\n'+str(fs)+'\n'+'\n'.join(','.join(map(str,values)) for _ in range(n)))

def test_read_and_align_mock_sensor_fixture(tmp_path):
    # Test fixture is artificial, not collected human physiology.
    make_signal(tmp_path/'HR.csv',1,120,[72]);make_signal(tmp_path/'EDA.csv',4,480,[1.2]);make_signal(tmp_path/'TEMP.csv',4,480,[34]);make_signal(tmp_path/'ACC.csv',32,3840,[0,64,0])
    t,fs,arr=read_e4_csv(tmp_path/'HR.csv');assert fs==1 and arr.shape==(120,1)
    df=session_features(tmp_path,window_s=15)
    assert len(df)==8 and df.hr_mean.iloc[0]==72 and df.acc_norm_mean_g.iloc[0]==1
