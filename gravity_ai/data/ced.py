"""Input/output datasets from Uppsala Coupled Electric Drives benchmark.

Requires original DATAPRBS.csv and DATAUNIF.csv user-downloaded with source/license.
No motor torque or astronaut-health data are present in these CSVs.
"""
from pathlib import Path
import pandas as pd

def load_ced(root):
    root=Path(root)
    pairs=[('DATAPRBS.csv',a,b) for a,b in [('u1','z1'),('u2','z2'),('u3','z3')]]
    pairs += [('DATAUNIF.csv',a,b) for a,b in [('u11','z11'),('u12','z12')]]
    output=[]
    for filename,u,z in pairs:
        path=root/filename
        if not path.exists():raise FileNotFoundError(f'Get the original {filename} from Uppsala University and store at {path}')
        d=pd.read_csv(path)
        if not set((u,z)).issubset(d.columns):raise ValueError(f'{filename}: expected measured columns {u},{z}')
        frame=d[[u,z]].rename(columns={u:'voltage_input_V',z:'recorded_speed_signal_dataset_units'}).copy()
        frame['relative_time_s']=frame.index*.02
        frame['experiment_id']=filename+':'+u+'/'+z
        frame['source']='Uppsala_CoupledElectricDrives'
        output.append(frame)
    return pd.concat(output,ignore_index=True)
