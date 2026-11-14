"""Reproduce a mission, export telemetry CSV, summary JSON, and an illustrative plot."""
from pathlib import Path
import json,csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gravity_ai.data.catalog_loader import project_root
from gravity_ai.simulation.spacecraft import simulate,MissionConfig
from gravity_ai.simulation.physics import SpacecraftParams,G0

def main():
    root=project_root();rows=simulate(SpacecraftParams(),MissionConfig(duration_s=1200,dt_s=2,peak_thrust_g=.8,target_gravity_g=1.))
    out=root/'outputs'/'simulations';out.mkdir(parents=True,exist_ok=True)
    with (out/'mission_1200s.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    t=[x['time_s'] for x in rows]
    fig,ax=plt.subplots(figsize=(11,4));ax.plot(t,[x['crew_gravity_g'] for x in rows],label='Actual gravity');ax.plot(t,[x['target_gravity_g'] for x in rows],label='Target gravity');ax.set(xlabel='Physical mission time (s)',ylabel='Local crew-floor apparent gravity (G)',title='Idealized torque-limited spacecraft gravity control');ax.legend();fig.tight_layout();fig.savefig(out/'gravity_tracking.png',dpi=150);plt.close(fig)
    summary={'data_type':'SYNTHETIC_PHYSICS_NOT_MEDICAL','samples':len(rows),'mean_abs_error_g':sum(abs(x['crew_gravity_g']-1) for x in rows)/len(rows),'max_abs_error_g':max(abs(x['crew_gravity_g']-1) for x in rows),'files':['mission_1200s.csv','gravity_tracking.png']}
    (out/'mission_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
