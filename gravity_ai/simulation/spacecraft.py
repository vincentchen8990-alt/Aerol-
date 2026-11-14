"""Mission runner; outputs are true simulated state, not personal health data."""
from dataclasses import dataclass
from gravity_ai.simulation.physics import SpacecraftParams,SpacecraftState,G0,equilibrium_omega,omega_to_rpm,step
from gravity_ai.simulation.mission import axial_acceleration_g
from gravity_ai.controllers.pid import TorquePID

@dataclass
class MissionConfig:
    duration_s:float=1200.
    dt_s:float=2.
    peak_thrust_g:float=.8
    target_gravity_g:float=1.
    initialize_spinning:bool=True


def simulate(params:SpacecraftParams|None=None,mission:MissionConfig|None=None,controller=None, max_points:int|None=None):
    params=(params or SpacecraftParams()).validate()
    m=mission or MissionConfig()
    if m.dt_s<=0 or m.duration_s<=0 or m.duration_s/m.dt_s>200000:raise ValueError('invalid mission duration or step')
    if not(0<=m.peak_thrust_g<=1.5) or not(.1<=m.target_gravity_g<=2.):raise ValueError('invalid thrust or target')
    ctrl=controller or TorquePID()
    initial=equilibrium_omega(params.radius_m,m.target_gravity_g*G0,0.) if m.initialize_spinning else 0.
    s=SpacecraftState(ring_omega_rad_s=initial,hub_omega_rad_s=0.)
    rows=[]
    n=round(m.duration_s/m.dt_s)
    every=max(1,n//max_points) if max_points else 1
    for i in range(n):
        at= axial_acceleration_g(s.time_s,m.duration_s,m.peak_thrust_g)*G0
        torque,ref=ctrl.command(params,s,at,m.target_gravity_g*G0,m.dt_s)
        s,rec=step(params,s,torque,at,m.dt_s)
        if i%every==0 or i==n-1:
            rec['target_gravity_g']=m.target_gravity_g
            rec['target_rpm']=omega_to_rpm(ref) if ref is not None else None
            rec['feasible']=ref is not None
            rec['flight_phase']='acceleration' if rec['axial_accel_m_s2']>0.01 else ('braking' if rec['axial_accel_m_s2']<-.01 else 'coasting')
            rows.append(rec)
    return rows
