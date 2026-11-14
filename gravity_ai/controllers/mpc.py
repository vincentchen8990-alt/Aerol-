"""Small research MPC for fixed-axis ring with finite torque; falls back to PID."""
from scipy.optimize import minimize
from gravity_ai.controllers.pid import TorquePID
from gravity_ai.simulation.physics import step,equilibrium_omega,G0

class GravityMPC:
    def __init__(self,horizon=5):
        self.horizon=horizon
        self.fallback=TorquePID()
        self.previous=0.
    def command(self,params,state,axial_accel_m_s2,target_m_s2,dt_s):
        target=equilibrium_omega(params.radius_m,target_m_s2,axial_accel_m_s2)
        if target is None: return 0.,None
        scale=params.motor_torque_limit_nm
        def objective(x):
            s=state;score=0.
            for k,z in enumerate(x):
                s,out=step(params,s,float(z)*scale,axial_accel_m_s2,dt_s)
                score+= 10.*((out['crew_gravity_m_s2']-target_m_s2)/G0)**2 + 2*(s.ring_omega_rad_s-target)**2 + .003*float(z)**2
                if k: score+=.004*(x[k]-x[k-1])**2
            return score
        result=minimize(objective,[self.previous/scale]*self.horizon,method='L-BFGS-B',bounds=[(-1,1)]*self.horizon,options={'maxiter':35})
        if result.success:
            self.previous=float(result.x[0]*scale)
            return self.previous,target
        return self.fallback.command(params,state,axial_accel_m_s2,target_m_s2,dt_s)
