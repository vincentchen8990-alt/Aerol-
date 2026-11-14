"""Baseline torque-limited spin-rate feedback, never a flight-certified controller."""
from dataclasses import dataclass
from gravity_ai.simulation.physics import equilibrium_omega

@dataclass
class TorquePID:
    kp:float=.065
    ki:float=.0004
    integral_limit:float=.12
    integral:float=0.

    def command(self,params,state,axial_accel_m_s2,target_m_s2,dt_s):
        w=equilibrium_omega(params.radius_m,target_m_s2,axial_accel_m_s2)
        if w is None:
            # Infeasible: independently bounded controller keeps prior speed;
            # simulator reports infeasibility instead of pretending 1G achieved.
            return 0.,None
        err=w-state.ring_omega_rad_s
        self.integral=max(-self.integral_limit,min(self.integral_limit,self.integral+err*dt_s))
        feedforward=params.bearing_drag_nms*state.encoder_omega_rad_s
        tau=params.ring_inertia_kg_m2*(self.kp*err+self.ki*self.integral)+feedforward
        return max(-params.motor_torque_limit_nm,min(params.motor_torque_limit_nm,tau)),w
