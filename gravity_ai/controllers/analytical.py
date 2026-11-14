"""Equilibrium reference speed, not an instantaneous achievable command."""
from gravity_ai.simulation.physics import equilibrium_omega, omega_to_rpm

def target_spin(radius_m, target_m_s2, axial_accel_m_s2):
    w=equilibrium_omega(radius_m,target_m_s2,axial_accel_m_s2)
    return {'feasible': w is not None, 'omega_rad_s':w,'rpm':omega_to_rpm(w) if w is not None else None}
