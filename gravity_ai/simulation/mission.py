"""Smooth example thrust timeline, expressed in real physical seconds."""
from math import isfinite

def smoothstep(u):
    u=max(0.,min(1.,u));return u*u*(3-2*u)

def axial_acceleration_g(time_s:float,duration_s:float,peak_g:float=.8):
    if duration_s<=0: raise ValueError('duration must be positive')
    if not (0<=peak_g<=1.5) or not isfinite(peak_g): raise ValueError('invalid peak_g')
    u=max(0.,min(1.,time_s/duration_s))
    edge=.04
    if u<.30:
        return peak_g * smoothstep(u/edge) * smoothstep((.30-u)/edge)
    if u<.70: return 0.
    return -peak_g * smoothstep((u-.70)/edge) * smoothstep((1-u)/edge)
