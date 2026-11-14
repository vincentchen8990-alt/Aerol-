import math
import pytest
from gravity_ai.simulation.physics import *
from gravity_ai.simulation.mission import axial_acceleration_g

def test_ideal_rpm():
    for e,rpm in [(0,2.990),(.5,2.783),(.8,2.316)]:
        got=equilibrium_omega(100,G0,e*G0)
        assert omega_to_rpm(got)==pytest.approx(rpm,abs=.002)

def test_infeasible():assert equilibrium_omega(100,G0,1.2*G0) is None

def test_hub_reaction_and_angular_momentum():
    c=SpacecraftParams(bearing_drag_nms=0)
    s=SpacecraftState(ring_omega_rad_s=.3)
    n,_=step(c,s,1e6,0.,1.)
    assert n.ring_omega_rad_s>.3 and n.hub_omega_rad_s<0
    assert c.ring_inertia_kg_m2*n.ring_omega_rad_s+c.hub_inertia_kg_m2*n.hub_omega_rad_s==pytest.approx(c.ring_inertia_kg_m2*.3,rel=1e-12)

def test_local_gravity_and_head_gradient():
    c=SpacecraftParams()
    s=SpacecraftState(ring_omega_rad_s=equilibrium_omega(c.radius_m,G0))
    f=crew_acceleration(c,s,0.);h=crew_acceleration(c,s,0.,radius_m=c.radius_m-c.crew_height_m)
    assert magnitude(f)==pytest.approx(G0,rel=1e-12)
    assert magnitude(h)<magnitude(f)

def test_spin_transient_tangential():
    c=SpacecraftParams();s=SpacecraftState(ring_omega_rad_s=.30)
    n,rec=step(c,s,8e6,0.,1.)
    assert rec['ring_alpha_rad_s2']>0 and rec['crew_gravity_tangential_m_s2']<0
    assert rec['crew_gravity_m_s2']>rec['crew_gravity_radial_m_s2']

def test_braking_sign():
    assert axial_acceleration_g(900,1200,.8)<0
    assert axial_acceleration_g(500,1200,.8)==0

def test_coriolis_rotating_basis():
    a=coriolis_accel(0.3,1.0,0.0)
    assert a==(0.0,-.6)
