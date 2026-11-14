"""Idealized fixed-axis habitat dynamics. Units SI except input *_g ratios.

Educational computational research model only: no flight or medical qualification.
The inertial ring speed—not encoder speed relative to a moving hub—sets centrifugal loading.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math

G0 = 9.80665


@dataclass
class SpacecraftParams:
    radius_m: float = 100.0
    ring_inertia_kg_m2: float = 4.0e9
    hub_inertia_kg_m2: float = 1.0e10
    motor_torque_limit_nm: float = 8.0e6
    motor_power_limit_w: float = 3.0e6
    motor_slew_nm_s: float = 8.0e5
    bearing_drag_nms: float = 2.0e5
    crew_height_m: float = 1.75
    reference_gravity_m_s2: float = G0

    def validate(self):
        for name in ('radius_m', 'ring_inertia_kg_m2', 'hub_inertia_kg_m2', 'motor_torque_limit_nm', 'motor_power_limit_w', 'motor_slew_nm_s', 'reference_gravity_m_s2'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if self.crew_height_m < 0 or self.crew_height_m >= self.radius_m:
            raise ValueError('crew_height_m must be between 0 and radius')
        if self.bearing_drag_nms < 0: raise ValueError('drag cannot be negative')
        return self


@dataclass
class SpacecraftState:
    time_s: float = 0.0
    ring_angle_rad: float = 0.0
    hub_angle_rad: float = 0.0
    ring_omega_rad_s: float = 0.0
    hub_omega_rad_s: float = 0.0
    applied_torque_nm: float = 0.0
    position_m: float = 0.0
    velocity_m_s: float = 0.0

    @property
    def encoder_omega_rad_s(self):
        return self.ring_omega_rad_s - self.hub_omega_rad_s


def equilibrium_omega(radius_m: float, target_m_s2: float, axial_accel_m_s2: float = 0.0):
    """Ideal positive spin rate for steady orthogonal axial and radial loads.

    None means no feasible steady solution from perpendicular rotation.
    """
    if radius_m <= 0 or target_m_s2 < 0: raise ValueError('Invalid radius or target')
    if abs(axial_accel_m_s2) > target_m_s2 + 1e-12: return None
    return math.sqrt(math.sqrt(max(0.0, target_m_s2**2 - axial_accel_m_s2**2)) / radius_m)


def omega_to_rpm(omega: float):
    return omega * 60.0 / (2 * math.pi)


def crew_acceleration(params: SpacecraftParams, state: SpacecraftState, axial_accel_m_s2: float, ring_alpha_rad_s2: float = 0.0, radius_m: float | None = None):
    """Apparent 'down' vector in local rotating radial/tangential/axis coordinates.

    Assumes crew is stationary relative to fixed-axis habitat; Coriolis from walking
    requires additional -2 omega cross relative-velocity term, computed separately.
    """
    r = params.radius_m if radius_m is None else radius_m
    rad = r * state.ring_omega_rad_s**2
    tan = -r * ring_alpha_rad_s2
    az = -axial_accel_m_s2
    return (rad, tan, az)


def magnitude(v):
    return math.sqrt(sum(float(t)**2 for t in v))


def coriolis_accel(omega_rad_s: float, v_radial: float, v_tangential: float):
    """Fictitious transverse acceleration in rotating basis: -2 Omega x v."""
    return (2*omega_rad_s*v_tangential, -2*omega_rad_s*v_radial)


def step(params: SpacecraftParams, state: SpacecraftState, commanded_torque_nm: float, axial_accel_m_s2: float, dt_s: float, external_ring_torque_nm: float = 0.0, external_hub_torque_nm: float = 0.0):
    """One symplectic Euler step. Returns (new_state, telemetry).

    Motors exchange opposite torques between ring and hub. Friction also exchanges
    opposite torques. The hub is NOT artificially held inertially stationary.
    """
    if dt_s <= 0 or not math.isfinite(dt_s): raise ValueError('dt_s must be positive')
    params.validate()
    req = max(-params.motor_torque_limit_nm, min(params.motor_torque_limit_nm, float(commanded_torque_nm)))
    rel = state.encoder_omega_rad_s
    allowed = params.motor_slew_nm_s * dt_s
    slewed = max(state.applied_torque_nm-allowed, min(state.applied_torque_nm+allowed, req))
    max_power_torque = params.motor_power_limit_w / max(abs(rel), .005)
    applied = max(-max_power_torque, min(max_power_torque, slewed))
    applied = max(-params.motor_torque_limit_nm, min(params.motor_torque_limit_nm, applied))
    drag_on_ring = -params.bearing_drag_nms * rel
    alpha_ring = (applied + drag_on_ring + external_ring_torque_nm) / params.ring_inertia_kg_m2
    alpha_hub = (-applied - drag_on_ring + external_hub_torque_nm) / params.hub_inertia_kg_m2
    wr = state.ring_omega_rad_s + dt_s*alpha_ring
    wh = state.hub_omega_rad_s + dt_s*alpha_hub
    n = SpacecraftState(
        time_s=state.time_s+dt_s,
        ring_angle_rad=state.ring_angle_rad + dt_s*wr,
        hub_angle_rad=state.hub_angle_rad + dt_s*wh,
        ring_omega_rad_s=wr, hub_omega_rad_s=wh,
        applied_torque_nm=applied,
        position_m=state.position_m + state.velocity_m_s*dt_s + .5*axial_accel_m_s2*dt_s**2,
        velocity_m_s=state.velocity_m_s+axial_accel_m_s2*dt_s,
    )
    gfoot = crew_acceleration(params, n, axial_accel_m_s2, alpha_ring)
    ghead = crew_acceleration(params, n, axial_accel_m_s2, alpha_ring, params.radius_m-params.crew_height_m)
    telem = {
        'time_s':n.time_s, 'ring_omega_rad_s':wr, 'hub_omega_rad_s':wh,
        'encoder_omega_rad_s':n.encoder_omega_rad_s,'ring_rpm':omega_to_rpm(wr),
        'hub_rpm':omega_to_rpm(wh), 'requested_motor_torque_nm':float(commanded_torque_nm),
        'applied_motor_torque_nm':applied, 'motor_power_w':abs(applied*rel),
        'ring_alpha_rad_s2':alpha_ring, 'hub_alpha_rad_s2':alpha_hub,
        'axial_accel_m_s2':axial_accel_m_s2, 'crew_gravity_radial_m_s2':gfoot[0],
        'crew_gravity_tangential_m_s2':gfoot[1], 'crew_gravity_axial_m_s2':gfoot[2],
        'crew_gravity_m_s2':magnitude(gfoot), 'crew_gravity_g':magnitude(gfoot)/G0,
        'head_gravity_m_s2':magnitude(ghead), 'head_to_foot_gradient_m_s2':magnitude(gfoot)-magnitude(ghead),
        'crew_lateral_m_s2':abs(gfoot[1]),
        'position_m':n.position_m,'velocity_m_s':n.velocity_m_s,
        'angular_momentum_kg_m2_s':params.ring_inertia_kg_m2*wr+params.hub_inertia_kg_m2*wh,
        'torque_saturated': abs(applied-float(commanded_torque_nm)) > 1e-5,
        'render_ring_angle_rad':n.ring_angle_rad,
    }
    return n,telem


def to_dict(params): return asdict(params)
