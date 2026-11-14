"""Request bodies shared by the simulation and learned-prediction endpoints."""
from pydantic import BaseModel, Field


class SimRequest(BaseModel):
    radius_m: float = Field(default=100, gt=5, le=1000)
    duration_s: float = Field(default=1200, ge=60, le=86400)
    dt_s: float = Field(default=2.0, ge=.1, le=10)
    peak_thrust_g: float = Field(default=.8, ge=0, le=1.5)
    target_gravity_g: float = Field(default=1., ge=.1, le=1.5)
    ring_inertia_kg_m2: float = Field(default=4e9, gt=0)
    hub_inertia_kg_m2: float = Field(default=1e10, gt=0)
    motor_torque_limit_nm: float = Field(default=8e6, gt=0)
    max_points: int = Field(default=850, ge=50, le=2000)
    controller: str = 'pid'
