"""Research API: simulation and catalog services, no live hardware or medical control."""
from __future__ import annotations
from pathlib import Path
from fastapi import FastAPI,HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel,Field
from gravity_ai.data.catalog_loader import validate,project_root
from gravity_ai.simulation.physics import SpacecraftParams,G0
from gravity_ai.simulation.spacecraft import simulate,MissionConfig
from gravity_ai.simulation.astronaut import synthetic_profile
from gravity_ai.models.gravity_response import propose_personalized_gravity
from gravity_ai.data.prior_research import coverage_summary,load_reference
from gravity_ai.controllers.pid import TorquePID
from gravity_ai.controllers.mpc import GravityMPC
from gravity_ai.api.schemas import SimRequest
from gravity_ai.api.learned_dynamics import router as learned_dynamics_router,model_status
from gravity_ai.api.measured_iss import router as measured_iss_router
from gravity_ai.measured.service import models_summary as measured_models_summary

app=FastAPI(title='Personalized Gravity Research Simulator',version='0.1.0',
description='Computational prototype; NOT a medical device, flight controller, or validated health prescription.')
app.add_middleware(CORSMiddleware,allow_origins=['http://localhost:5173','http://127.0.0.1:5173'],allow_methods=['GET','POST'],allow_headers=['*'])


@app.get('/api/health')
def health():return {'service':'online','research_only':True,'medical_gravity_autonomy':False}

@app.get('/api/catalog')
def catalog():
    c=validate()
    return {'counts':c['counts'],'sources':c['sources'],'mapping_coverage':c['mapping_coverage'],
            'note':'Source catalogs only; participant physiological measurements not included'}

@app.get('/api/prior-research')
def previous_research():
    return coverage_summary()

@app.get('/api/models')
def models():
    root=project_root()
    return {'synthetic_dynamics_trained':(root/'outputs/checkpoints/dynamics_synthetic.pt').is_file(),
            'learned_dynamics_predictor':model_status(),
            # Archived measured-ISS models (MARS-S RPM, NASA PIMS): separate identifiers, separate files.
            'measured_iss':measured_models_summary(),
            'measured_wearable_trained':(root/'outputs/checkpoints/physiology_ridge.joblib').is_file(),
            'measured_bmd_trained':any((root/'outputs/checkpoints').glob('bone_*.joblib')),
            'physiological_gravity_prescription_validated':False}

@app.get('/api/demo-profile')
def demo_profile(index:int=1):
    if not 1<=index<=5:raise HTTPException(422,'index must be 1..5')
    return synthetic_profile(index)

@app.get('/api/personalized-gravity')
def gravity_recommendation():return propose_personalized_gravity([])

@app.post('/api/simulate')
def simulation(input:SimRequest):
    if input.duration_s/input.dt_s>60000:raise HTTPException(422,'too many integration steps for interactive request')
    if input.controller not in ('pid','mpc'):raise HTTPException(422,'controller must be pid or mpc')
    p=SpacecraftParams(radius_m=input.radius_m,ring_inertia_kg_m2=input.ring_inertia_kg_m2,
      hub_inertia_kg_m2=input.hub_inertia_kg_m2,motor_torque_limit_nm=input.motor_torque_limit_nm)
    m=MissionConfig(duration_s=input.duration_s,dt_s=input.dt_s,peak_thrust_g=input.peak_thrust_g,
        target_gravity_g=input.target_gravity_g)
    try:
        data=simulate(p,m,controller=TorquePID() if input.controller=='pid' else GravityMPC(),max_points=input.max_points)
    except (ValueError,ArithmeticError) as e:raise HTTPException(422,str(e))
    return {'data_type':'SYNTHETIC_SPACECRAFT_DYNAMICS','parameters':input.model_dump(),
      'source':'deterministic numerical spacecraft physics; not biological training data',
      'target_gravity_m_s2':input.target_gravity_g*G0,
      'samples':data,
      'summary':{'mean_abs_error_g':sum(abs(x['crew_gravity_g']-input.target_gravity_g) for x in data)/len(data),
      'max_abs_error_g':max(abs(x['crew_gravity_g']-input.target_gravity_g) for x in data),
      'torque_saturation_samples':sum(bool(x['torque_saturated']) for x in data)}}

# Learned-model predictions (display/comparison only; PID/MPC select every command).
app.include_router(learned_dynamics_router)
# Archived ISS measurements (MARS-S RPM 2016/2017, NASA PIMS Nov 2026): one-step forecasts only.
app.include_router(measured_iss_router)

# A dependency-free usable 3D projection is served here. The optional React/Three.js
# project lives in frontend/src and can be built separately with npm.
offline=project_root()/'frontend'/'offline'
app.mount('/',StaticFiles(directory=str(offline),html=True),name='browser_demo')
