"""Archived measured-ISS endpoints: MARS-S centrifuge RPM and NASA PIMS acceleration.

ARCHIVED MEASUREMENTS, NOT LIVE TELEMETRY. One-step predictions only, from caller-supplied
measured histories whose timestamps must prove the cadence, or from replayed flight records.
Invalid input or a missing/altered model returns an explicit unavailable result (HTTP 422/503)
instead of a simulated substitute. Nothing here selects a motor command, and the simulated ring
dynamics endpoints (/api/simulate, /api/dynamics/compare) are untouched.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from gravity_ai.measured import service
from gravity_ai.measured.assets import AssetError
from gravity_ai.measured.inference import MeasuredInputError
from gravity_ai.measured.service import ModelUnavailable

router = APIRouter(prefix='/api/measured', tags=['archived measured ISS data'])


class RpmPredictRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    history_rpm: Any = Field(..., description='Eight measured MARS-S RPM values, oldest to newest (shape (8,)).')
    timestamps: list[Any] | None = Field(None, description='Eight ISO-8601 timestamps exactly 1 s apart, as recorded.')
    steps: int = Field(1, description='Must be 1: the RPM model failed recursive forecasting; one-step only.')


class PimsPredictRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    history_g: Any = Field(..., description='Eight [ax_g, ay_g, az_g] measurements in standard gravity g, oldest to newest (shape (8, 3)).')
    timestamps_utc: list[Any] | None = Field(None, description='Eight explicit-UTC ISO-8601 timestamps exactly 16 s apart.')
    steps: int = Field(1, description='Must be 1: this integration exposes only the next 16-second sample.')


def _unavailable(status_code: int, detail: dict, **extra):
    raise HTTPException(status_code, {**detail, 'available': False, 'status': 'unavailable', 'mode': service.MODE,
                                      'live_telemetry': False, **extra})


@router.get('/status')
def measured_status():
    return service.status()


@router.get('/hardware')
def measured_hardware():
    try:
        return service.hardware()
    except AssetError as e:
        _unavailable(503, e.detail())


@router.post('/rpm/predict')
def rpm_predict(req: RpmPredictRequest):
    if req.steps != 1:
        _unavailable(422, {'code': 'one_step_only', 'message': 'The MARS-S RPM model is one-step only: its recursive '
                           'forecasts failed evaluation (unstable/divergent). Supply fresh measured history for each step.'},
                     forecast_rpm=None)
    try:
        return service.rpm_one_step(req.history_rpm, req.timestamps)
    except MeasuredInputError as e:
        _unavailable(422, e.detail(), forecast_rpm=None)
    except ModelUnavailable as e:
        _unavailable(503, {'code': 'model_unavailable', 'message': f'MARS-S RPM model unavailable: {e}', 'model': e.status},
                     forecast_rpm=None)


@router.post('/pims/predict')
def pims_predict(req: PimsPredictRequest):
    if req.steps != 1:
        _unavailable(422, {'code': 'one_step_only', 'message': 'Only the next 16-second sample is provided; supply '
                           'fresh measured history for each step.'}, forecast_g=None)
    try:
        return service.pims_one_step(req.history_g, req.timestamps_utc)
    except MeasuredInputError as e:
        _unavailable(422, e.detail(), forecast_g=None)
    except ModelUnavailable as e:
        _unavailable(503, {'code': 'model_unavailable', 'message': f'PIMS acceleration model unavailable: {e}', 'model': e.status},
                     forecast_g=None)


@router.get('/rpm/replay')
def rpm_replay():
    try:
        return service.rpm_replay()
    except AssetError as e:
        _unavailable(503, {**e.detail(), 'code': 'replay_unavailable', 'reason_code': e.code})


@router.get('/pims/replay')
def pims_replay():
    try:
        return service.pims_replay()
    except AssetError as e:
        _unavailable(503, {**e.detail(), 'code': 'replay_unavailable', 'reason_code': e.code})
