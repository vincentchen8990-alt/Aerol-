from gravity_ai.simulation.spacecraft import simulate,MissionConfig
from gravity_ai.simulation.physics import SpacecraftParams
from gravity_ai.controllers.safety import GravityRequest,evaluate_research_request

def test_mission_timeline_and_output():
    rows=simulate(SpacecraftParams(),MissionConfig(duration_s=120,dt_s=1,peak_thrust_g=.5),max_points=100)
    assert 90<=len(rows)<=120
    assert rows[-1]['time_s']==120
    assert all(row['crew_gravity_g']>=0 for row in rows)
    assert all(abs(row['ring_rpm'])<100 for row in rows)

def test_unvalidated_gravity_dose_denied():
    assert not evaluate_research_request(GravityRequest(target_g=.9,approved_by_human=True))['allowed']
