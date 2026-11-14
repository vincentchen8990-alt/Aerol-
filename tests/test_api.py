from fastapi.testclient import TestClient
from gravity_ai.api.app import app
client=TestClient(app)
def test_api_health_and_catalog():
    assert client.get('/api/health').status_code==200
    assert client.get('/api/catalog').json()['counts']['signals']==9
    assert client.get('/api/personalized-gravity').json()['status']=='insufficient_evidence'
def test_simulate_api():
    resp=client.post('/api/simulate',json={'duration_s':120,'dt_s':2,'max_points':50})
    assert resp.status_code==200,resp.text
    data=resp.json()
    assert data['data_type']=='SYNTHETIC_SPACECRAFT_DYNAMICS'
    assert len(data['samples'])>=50
    assert 'render_ring_angle_rad' in data['samples'][0]
def test_static_viewer():
    r=client.get('/')
    assert r.status_code==200 and 'Orbital Lab' in r.text

def test_previous_research_restored():
    r=client.get('/api/prior-research')
    assert r.status_code==200 and r.json()['source_records']==40
