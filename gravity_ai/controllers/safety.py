"""Research only: no autonomous physiology-driven medical prescription."""
from dataclasses import dataclass
@dataclass
class GravityRequest:
    target_g:float
    approved_by_human:bool=False
    evidence_validated:bool=False

def evaluate_research_request(request:GravityRequest,simulation_limits=(.3,1.5)):
    lo,hi=simulation_limits
    if not lo<=request.target_g<=hi:return {'allowed':False,'reason':'Outside configured demonstration limits'}
    if not request.approved_by_human:return {'allowed':False,'reason':'Manual approval required'}
    if not request.evidence_validated:return {'allowed':False,'reason':'Biological prescription unsupported; manual simulation only'}
    return {'allowed':True,'reason':'Research-mode request allowed by configuration; no hardware access'}
