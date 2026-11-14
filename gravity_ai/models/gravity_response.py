"""Explicit unvalidated response: do not manufacture gravity treatment labels."""
def propose_personalized_gravity(health_history,gravity_exposure=None):
    return {'status':'insufficient_evidence','recommended_gravity_g':None,
            'message':'No validated longitudinal individual gravity-response model is available. Use manually selected simulation target.',
            'data_type':'research_uncertainty','requested_autonomous_adjustment':False}
