"""Time-window physiological baselines are not validated astronaut health predictors."""
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def ridge_model():return make_pipeline(SimpleImputer(strategy='median',add_indicator=False),StandardScaler(),Ridge(alpha=2.))
def boosting_model():return make_pipeline(SimpleImputer(strategy='median'),HistGradientBoostingRegressor(max_iter=80,max_leaf_nodes=12,random_state=42))
