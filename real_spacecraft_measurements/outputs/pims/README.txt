NASA ISS MAMS measured acceleration forecasting
================================================
Validation-selected model: mlp_residual
All inputs and outputs use g (multiply by 1e6 for micro-g).
Input exactly eight 3-axis observations, oldest first, 16 seconds apart.
The first and last input observations are 112 seconds apart.
Output forecasts the sample 16 seconds after the last input.

Run from this directory:
  python inference.py --history example_history_g.json --steps 15

Python use:
  from inference import load_model, predict_one, rollout
  model = load_model('selected_model.json')
  forecast_g = predict_one(model, history_8_by_3_g)

Use training_report.json for actual metrics, source URLs, limitations and checks.
selection_frozen_before_test.json records the decision before test evaluation.
held_out_predictions.csv contains every evaluated one-step test prediction.
fitted_models.joblib requires the recorded sklearn version; JSON inference needs only NumPy.
This forecasts filtered ISS disturbances; it is not a rotating-ring or actuator-response model.
