from gravity_ai.simulation.generate import generate_dataset
from gravity_ai.train.dynamics import train
from gravity_ai.train.dynamics_config import load_config


def test_small_synthetic_training(tmp_path):
    cfg, sources = load_config(None, overrides={
        'data.episodes': 20, 'data.steps_per_episode': 12, 'data.dt_values_s': [2.0],
        'train.dataset_dir': str(tmp_path / 'data'), 'train.output_root': str(tmp_path / 'runs'), 'train.epochs': 2})
    report = generate_dataset(cfg['data'], tmp_path / 'data', log=lambda *_: None)
    assert report['rows'] == 240 and report['data_type'] == 'SIMULATED_SPACECRAFT_DYNAMICS'
    assert report['split_episode_counts'] == {'train': 14, 'val': 3, 'test': 3}
    manifest = train(cfg, sources, tmp_path, run_id='small', log=lambda *_: None)
    assert manifest['status'] == 'completed' and manifest['start_mode'] == 'from_scratch'
    assert manifest['dataset']['split_episode_counts'] == {'train': 14, 'val': 3, 'test': 3}
    assert (tmp_path / 'runs' / 'small' / 'checkpoint_best.pt').exists()
