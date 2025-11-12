import yaml

def create_detail_settings(save_path: str = 'detail_settings.yaml'):
    """
    Create a detail settings file
    """
    detail_settings = {
        'prominent_factor': 5,
        'min_distance': 5,
        'convergence_threshold': 1e-5,
        'max_iterations': 20
    }
    with open(save_path, 'w') as f:
        yaml.dump(detail_settings, f)