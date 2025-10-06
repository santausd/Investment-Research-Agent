import os
def load_env(filepath="config/aai_520_proj.config"):
    """
    Loads environment variables from the aai_520_project.config.
    Each line in the file should be in the format KEY=VALUE.
    """
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    if key not in os.environ or not os.environ[key]: 
                        os.environ[key] = value
        print(f"Environment variables loaded from {filepath}")
    except FileNotFoundError:
        print(f"Error: Config file not found at {filepath}. Make sure it is in the project config directory.")
    except Exception as e:
        print(f"Error loading environment variables from {filepath}: {e}")
