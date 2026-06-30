import os
import yaml

try:
    from ament_index_python.packages import get_package_share_directory
except Exception:  # ROS2 is optional outside runtime.
    get_package_share_directory = None


class ConfigMixin:
    def load_config(self, config_path=None):
            if config_path is None:
                try:
                    pkg_share = get_package_share_directory("learning_preview_controller")
                    config_path = os.path.join(pkg_share, "config", "preview_params.yaml")
                except Exception:
                    pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
                    config_path = os.path.join(pkg_root, "config", "preview_params.yaml")

            if not os.path.exists(config_path):
                raise FileNotFoundError(f"MPC config file not found: {config_path}")

            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)

            return cfg
