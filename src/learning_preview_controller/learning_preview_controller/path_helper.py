import os

import yaml

try:
    from ament_index_python.packages import get_package_share_directory
except Exception:
    get_package_share_directory = None

from .path import PathMixin
from .utils import UtilsMixin


def resolve_config_path(config_path=None, default_config_name="preview_params.yaml"):
    """Resolve a controller config path inside learning_preview_controller/config."""

    if config_path:
        return str(config_path)

    if get_package_share_directory is not None:
        try:
            pkg_share = get_package_share_directory("learning_preview_controller")
            return os.path.join(pkg_share, "config", default_config_name)
        except Exception:
            pass

    pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(pkg_root, "config", default_config_name)


def load_yaml_config(config_path=None, default_config_name="preview_params.yaml"):
    resolved = resolve_config_path(config_path, default_config_name)
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"Config file not found: {resolved}")

    with open(resolved, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Config file is empty or invalid: {resolved}")

    return cfg, resolved


class PathOnlyHelper(PathMixin, UtilsMixin):
    """Generate/reference a path using only the `path:` section of a YAML file.

    This helper intentionally does not read Preview/MPC/Learning parameters.
    It is used by the Stanley node and the independent plot node.
    """

    def __init__(self, path_cfg):
        if not isinstance(path_cfg, dict):
            raise ValueError("Config must contain a dict section named `path`.")

        self.path_cfg = path_cfg
        self.ID_last = 0
        self.lateral_err = 0.0
        self.ref_path = self.generate_ref_path()

    def generate_ref_path(self):
        path_type = str(
            self.path_cfg.get("type", "straight_circle")
        ).strip().lower()

        if path_type in ("straight_circle", "circle"):
            ref_path = self.GenerateStraightCircleRef()
        elif path_type in ("sine", "sin"):
            ref_path = self.GenerateSineRef()
        elif path_type in ("double_lane_change", "double_lane", "dlc"):
            ref_path = self.GenerateDoubleLaneChangeRef()
        else:
            raise ValueError(
                f"Unsupported path.type: {path_type}. "
                "Use straight_circle, sine, or double_lane_change."
            )

        return self.NormalizeRefPathStart(ref_path)
