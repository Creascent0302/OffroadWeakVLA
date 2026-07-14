"""Global constants shared by every module.

Gear semantics (fixed across the whole project):
    0 = S : small contact area  -> efficient, agile, low traction
    1 = M : medium contact area
    2 = L : large contact area  -> costly, high traction (the safe fallback)

Language semantics (must stay consistent in language.py, conformal/select.py
and the tests):
    "careful / slow / fragile" -> conservative -> LARGER gear
    "fast / efficient / save energy" -> efficient -> SMALLER gear
"""

# ============================================================================
# 【中文导读】全局常量。整个项目共用这一份定义，改这里就等于改全局。
#   档位 GEARS = [0,1,2] = [S 小接地面积, M 中, L 大接地面积]
#   接地面积 CONTACT_AREA 同时充当“单位距离能耗代理”：越大越抓地、也越费力
#   兜底档 FALLBACK_GEAR = 2(L)：任何“无法认证安全”的情况都退到最大牵引
#   语义方向（三处必须一致：language.py / conformal/select.py / tests）：
#     “小心/慢/易碎” → 保守 → 偏大档;  “快/省电/高效” → 高效 → 偏小档
# ============================================================================


from __future__ import annotations

GEARS: list[int] = [0, 1, 2]
NUM_GEARS: int = len(GEARS)

GEAR_NAMES: dict[int, str] = {0: "S", 1: "M", 2: "L"}

# Contact area, also used as the per-step energy/effort proxy: larger = costlier.
CONTACT_AREA: dict[int, float] = {0: 0.0, 1: 0.5, 2: 1.0}

# The gear the system falls back to whenever it cannot certify any gear.
FALLBACK_GEAR: int = 2

SLIP_MIN: float = 0.0
SLIP_MAX: float = 1.0

# Image geometry. DINOv2 ViT/14 needs a side length that is a multiple of 14.
IMAGE_SIZE: int = 224
PATCH_SIZE: int = 14

# ImageNet statistics used by the frozen DINOv2 backbone.
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

SPLITS: tuple[str, ...] = ("train", "cal", "test", "ood")


def gear_name(gear: int) -> str:
    return GEAR_NAMES[int(gear)]


def contact_area(gear: int) -> float:
    return CONTACT_AREA[int(gear)]
