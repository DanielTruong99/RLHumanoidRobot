"""Package containing asset and sensor configurations."""

import os
import toml

# Conveniences to other module directories via relative paths
ISAACLAB_ASSETS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
"""Path to the extension source directory."""

ISAACLAB_ASSETS_DATA_DIR = os.path.join(ISAACLAB_ASSETS_EXT_DIR, "data")
"""Path to the extension data directory."""


LOCAL_ASSETS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
LOCAL_ASSETS_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

##
# Configuration for different assets.
##

from .leg_robot import *
from .leg_parkour import *
# from .allegro import *
# from .ant import *
# from .anymal import *
# from .cartpole import *
# from .franka import *
# from .humanoid import *
# from .kinova import *
# from .quadcopter import *
# from .ridgeback_franka import *
# from .sawyer import *
# from .shadow_hand import *
# from .unitree import *
# from .universal_robots import *