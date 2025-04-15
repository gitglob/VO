import numpy as np
from src.others.utils import get_yaw
from config import SETTINGS


debug = SETTINGS["generic"]["debug"]


def is_keyframe(T: np.ndarray, num_tracked_points: int = 9999):
    """ Determine if motion expressed by t, R is significant by comparing to tresholds. """
    tx = T[0, 3] # The x component points right
    ty = T[1, 3] # The y component points down
    tz = T[2, 3] # The z component points forward

    trans = np.sqrt(tx**2 + ty**2 + tz**2)
    angle = abs(get_yaw(T[:3, :3]))

    is_keyframe = num_tracked_points > SETTINGS["keyframe"]["num_tracked_points"]
    # is_keyframe = is_keyframe and (trans > SETTINGS["keyframe"]["distance"] or angle > SETTINGS["keyframe"]["angle"])
    
    if debug:
        print(f"\t Tracked points: {num_tracked_points}, dist: {trans:.3f}, angle: {angle:.3f}")
        if is_keyframe:
            print("\t\t Keyframe!")
        else:
            print("\t\t Not a keyframe!")

    return is_keyframe