import numpy as np
from src.others.frame import Frame
from src.local_mapping.local_map import Map
from config import SETTINGS


debug = SETTINGS["generic"]["debug"]


def is_keyframe(t_frame: Frame, keyframes: set[Frame], local_map: Map):
    """
    New Keyframe conditions:
        1) More than 20 frames must have passed from the last global relocalization.
        2) Local mapping is idle, or more than 20 frames have passed from last keyframe insertion.
        3) Current frame tracks at least 50 points.
        4) Current frame tracks less than 90% points than Kref .
    """
    last_reloc_count = count_since_last_relocalization(keyframes)
    cond1 = last_reloc_count > 20

    # Condition 2 is only True cause we are not using threads, so local mapping is always IDLE
    num_frames_passed = t_frame.id - keyframes[-1].id
    local_mapping_idle = True
    cond2 = local_mapping_idle or num_frames_passed > 20

    cond3 = t_frame.num_tracked_points > 50

    ref_frame = keyframes[local_map.ref_frame_id]
    A = ref_frame.tracked_points
    B = t_frame.tracked_points
    common_features_ratio = len(A.intersection(B)) / len(A)
    cond4 = common_features_ratio < 0.9
    
    if cond1 and cond2 and cond3 and cond4:
        print("\t\t Keyframe!")
    else:
        print("\t\t Not a keyframe!")

    return is_keyframe

def count_since_last_relocalization(frames: list[Frame]):
    """
    Counts how many frames have passed since the last global relocalization.
    If no global relocalization has taken place, returns a really big number.
    """
    count = 0
    # Iterate from the end toward the beginning
    for f in reversed(frames):
        if f.relocalization == True:
            # Stop when we find the last object with relocalization True
            return count
        count += 1

    if count == len(frames):
        count = 99999

    return count