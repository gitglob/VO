import src.utils as utils
import src.local_mapping as mapping
import src.globals as ctx
from config import SETTINGS, log


debug = SETTINGS["generic"]["debug"]


def is_keyframe(t_frame: utils.Frame):
    """
    New Keyframe conditions:
        1) More than X frames must have passed from the last global relocalization.
        2) Local mapping is idle, or more than X frames have passed from last keyframe insertion.
        3) Current frame tracks at least 50 points.
        4) Current frame tracks less than 90% points than Kref .
    """
    keyframes = ctx.map.keyframes
    local_map = ctx.local_map

    other_frames = list(keyframes.values())[:-1]

    last_reloc_kf_id = ctx.map.last_reloc
    num_frames_since_last_reloc = t_frame.id - last_reloc_kf_id
    c1 = num_frames_since_last_reloc > 2

    # Condition 2 is only True cause we are not using threads, so local mapping is always IDLE
    num_frames_passed = t_frame.id - other_frames[-1].id
    local_mapping_idle = True
    c2 = local_mapping_idle or num_frames_passed > 2

    c3 = t_frame.num_tracked_points > 50

    ref_frame = keyframes[local_map.ref]
    A = ref_frame.tracked_points
    B = t_frame.tracked_points
    common_features_ratio = len(A.intersection(B)) / len(A)
    c4 = common_features_ratio < 0.9
    
    is_keyframe = c1 and c2 and c3 and c4
    if is_keyframe:
        log.info("\t\t Keyframe!")
    else:
        log.warning("\t\t Not a keyframe!")
        if not c1:
            log.warning(f"\t\t # of frames since last relocalization: {num_frames_since_last_reloc} <= 2!")
        if not c2:
            log.warning(f"\t\t # of frames since last keyframe: {num_frames_passed} <= 2!")
        if not c3:
            log.warning(f"\t\t # of tracked points: {t_frame.num_tracked_points} <= 50!")
        if not c4:
            log.warning(f"\t\t Common features ratio: {common_features_ratio} > 0.9!")
        pass

    return is_keyframe
