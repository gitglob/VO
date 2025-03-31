import numpy as np

from src.others.frame import Frame
from src.others.local_map import Map



class convisibilityGraph:
    """
    Class to represent a convisibility graph for the vSLAM system.

    Each node is a keyframe and an edge between two keyframes exists 
    if they share observations of the same map points (at least 15).
    """
    def __init__(self):
        self.nodes: dict[tuple[Frame, Frame, np.ndarray]] = {}
        self.n = 0

    def add_node(self, keyframe: Frame, map: Map):
        # Iterate over the keyframe keypoints that are in the current map
        kpts_in_map = keyframe.keypoints_in_map(map)

        self.nodes[keyframe.id] = []
        # Iterate over all keyframes
        for node_id, v in self.nodes[:-1]:
            # Extract the convisibility info
            (neigh_keyframe, neigh_keyframe, neigh_common_pts) = v
            # Extract the map points from the keyframe
            other_kpts_in_map = neigh_keyframe.keypoints_in_map(map)

            # Get the number of common keypoints
            common_kpts_mask = np.isin(kpts_in_map, other_kpts_in_map)
            common_kpt_ids = kpts_in_map[common_kpts_mask]

            # Check if there are enough common keypoints
            num_common_kpts = common_kpts_mask.sum()
            if num_common_kpts >= 15:
                # Add an edge between the two keyframes
                self.add_edge(keyframe.id, neigh_keyframe.id)

                # Add a new keyframe to the graph
                self.nodes[keyframe.id].append((keyframe, neigh_keyframe, common_kpt_ids))

        self.n += 1
