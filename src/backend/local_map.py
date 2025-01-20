from typing import List 
import numpy as np
from src.frame import Frame


class Map():
    def __init__(self, frame_id: int):
        self.origin_frame = frame_id  # ID of the frame when the map was first created
        self.entries: List = []       # Entries for each map push
        """
        Each entry looks like this:
        entry {
            "points": np.ndarray,                       # Map 3D points in world coordinates
            "point_ids": np.ndarray,                    # Map 3D point unique IDs
            "keyframe": Frame,                          # Keyframe that was used to triangulate the 3d points
            "ref_keyframe": Frame,                      # Reference keyframe that was used to triangulate the 3d points
            "type": String [initialization, tracking]   # Whether the points were computed in the initialization (2d-2d) or the tracking stage (3d-2d)
        }
        """

    @property
    def points(self):
        return np.vstack([entry["points"] for entry in self.entries])
    @property
    def point_ids(self):
        return np.hstack([entry["point_ids"] for entry in self.entries])

    def add_initialization_points(self, points: np.ndarray, point_ids: np.ndarray, frame: Frame, ref_frame: Frame):
        print(f"[Map] Adding {len(points)} initialization points from frame {frame.id}")
        entry = {
            "points": points,
            "point_ids": point_ids,
            "keyframe": frame,
            "ref_keyframe": ref_frame,
            "type": "initialization"
        }
        self.entries.append(entry)

    def add_tracking_points(self, points: np.ndarray, point_ids: np.ndarray, frame: Frame, ref_frame: Frame):
        print(f"[Map] Adding {len(points)} tracking points from frame {frame.id}")
        entry = {
            "points": points,
            "point_ids": point_ids,
            "keyframe": frame,
            "ref_keyframe": ref_frame,
            "type": "tracking"
        }
        self.entries.append(entry)

    def cleanup(
        self,
        T_cam_world: np.ndarray,
        K: np.ndarray,
        image_width: int = 1226,
        image_height: int = 370
    ):
        """
        Remove map points that are 
            (1) not in current view
            (2) with a view angle larger than the threshold
            (3) rarely matched as inlier point
        """
        print("[Map] Cleaning up map points")

        # 0) Iterate over every entry in the map
        for entry in self.entries:
            points = entry["points"]  # shape: (N, 3)

            # 1) Convert map points to homogeneous: (X, Y, Z, 1).
            N = points.shape[0]
            ones = np.ones((N, 1))
            points_world_hom = np.hstack([points, ones])  # (N, 4)

            # 2) Transform points to camera coords:
            #       X_cam = T_cam_world @ X_world
            points_cam_hom = (T_cam_world @ points_world_hom.T).T  # (N, 4)
            points_cam = points_cam_hom[:, :3]

            # 3) Keep only points in front of the camera (z > 0).
            z_positive_mask = points_cam[:, 2] > 0
            points_cam = points_cam[z_positive_mask]

            # 4) Project into pixel coordinates using K.
            x_cam = points_cam[:, 0]
            y_cam = points_cam[:, 1]
            z_cam = points_cam[:, 2]

            fx = K[0, 0]
            fy = K[1, 1]
            cx = K[0, 2]
            cy = K[1, 2]

            u = fx * (x_cam / z_cam) + cx
            v = fy * (y_cam / z_cam) + cy

            # 5) Check if the projected points lie within image boundaries.
            in_view_mask = (
                (u >= 0) & (u < image_width) &
                (v >= 0) & (v < image_height)
            )

            entry["points"] = points_cam[in_view_mask]
            entry["point_ids"] = entry["point_ids"][z_positive_mask][in_view_mask]

    def get_points_in_view(
        self,
        T_cam_world: np.ndarray,
        K: np.ndarray,
        image_width: int = 1226,
        image_height: int = 370
    ):
        print("[Map] Getting points in the current view...")
        """
        Returns the points and descriptors that are in the current view.

        Returns:
            (points_in_view_3d, points_in_view_2d, descriptors_in_view)
            where:
                points_in_view_3d: (M, 3) array of 3D points in *camera* coords
                                   that lie in front of the camera and within
                                   the image boundaries.
                points_in_view_2d: (M, 2) array of their (u, v) pixel coords.
                descriptors_in_view: (M, 32) array of the associated descriptors
                                     from the keyframe(s) that created them.
        """

        # 0) Collect ALL points and descriptors from the map
        #    We'll do this by concatenating the data from all entries
        all_points_list = []
        all_desc_list = []
        for entry in self.entries:
            # Extract the keyframe, its reference keyframe, and the match
            keyframe = entry["keyframe"]
            ref_keyframe = entry["ref_keyframe"]
            match = keyframe.match[ref_keyframe.id]

            # Extract the points
            all_points_list.append(match["points"])          # shape: (N, 3)

            # Extract the keyframe descriptors of the match
            descriptors = keyframe.descriptors                                        # shape: (M, 32)
            match_indices = np.array([m.queryIdx for m in match["matches"]])          # shape: (N,)
            match_descriptors = np.array([descriptors[idx] for idx in match_indices]) # shape: (N, 32)

            # Extract the inlier mask
            triangulation_mask = match["triangulation_match_mask"] # shape: (N, 3)

            # Only keep the descriptors that correspond to the triangulated points
            all_desc_list.append(match_descriptors[triangulation_mask])  # shape: (N, 32)

            assert(len(match_descriptors[triangulation_mask]) == len(match["points"]))

        if len(all_points_list) == 0:
            # No map points at all
            return (np.zeros((0,3)), np.zeros((0,2)), np.zeros((0,32)))

        all_points = np.vstack(all_points_list)        # shape: (sumN, 3)
        all_descriptors = np.vstack(all_desc_list)     # shape: (sumN, 32)

        # 1) Convert map points to homogeneous: (X, Y, Z, 1).
        N = all_points.shape[0]
        ones = np.ones((N, 1))
        points_world_hom = np.hstack([all_points, ones])  # (N, 4)

        # 2) Transform points to camera coords:
        #       X_cam = T_cam_world @ X_world
        points_cam_hom = (T_cam_world @ points_world_hom.T).T  # (N, 4)
        points_cam = points_cam_hom[:, :3]

        # 3) Keep only points in front of the camera (z > 0).
        z_positive_mask = points_cam[:, 2] > 0
        points_cam = points_cam[z_positive_mask]
        all_descriptors = all_descriptors[z_positive_mask]

        # 4) Project into pixel coordinates using K.
        x_cam = points_cam[:, 0]
        y_cam = points_cam[:, 1]
        z_cam = points_cam[:, 2]

        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        u = fx * (x_cam / z_cam) + cx
        v = fy * (y_cam / z_cam) + cy

        # 5) Check if the projected points lie within image boundaries.
        in_view_mask = (
            (u >= 0) & (u < image_width) &
            (v >= 0) & (v < image_height)
        )

        points_in_view_3d = points_cam[in_view_mask]
        u_in_view = u[in_view_mask]
        v_in_view = v[in_view_mask]
        points_in_view_2d = np.column_stack([u_in_view, v_in_view])
        descriptors_in_view = all_descriptors[in_view_mask]

        return points_in_view_3d, points_in_view_2d, descriptors_in_view
    

