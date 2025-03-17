from typing import List 
import numpy as np
from src.frame import Frame
from src.utils import invert_transform
from config import image_width, image_height, debug


class Map():
    def __init__(self, frame_id: int):
        self.origin_frame = frame_id  # ID of the frame when the map was first created
        self.entries: List = []       # Entries for each map push
        """
        Each entry looks like this:
        entry {
            "points": np.ndarray,                       # Map 3D points in world coordinates
            "point_ids": np.ndarray,                    # Map 3D point unique IDs

            "q_keyframe": Frame,                        # Reference keyframe that was used to triangulate the 3d points
            "t_keyframe": Frame,                        # Keyframe that was used to triangulate the 3d points

            "type": String [initialization, tracking]   # Whether the points were computed in the initialization (2d-2d) or the tracking stage (3d-2d)
        }
        """

    @property
    def points(self):
        return np.vstack([entry["points"] for entry in self.entries])
    @property
    def point_ids(self):
        return np.hstack([entry["point_ids"] for entry in self.entries])

    def add_initialization_points(self, points: np.ndarray, point_ids: np.ndarray, q_frame: Frame, t_frame: Frame):
        entry = {
            "points": points,
            "point_ids": point_ids,
            "q_keyframe": q_frame,
            "t_keyframe": t_frame,
            "type": "initialization"
        }
        self.entries.append(entry)
        print(f"[Map] Adding {len(points)} initialization points from frame {t_frame.id}. Total: {len(self.points)} points.")

    def add_tracking_points(self, points: np.ndarray, point_ids: np.ndarray, q_frame: Frame, t_frame: Frame):
        print(f"[Map] Adding {len(points)} tracking points from frame {t_frame.id}")
        entry = {
            "points": points,
            "point_ids": point_ids,
            "q_keyframe": q_frame,
            "t_keyframe": t_frame,
            "type": "tracking"
        }
        self.entries.append(entry)

    def get_points_in_view(
        self,
        T_cw: np.ndarray,
        K: np.ndarray
    ):
        """
        Returns the points and descriptors that are in the current view.

        Args:
            T_cw: The transform from the current camera coordinate frame to the world frame

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
        print("[Map] Getting points in the current view...")

        # 0) Collect ALL points and descriptors from the map
        #    We'll do this by concatenating the data from all entries
        map_points = []
        map_desc = []
        for entry in self.entries:
            # Extract the keyframe, its reference keyframe, and the match
            q_keyframe = entry["q_keyframe"]
            t_keyframe = entry["t_keyframe"]
            match = q_keyframe.match[t_keyframe.id]

            # Extract the points
            match_points = match["points"]         # shape: (M, 3)
            map_points.append(match_points)

            # Extract all the keyframe descriptors
            t_descriptors = t_keyframe.descriptors # shape: (N, 32)

            # Extract the keyframe descriptors that correspond to the match
            t_indices = np.array([m.trainIdx for m in match["matches"]])        # shape: (M,)
            t_descriptors = np.array([t_descriptors[idx] for idx in t_indices]) # shape: (M, 32)

            # Extract the triangulation mask of the match
            triangulation_match_mask = match["triangulation_match_mask"] # shape: (M, )

            # Only keep the descriptors that correspond to the triangulated points
            match_triang_descriptors = t_descriptors[triangulation_match_mask]  # shape: (T, 32)
            map_desc.append(match_triang_descriptors)

            assert(len(match_triang_descriptors) == len(match["points"]))

        # Concatenate all points and descriptors
        points_w = np.vstack(map_points) # shape: (L, 3)
        map_desc = np.vstack(map_desc)   # shape: (L, 32)

        # No map points at all
        if len(points_w) == 0:
            return None, None, None

        # 1) Convert map points to homogeneous: (X, Y, Z, 1).
        ones = np.ones((len(points_w), 1))
        points_w_hom = np.hstack([points_w, ones])  # (L, 4)

        # 2) Transform points to camera coords:
        #       X_cam = T_cw @ X_world
        T_wc = invert_transform(T_cw)
        points_c_hom = (T_wc @ points_w_hom.T).T # (L, 4)
        points_c = points_c_hom[:, :3]           # (L, 3)

        # 3) Keep only points in front of the camera (z > 0).
        z_positive_mask = points_c[:, 2] > 0
        points_w = points_w[z_positive_mask]
        points_c = points_c[z_positive_mask]
        map_desc = map_desc[z_positive_mask]

        # 4) Project into pixel coordinates using K.
        x_cam = points_c[:, 0]
        y_cam = points_c[:, 1]
        z_cam = points_c[:, 2]

        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        u = fx * x_cam / z_cam + cx
        v = fy * y_cam / z_cam + cy

        # 5) Check if the projected points lie within image boundaries.
        in_view_mask = (
            (u >= 0) & (u < image_width) &
            (v >= 0) & (v < image_height)
        )

        # Keep only the points that are in view
        points_w_in_view = points_w[in_view_mask]
        points_c_in_view = points_c[in_view_mask]
        u_in_view = u[in_view_mask]
        v_in_view = v[in_view_mask]

        # Gather the in view pixels and their descriptors
        pxs_c_n_view = np.column_stack([u_in_view, v_in_view])
        descriptors_in_view = map_desc[in_view_mask]

        if debug:
            print(f"\tFound {len(points_w_in_view)} map points in the predicted camera pose view.")

        return points_w_in_view, pxs_c_n_view, descriptors_in_view
    
    def cleanup(
        self,
        T_wc: np.ndarray,
        K: np.ndarray
    ):
        """
        Args:
            T_wc: Transformation from world to camera coordinate frame
            K: Intrinsics matrix

        Remove map points that are 
            (1) not in current view
            (2) with a view angle larger than the threshold
            (3) rarely matched as inlier point
        """
        print("[Map] Cleaning up map points")

        # 0) Iterate over every entry in the map
        num_removed_points = 0
        for entry in self.entries:
            points = entry["points"]  # shape: (N, 3)

            # 1) Convert map points to homogeneous: (X, Y, Z, 1).
            N = points.shape[0]
            ones = np.ones((N, 1))
            map_points = np.hstack([points, ones])  # (N, 4)

            # 2) Transform points to camera coords:
            #       X_cam = T_cw @ X_world
            points_c_hom = (T_wc @ map_points.T).T  # (N, 4)
            points_c = points_c_hom[:, :3]

            # 3) Keep only points in front of the camera (z > 0).
            z_positive_mask = points_c[:, 2] > 0
            points_c = points_c[z_positive_mask]

            # 4) Project into pixel coordinates using K.
            x_cam = points_c[:, 0]
            y_cam = points_c[:, 1]
            z_cam = points_c[:, 2]

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
            num_removed_points += np.count_nonzero(~in_view_mask)

            entry["points"] = points_c[in_view_mask]
            entry["point_ids"] = entry["point_ids"][z_positive_mask][in_view_mask]

        print(f"[Map] Removed {num_removed_points} points from the map!")
