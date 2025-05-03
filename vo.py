import time
import numpy as np

import src.utils as utils
import src.visualization as vis
import src.initialization as init
import src.tracking as track
import src.place_recognition as pr
import src.local_mapping as mapping
import src.backend as backend
import src.globals as ctx

from config import main_dir, data_dir, results_dir, SETTINGS, log


"""
Important notes:
- Both the Kitti and OpenCV camera frames follow the same convention:
    - x: right
    - y: down
    - z: forward
"""


CLEANUP = SETTINGS["generic"]["cleanup"]
debug = SETTINGS["generic"]["debug"]
parallel = SETTINGS["generic"]["parallel"]
scene = SETTINGS["generic"]["scene"]
log.info(f"\t\tUsing dataset: `{scene}` ...")


def initialize(i: int, q_frame: utils.Frame, t_frame: utils.Frame):
    log.info("")
    log.info("~~~~Initialization~~~~")
    t0 = time.perf_counter()

    # Feature matching
    matches = init.matchFeatures(q_frame, t_frame) # (N) : N < M
    if matches is None:
        log.info("Feature matching failed!")
        return False

    # Extract the initial pose using the Essential or Homography matrix (2d-2d)
    matches, T_q2t, is_initialized = init.estimate_pose(matches, q_frame, t_frame)
    if not is_initialized:
        log.info("Pose initialization failed!")
        return False
    T_t2q = utils.invert_transform(T_q2t)

    # Calculate the next pose with scale ambiguity
    T_q2w = q_frame.pose
    T_t2w_unscaled = T_q2w @ T_t2q

    # Estimate the depth scale
    scale = utils.estimate_depth_scale([T_q2w, T_t2w_unscaled], [q_frame.gt, t_frame.gt])

    # Remove scale ambiguity
    T_t2q[:3, 3] *= scale

    # Set the pose in the current frame
    T_t2w = T_q2w @ T_t2q
    t_frame.set_pose(T_t2w)

    # Triangulate the 3D points using the initial pose
    triang_result = init.triangulate_points(matches, T_q2t, q_frame, t_frame, scale)
    if triang_result is None:
        is_initialized = False
        log.info("Triangulation failed!")
        return False
    w_points, matches = triang_result

    # Push the keyframes and triangulated points to the map
    ctx.map.add_keyframe(t_frame)
    ctx.cgraph.add_keyframe(t_frame.id)
    ctx.map.add_new_points(w_points, matches, q_frame, t_frame)
    ctx.cgraph.update_edges()

    # Validate the scale
    utils.validate_scale([q_frame.pose, t_frame.pose], [q_frame.gt, t_frame.gt])

    # Perform Bundle Adjustment
    ba = backend.poseBA()
    # ba = backend.globalBA()
    ba.optimize()

    # Plots
    # plot_BA()
    if debug:
        vis.plot_BA2d(results_dir / "ba/global" / f"{i}.png")
        utils.save_image(t_frame.img, results_dir / "keyframes" / f"{i}_bw.png")
    vis.plot_trajectory(results_dir / "trajectory/" / f"{i}_a.png")

    q_frame = t_frame
    log.info(f"\t # of points: {ctx.map.num_points}")
    log.info(f"\t # of keyframes: {ctx.map.num_keyframes}")
    log.info(f"\t\t\t ... took {time.perf_counter() - t0:.2f} seconds!")

    return True


def tracking(i: int, t_frame: utils.Frame):
    log.info("")
    log.info("~~~~Tracking~~~~")
    t1 = time.perf_counter()

    num_matches = track.map_search(t_frame)
    if num_matches < 20:
        log.error(f"Tracking failed! {num_matches} (<20) matches found!")
        return False

    # Estimate the new pose using PnP
    pnp_success = track.estimate_relative_pose(t_frame)
    if not pnp_success:
        return False
    
    # Add the new frame to the map and convisibility graph, along with the new edges
    ctx.map.add_keyframe(t_frame)
    t_point_ids = t_frame.get_map_point_ids()
    ctx.cgraph.add_keyframe_with_points(t_frame.id, t_point_ids)
    ctx.cgraph.update_edges()

    # Perform pose optimization - ORB uses singlePoseBA, but poseBA gives better results
    # ba = backend.singlePoseBA(t_frame)
    ba = backend.poseBA()
    ba.optimize()

    # Plot the BA
    # plot_BA()
    if debug:
        vis.plot_BA2d(results_dir / "ba/single_pose" / f"{i}.png")
    vis.plot_trajectory(results_dir / "trajectory/" / f"{i}_b.png")
    
    # ########### Track Local Map ###########
    # Set the visible mask
    ctx.map.view(t_frame)

    # Extract a local map from the map
    ctx.local_map = ctx.map.create_local_map(t_frame)
    
    # Set the found mask
    ctx.map.tracked(t_frame)
    
    # ########### New Keyframe Decision ###########
    # Check if this t_frame is a keyframe
    if not t_frame.is_keyframe():
        ctx.map.remove_keyframe(t_frame.id)
        ctx.cgraph.remove_keyframe(t_frame.id)
        return False

    # Save the keyframe
    if debug:
        utils.save_image(t_frame.img, results_dir / "keyframes" / f"{i}_bw.png")

    log.info(f"\t\t\t ... took {time.perf_counter() - t1:.2f} seconds!")

    return True


def local_mapping(i: int, t_frame: utils.Frame):
    # ########### Map Point/Keyframe Creating/Culling ###########
    log.info("")
    log.info("~~~~Updating Map~~~~")
    t2 = time.perf_counter()

    # Clean up map points that are not seen anymore
    ctx.map.cull_points()
    ctx.cgraph.update_edges()

    # Create new map points
    if parallel:
        ctx.map.create_points_parallel(t_frame)
    else:
        ctx.map.create_points(t_frame)
    ctx.cgraph.update_edges()

    # Perform Local Bundle Adjustment
    # ba = backend.globalBA()
    ba = backend.localBA(t_frame)
    ba.optimize()
    ctx.cgraph.update_edges()

    # Plot the BA
    # plot_BA()
    if debug:
        vis.plot_BA2d(results_dir / "ba/local" / f"{i}.png")
    vis.plot_trajectory(results_dir / "trajectory/" / f"{i}_c.png")

    # Clean up redundant frames
    ctx.map.cull_keyframes(t_frame)
    ctx.cgraph.update_edges()
    log.info(f"\t # of points: {ctx.map.num_points}")
    log.info(f"\t # of keyframes: {ctx.map.num_keyframes}")
    log.info(f"\t\t\t ... took {time.perf_counter() - t2:.2f} seconds!")


def loop_closure(i: int, t_frame: utils.Frame):
    # ########### Loop Closing ###########
    if (i > 10) and (ctx.map.num_keyframes_since_last_loop > 10):
        log.info("")
        log.info("~~~~Loop Closing~~~~")
        t3 = time.perf_counter()

        # Find candidates for loop closure
        candidate_kfs = pr.detect_candidates(t_frame)
        if candidate_kfs is not None:
            # Iterate over all possible candidates
            if debug:
                utils.save_image(t_frame.img, results_dir / f"loop/{t_frame.id}/{t_frame.id}.png")
            for cand_kf in candidate_kfs:
                if debug:
                    utils.save_image(cand_kf.img, results_dir / f"loop/{t_frame.id}/frames/{cand_kf.id}.png")

                # Search for matches with current frame
                num_matches = pr.frame_search(cand_kf, t_frame)
                ctx.cgraph.update_edges()
                if num_matches < 20: continue
                
                # Estimate the new world pose using PnP (3d-2d)
                T_q2t = pr.estimate_relative_pose(cand_kf, t_frame)
                if T_q2t is None: continue
                log.info(f"Found loop closure between {t_frame.id} and {cand_kf.id} with {num_matches} matches!")

                # Add loop edge to the convisibility graph
                ctx.map.add_loop_closure(cand_kf, t_frame)
                ctx.cgraph.add_loop_edge(cand_kf.id, t_frame.id, num_matches)

                # Perform pose optimization
                ba = backend.poseBA()
                ba.add_loop_edge(cand_kf, t_frame, T_q2t)
                ba.optimize()

                # Plot the BA
                # plot_BA()
                if debug:
                    vis.plot_BA2d(results_dir / "ba/pose" / f"{i}.png")
                vis.plot_trajectory(results_dir / "trajectory/c" / f"{i}.png")

                break

        log.info(f"\t\t\t ... took {time.perf_counter() - t3:.2f} seconds!")


def main():
    # Clean previous results
    if CLEANUP:
        utils.delete_subdirectories(results_dir)

    # Read the data
    data = utils.Dataset(data_dir, scene)

    # Plot the ground truth trajectory
    gt = data.ground_truth()
    vis.plot_ground_truth(gt)

    ## Setup Globals
    # Read the vocabulary and initialize the BoW database
    ctx.vocab = pr.load_vocabulary("cv2").astype(np.uint8) # Basically contains 1000 descriptors
    ctx.bow_db = {} # contains visual_word_id -> keyframe_that_sees_it dicts
    for i in range(len(ctx.vocab)): ctx.bow_db[i] = set()

    # Initialize map and convisibility graph
    ctx.map = mapping.Map()
    ctx.cgraph = backend.ConvisibilityGraph()

    # Run the main VO loop
    i = -1
    is_initialized = False
    while not data.finished():
        # Advance the iteration
        i+=1
        log.info("")
        log.info("")
        log.info(f"\t Iteration: {i} / {data.length()}")

        # Capture new image frame (current_frame)
        t, img, gt_pose = data.get()

        # Create a frame and extract its ORB features
        t_frame = utils.Frame(i, img)
        t_frame.set_gt(gt_pose)
        t_frame.set_time(t)
        t_frame.compute_bow()

        # Iteration #0
        if t_frame.id == 0:
            log.info("")
            log.info("~~~~First Frame~~~~")
            assert np.all(np.eye(4) - gt_pose < 1e-6)

            # Bookkeping
            t_frame.set_pose(gt_pose)
            ctx.map.add_keyframe(t_frame)
            ctx.cgraph.add_keyframe(t_frame.id)

            if debug:
                utils.save_image(t_frame.img, results_dir / "keyframes" / f"{i}_bw.png")
        else:                    
            # ########### Initialization ###########
            if not is_initialized:
                is_initialized = initialize(i, q_frame, t_frame)
                if not is_initialized:
                    continue
            # ########### Tracking ###########
            else:
                success = tracking(i, t_frame)
                if not success: continue
                local_mapping(i, t_frame)
                loop_closure(i, t_frame)
                
        # Advance to the next iteration
        q_frame = t_frame

    # Perform one final optimization
    ba = backend.globalBA()
    ba.finalize()

    # Save final map and trajectory
    vis.plot_trajectory_3d()

if __name__ == "__main__":
    main()