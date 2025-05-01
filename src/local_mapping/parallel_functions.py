import numpy as np

import src.globals as ctx
import src.utils as utils
from .point_association import search_for_triangulation
from config import SETTINGS


# 1) Persistent pool: one per Python process
MIN_PARALLAX = SETTINGS["map"]["min_parallax"]
MAX_REPROJECTION = SETTINGS["map"]["max_reprojection"]

# 2) Per-neighbor worker
def process_neighbor(args) -> tuple[list, dict]:
    q_id, t_id, ratio_factor = args
    q_frame = ctx.map.keyframes[q_id]
    t_frame = ctx.map.keyframes[t_id]

    local_results = []   # will hold tuples (world_pt, match, q_frame)
    counters = {
        'epipolar': 0, 'cheirality': 0, 'reprojection': 0,
        'parallax': 0, 'distance': 0, 'scale': 0, 'depth': 0
    }

    # a) Baseline / depth check
    baseline = np.linalg.norm(t_frame.pose[:3,3] - q_frame.pose[:3,3])
    median_depth = q_frame.median_depth(ctx.map)
    if baseline / median_depth < 0.01:
        return local_results, counters

    # b) Descriptor‐based matches
    matches = search_for_triangulation(q_frame, t_frame)
    if len(matches) < 5:
        return local_results, counters

    # c) Get keypoints arrays
    q_kpts = np.array([q_frame.keypoints[m.queryIdx] for m in matches])
    t_kpts = np.array([t_frame.keypoints[m.trainIdx] for m in matches])

    # d) Epipolar filter
    q_pix = np.float64([kp.pt for kp in q_kpts])
    t_pix = np.float64([kp.pt for kp in t_kpts])
    ret = utils.enforce_epipolar_constraint(q_pix, t_pix)
    if ret is None:
        return local_results, counters
    epi_mask, _, _ = ret
    counters['epipolar'] += np.sum(~epi_mask)
    matches, q_kpts, t_kpts = (
        np.array(matches)[epi_mask],
        q_kpts[epi_mask],
        t_kpts[epi_mask]
    )
    if len(matches) == 0:
        return local_results, counters

    # e) Prepare transforms
    T_q2t = utils.invert_transform(t_frame.pose) @ q_frame.pose

    # f) Batch‐triangulate & transform
    q_pts = utils.triangulate(
        np.float64([kp.pt for kp in q_kpts]),
        np.float64([kp.pt for kp in t_kpts]),
        T_q2t
    )
    t_pts = utils.transform_points(q_pts, T_q2t)
    if q_pts is None or len(q_pts) == 0:
        return local_results, counters

    # g) Cheirality
    che_mask = utils.filter_cheirality(q_pts, t_pts)
    counters['cheirality'] += np.sum(~che_mask)
    matches, q_pts, t_pts, q_kpts, t_kpts = (
        matches[che_mask], q_pts[che_mask], t_pts[che_mask],
        q_kpts[che_mask], t_kpts[che_mask]
    )
    if len(matches) == 0:
        return local_results, counters

    # h) Parallax
    par_mask = utils.filter_parallax(q_pts, t_pts, T_q2t, MIN_PARALLAX)
    counters['parallax'] += np.sum(~par_mask)
    matches, q_pts, t_pts, q_kpts, t_kpts = (
        matches[par_mask], q_pts[par_mask], t_pts[par_mask],
        q_kpts[par_mask], t_kpts[par_mask]
    )
    if len(matches) == 0:
        return local_results, counters

    # i) Reprojection
    reproj_mask, _ = utils.filter_by_reprojection(
        q_pts,
        np.float64([kp.pt for kp in t_kpts]),
        T_q2t,
        MAX_REPROJECTION
    )
    counters['reprojection'] += np.sum(~reproj_mask)
    matches, q_pts, t_pts, q_kpts, t_kpts = (
        matches[reproj_mask], q_pts[reproj_mask],
        t_pts[reproj_mask], q_kpts[reproj_mask],
        t_kpts[reproj_mask]
    )
    if len(matches) == 0:
        return local_results, counters

    # j) Distance & scale consistency
    q_dists = np.linalg.norm(q_pts - q_frame.pose[:3,3], axis=1)
    t_dists = np.linalg.norm(t_pts - t_frame.pose[:3,3], axis=1)
    dist_mask = (q_dists > 0) & (t_dists > 0)
    counters['distance'] += np.sum(~dist_mask)
    matches, q_pts, t_pts, q_kpts, t_kpts = (
        matches[dist_mask], q_pts[dist_mask], t_pts[dist_mask],
        q_kpts[dist_mask], t_kpts[dist_mask]
    )
    if len(matches) == 0:
        return local_results, counters

    q_scales = np.array([q_frame.scale_factors[k.octave] for k in q_kpts])
    t_scales = np.array([t_frame.scale_factors[k.octave] for k in t_kpts])
    ratio_d = t_dists[dist_mask] / q_dists[dist_mask]
    ratio_o = t_scales / q_scales
    scale_mask = (ratio_o / ratio_factor < ratio_d) & (ratio_d < ratio_o * ratio_factor)
    counters['scale'] += np.sum(~scale_mask)
    matches, q_pts, t_pts = matches[scale_mask], q_pts[scale_mask], t_pts[scale_mask]
    if len(matches) == 0:
        return local_results, counters

    # k) Depth filter
    q_med = np.median(q_pts[:,2])
    t_med = np.median(t_pts[:,2])
    depth_mask = (q_pts[:,2] < 5*q_med) & (t_pts[:,2] < 5*t_med)
    counters['depth'] += np.sum(~depth_mask)
    matches, q_pts = matches[depth_mask], q_pts[depth_mask]
    if len(matches) == 0:
        return local_results, counters

    # l) Transform to world and collect
    w_pts = utils.transform_points(q_pts, q_frame.pose)
    for w, m in zip(w_pts, matches):
        local_results.append((w.tolist(), m.distance, m.queryIdx, m.trainIdx, q_frame.id))

    return local_results, counters