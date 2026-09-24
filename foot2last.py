#!/usr/bin/env python3
"""
foot2last.py — Shoe-making measurements from 3D foot scans (.obj/.stl/.gltf).

Handles normalised/unknown-unit meshes by auto-scaling.  Detects the scanning
base plane (the axis with most vertices near its minimum) and uses that as
the foot's length direction.  Cross-sections computed via vertex masking
(robust for solid-cube type scans).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh
from scipy.spatial import ConvexHull

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOG = logging.getLogger("foot2last")

DEFAULT_NORMALISED_SCALE = 508.0  # mm/unit for models with ~0.50 unit range

# ============================================================================
# Data classes
# ============================================================================
@dataclass
class CrossSection:
    z: float
    width: float = 0.0
    perimeter: float = 0.0
    n_points: int = 0
    center_x: float = 0.0

@dataclass
class Landmarks:
    """Anatomical landmarks detected on a foot scan."""
    heel_point: np.ndarray | None = None
    heel_top_point: np.ndarray | None = None
    toe_point: np.ndarray | None = None
    big_toe_point: np.ndarray | None = None
    second_toe_point: np.ndarray | None = None
    ball_point: np.ndarray | None = None
    instep_point: np.ndarray | None = None
    arch_point: np.ndarray | None = None
    ball_girth_point: np.ndarray | None = None

@dataclass
class FootMeasurements:
    side: str = "unknown"
    unit: str = "mm"
    length: float = 0.0
    length_to_big_toe: float = 0.0
    length_to_second_toe: float = 0.0
    heel_width: float = 0.0
    ball_width: float = 0.0
    waist_width: float = 0.0
    instep_height: float = 0.0
    heel_height: float = 0.0
    toe_box_height: float = 0.0
    ankle_height: float = 0.0
    ball_girth: float = 0.0
    instep_girth: float = 0.0
    heel_girth: float = 0.0
    waist_girth: float = 0.0
    volume: float = 0.0
    volume_hindfoot: float = 0.0
    volume_midfoot: float = 0.0
    volume_forefoot: float = 0.0
    toe_shape: str = "unknown"
    arch_type: str = "unknown"
    heel_shape: str = "unknown"
    us_men_size: float = 0.0
    us_women_size: float = 0.0
    eu_size: float = 0.0
    uk_size: float = 0.0
    mondopoint_cm: float = 0.0
    sections: list = field(default_factory=list)
    n_vertices: int = 0
    n_faces: int = 0
    scale_used: float = 1.0
    last_length: float = 0.0
    last_ball_width: float = 0.0
    last_instep_height: float = 0.0
    last_volume: float = 0.0
    last_formula_note: str = ""

    def to_dict(self):
        d = asdict(self)
        d["sections"] = [asdict(s) for s in self.sections]
        return d

# ============================================================================
# Geometry helpers
# ============================================================================
def _section_width_and_girth(pts_3d, length_axis, scale):
    if len(pts_3d) < 3:
        return 0.0, 0.0
    length_vec = np.zeros(3)
    length_vec[length_axis] = 1.0
    ref_up = np.array([0.0, 0.0, 1.0])
    if abs(length_vec @ ref_up) > 0.99:
        ref_up = np.array([0.0, 1.0, 0.0])
    u = np.cross(length_vec, ref_up)
    u_norm = np.linalg.norm(u)
    if u_norm < 1e-9:
        u = np.array([1.0, 0.0, 0.0])
        u_norm = 1.0
    u /= u_norm
    v = np.cross(length_vec, u)
    v /= np.linalg.norm(v)
    pts_2d = np.stack([pts_3d @ u, pts_3d @ v], axis=1)
    width = float(np.ptp(pts_2d[:, 0])) * scale
    try:
        hull = ConvexHull(pts_2d)
        girth = float(hull.area) * scale
    except Exception:
        centroid_2d = pts_2d.mean(axis=0)
        angles = np.arctan2(pts_2d[:, 1] - centroid_2d[1],
                            pts_2d[:, 0] - centroid_2d[0])
        sorted_idx = np.argsort(angles)
        sorted_pts = pts_2d[sorted_idx]
        diffs = np.diff(sorted_pts, axis=0)
        girth = float(np.sum(np.linalg.norm(diffs, axis=1))) * scale
    return width, girth

def _detect_base_axis(pts):
    """Find axis with most vertices near minimum = scanning base."""
    n = len(pts)
    best_axis = 0
    best_frac = 0.0
    for axis_idx in range(3):
        ax_min = pts[:, axis_idx].min()
        ax_max = pts[:, axis_idx].max()
        ax_range = ax_max - ax_min
        if ax_range < 1e-6:
            continue
        threshold = ax_min + 0.05 * ax_range
        base_count = np.sum(pts[:, axis_idx] < threshold)
        frac = base_count / n
        if frac > best_frac and frac > 0.25:
            best_frac = frac
            best_axis = axis_idx
    return best_axis

def _sample_sections(pts, length_axis, scale, n_sections=60):
    ax_min = pts[:, length_axis].min()
    ax_max = pts[:, length_axis].max()
    ax_range = ax_max - ax_min
    if ax_range < 1e-6:
        return []
    sections = []
    for frac in np.linspace(0.0, 1.0, n_sections):
        pos = ax_min + frac * ax_range
        tol = ax_range * 0.04
        mask = np.abs(pts[:, length_axis] - pos) < tol
        slice_pts = pts[mask]
        n_pts = len(slice_pts)
        if n_pts >= 3:
            w, g = _section_width_and_girth(slice_pts, length_axis, scale)
            sections.append(CrossSection(
                z=frac, width=w, perimeter=g,
                n_points=n_pts,
                center_x=float(slice_pts[:, (length_axis + 1) % 3].mean()) * scale,
            ))
        else:
            sections.append(CrossSection(z=frac, n_points=n_pts))
    return sections

# ============================================================================
# Landmark detection
# ============================================================================
def _detect_landmarks(pts, length_axis, scale):
    n = len(pts)
    if n < 50:
        raise ValueError("Mesh too small")

    vert_axis = 2  # Z-up for foot-in-cube scans
    other_axis = ({0, 1, 2} - {length_axis, vert_axis}).pop()

    length_vec = np.zeros(3)
    length_vec[length_axis] = 1.0
    proj = pts @ length_vec
    p_min, p_max = proj.min(), proj.max()
    foot_length = p_max - p_min
    if foot_length < 1e-6:
        raise ValueError("Degenerate length")

    heel_zone_mask = (proj >= p_min + 0.08 * foot_length) & \
                    (proj < p_min + 0.22 * foot_length)
    heel_zone_pts = pts[heel_zone_mask]
    if len(heel_zone_pts) > 0:
        heel_point = heel_zone_pts[heel_zone_pts[:, vert_axis].argmin()].copy()
        heel_top_point = heel_zone_pts[heel_zone_pts[:, vert_axis].argmax()].copy()
    else:
        heel_zone_pts = pts[proj < p_min + 0.25 * foot_length]
        heel_point = heel_zone_pts[heel_zone_pts[:, vert_axis].argmin()].copy()
        heel_top_point = heel_zone_pts[heel_zone_pts[:, vert_axis].argmax()].copy()

    toe_idx = np.argmax(proj)
    toe_point = pts[toe_idx].copy()

    toe_region_mask = proj > p_max - 0.08 * foot_length
    toe_region_pts = pts[toe_region_mask]
    big_toe_point = None
    second_toe_point = None
    if len(toe_region_pts) > 0:
        other_mean = toe_region_pts[:, other_axis].mean()
        for side, cond in [("big", toe_region_pts[:, other_axis] < other_mean),
                            ("second", toe_region_pts[:, other_axis] >= other_mean)]:
            if cond.any():
                idx_in_zone = np.where(toe_region_mask)[0][cond]
                pt = pts[idx_in_zone[proj[idx_in_zone].argmax()]].copy()
                if side == "big":
                    big_toe_point = pt
                else:
                    second_toe_point = pt

    best_width = 0.0
    ball_point = heel_point.copy()
    for frac in np.linspace(0.45, 0.75, 20):
        pos = p_min + frac * foot_length
        tol = foot_length * 0.04
        mask = np.abs(proj - pos) < tol
        slice_pts = pts[mask]
        if len(slice_pts) >= 3:
            w, _ = _section_width_and_girth(slice_pts, length_axis, scale)
            if w > best_width:
                best_width = w
                ball_point = slice_pts.mean(axis=0)

    mid_mask = (proj > p_min + 0.35 * foot_length) & \
               (proj < p_min + 0.55 * foot_length)
    mid_pts = pts[mid_mask]
    if len(mid_pts) > 0:
        instep_point = mid_pts[mid_pts[:, vert_axis].argmax()].copy()
    else:
        idx_max_z = np.argmax(pts[:, vert_axis])
        instep_point = pts[idx_max_z].copy()

    arch_candidates = mid_pts.copy()
    if len(arch_candidates) > 0:
        median_other = np.median(arch_candidates[:, other_axis])
        arch_point = arch_candidates[arch_candidates[:, other_axis] < median_other]
        if len(arch_point) > 0:
            arch_point = arch_point[arch_point[:, vert_axis].argmax()].copy()
        else:
            arch_point = None
    else:
        arch_point = None

    ball_girth_mask = (proj > p_min + 0.38 * foot_length) & \
                      (proj < p_min + 0.42 * foot_length)
    ball_girth_point = None
    if ball_girth_mask.any():
        bg_pts = pts[ball_girth_mask]
        if len(bg_pts) > 0:
            ball_girth_point = bg_pts.mean(axis=0)

    return Landmarks(
        heel_point=heel_point,
        heel_top_point=heel_top_point,
        toe_point=toe_point,
        big_toe_point=big_toe_point,
        second_toe_point=second_toe_point,
        ball_point=ball_point,
        instep_point=instep_point,
        arch_point=arch_point,
        ball_girth_point=ball_girth_point,
    )

# ============================================================================
# Shape classification
# ============================================================================
def _classify_toe_shape(pts, length_axis, scale):
    length_vec = np.zeros(3)
    length_vec[length_axis] = 1.0
    proj = pts @ length_vec
    p_min, p_max = proj.min(), proj.max()
    foot_length = p_max - p_min

    widths = []
    for frac in np.linspace(0.60, 1.0, 20):
        pos = p_min + frac * foot_length
        tol = foot_length * 0.04
        mask = np.abs(proj - pos) < tol
        slice_pts = pts[mask]
        if len(slice_pts) >= 3:
            w, _ = _section_width_and_girth(slice_pts, length_axis, scale)
            widths.append(w)
        else:
            widths.append(0.0)
    widths = np.array(widths)
    if len(widths) < 5:
        return "unknown"

    if widths[-1] < widths[0] * 0.40:
        return "pointed"
    if widths[-1] < widths[0] * 0.65:
        return "tapered"
    slope = widths[-1] - widths[0]
    if slope < -2.0 * scale:
        return "tapered"
    return "rounded"

def _classify_arch_type(pts, length_axis, scale):
    length_vec = np.zeros(3)
    length_vec[length_axis] = 1.0
    proj = pts @ length_vec
    p_min, p_max = proj.min(), proj.max()
    foot_length = p_max - p_min
    if foot_length < 1e-6:
        return "unknown"

    arch_widths = []
    for frac in np.linspace(0.35, 0.60, 15):
        pos = p_min + frac * foot_length
        tol = foot_length * 0.04
        mask = np.abs(proj - pos) < tol
        slice_pts = pts[mask]
        if len(slice_pts) >= 3:
            w, _ = _section_width_and_girth(slice_pts, length_axis, scale)
            arch_widths.append(w)
        else:
            arch_widths.append(0.0)
    arch_widths = np.array(arch_widths)
    if len(arch_widths) < 3:
        return "unknown"

    ball_w = 0.0
    for frac in np.linspace(0.45, 0.70, 10):
        pos = p_min + frac * foot_length
        tol = foot_length * 0.04
        mask = np.abs(proj - pos) < tol
        slice_pts = pts[mask]
        if len(slice_pts) >= 3:
            w, _ = _section_width_and_girth(slice_pts, length_axis, scale)
            ball_w = max(ball_w, w)

    if ball_w < 1.0:
        return "unknown"

    arch_mean = np.mean(arch_widths)
    ratio = arch_mean / ball_w
    if ratio < 0.50:
        return "high"
    if ratio < 0.68:
        return "normal"
    return "flat"

def _classify_heel_shape(pts, length_axis, scale):
    length_vec = np.zeros(3)
    length_vec[length_axis] = 1.0
    proj = pts @ length_vec
    p_min, p_max = proj.min(), proj.max()
    foot_length = p_max - p_min

    ball_width = 0.0
    for frac in np.linspace(0.45, 0.75, 15):
        pos = p_min + frac * foot_length
        tol = foot_length * 0.04
        mask = np.abs(proj - pos) < tol
        slice_pts = pts[mask]
        if len(slice_pts) >= 3:
            w, _ = _section_width_and_girth(slice_pts, length_axis, scale)
            ball_width = max(ball_width, w)

    heel_width = 0.0
    for frac in np.linspace(0.10, 0.22, 6):
        pos = p_min + frac * foot_length
        tol = foot_length * 0.04
        mask = np.abs(proj - pos) < tol
        slice_pts = pts[mask]
        if len(slice_pts) >= 3:
            w, _ = _section_width_and_girth(slice_pts, length_axis, scale)
            heel_width = max(heel_width, w)

    if ball_width < 1.0 or heel_width < 1.0:
        return "unknown"
    ratio = heel_width / ball_width
    if ratio < 0.40:
        return "narrow"
    if ratio > 0.58:
        return "wide"
    return "medium"

# ============================================================================
# Volume (convex-hull based for non-watertight foot-in-cube scans)
# ============================================================================
def _compute_volume(pts, length_axis, scale):
    try:
        ch = ConvexHull(pts)
        total = float(ch.volume) * (scale ** 3)
    except Exception:
        total = 0.0

    length_vec = np.zeros(3)
    length_vec[length_axis] = 1.0
    proj = pts @ length_vec
    p_min, p_max = proj.min(), proj.max()
    foot_length = p_max - p_min

    zones = {}
    for name, lo, hi in [("hindfoot", 0.0, 0.30),
                          ("midfoot", 0.30, 0.65),
                          ("forefoot", 0.65, 1.0)]:
        mask = (proj >= p_min + lo * foot_length) & \
               (proj < p_min + hi * foot_length)
        sub_pts = pts[mask]
        if len(sub_pts) < 20:
            zones[name] = 0.0
            continue
        try:
            ch_sub = ConvexHull(sub_pts)
            zones[name] = float(ch_sub.volume) * (scale ** 3)
        except Exception:
            zones[name] = 0.0

    return total, zones

# ============================================================================
# Main measurement
# ============================================================================
def measure_foot(pts, length_axis, scale, side_label, n_verts, n_faces):
    meas = FootMeasurements(
        side=side_label, unit="mm",
        n_vertices=n_verts, n_faces=n_faces,
        scale_used=scale,
    )
    if len(pts) < 50:
        return meas

    try:
        lm = _detect_landmarks(pts, length_axis, scale)
    except ValueError as exc:
        LOG.error("Landmark detection failed: %s", exc)
        return meas

    length_vec = np.zeros(3)
    length_vec[length_axis] = 1.0
    proj = pts @ length_vec
    p_min, p_max = proj.min(), proj.max()
    foot_length = p_max - p_min
    meas.length = round(float(foot_length * scale), 1)

    floor_vert = pts[:, 2].min()

    meas.length_to_big_toe = round(
        float(np.linalg.norm(lm.big_toe_point - lm.heel_point) * scale), 1) \
        if lm.big_toe_point is not None else 0.0
    meas.length_to_second_toe = round(
        float(np.linalg.norm(lm.second_toe_point - lm.heel_point) * scale), 1) \
        if lm.second_toe_point is not None else 0.0

    meas.instep_height = round(float((lm.instep_point[2] - floor_vert) * scale), 1)
    meas.heel_height = round(float((lm.heel_top_point[2] - floor_vert) * scale), 1)
    meas.toe_box_height = round(float((lm.toe_point[2] - floor_vert) * scale), 1)

    anc_mask = (proj > p_min + 0.70 * foot_length) & \
               (proj < p_min + 0.90 * foot_length)
    anc_pts = pts[anc_mask]
    meas.ankle_height = (
        round(float((anc_pts[:, 2].max() - floor_vert) * scale), 1)
        if len(anc_pts) > 0 else meas.instep_height
    )

    def _w(frac):
        pos = p_min + frac * foot_length
        tol = foot_length * 0.04
        mask = np.abs(proj - pos) < tol
        sp = pts[mask]
        if len(sp) >= 3:
            w, _ = _section_width_and_girth(sp, length_axis, scale)
            return round(w, 1)
        return 0.0

    # Heel width: 8-22% (exclude scanning base below 8%)
    heel_widths = [_w(f) for f in np.linspace(0.08, 0.22, 8)]
    meas.heel_width = max(heel_widths) if any(heel_widths) else 0.0

    # Ball width: 45-75%
    ball_widths = [_w(f) for f in np.linspace(0.45, 0.75, 15)]
    meas.ball_width = max(ball_widths) if any(ball_widths) else 0.0

    # Waist: 35-55%
    waist_widths = [_w(f) for f in np.linspace(0.35, 0.55, 10)]
    meas.waist_width = max(waist_widths) if any(waist_widths) else 0.0

    def _g(frac):
        pos = p_min + frac * foot_length
        tol = foot_length * 0.04
        mask = np.abs(proj - pos) < tol
        sp = pts[mask]
        if len(sp) >= 3:
            _, g = _section_width_and_girth(sp, length_axis, scale)
            return round(g, 1)
        return 0.0

    ball_girths = [_g(f) for f in np.linspace(0.40, 0.45, 5)]
    meas.ball_girth = max(ball_girths) if any(ball_girths) else 0.0
    instep_girths = [_g(f) for f in np.linspace(0.35, 0.50, 8)]
    meas.instep_girth = max(instep_girths) if any(instep_girths) else 0.0
    heel_girths = [_g(f) for f in np.linspace(0.08, 0.18, 5)]
    meas.heel_girth = max(heel_girths) if any(heel_girths) else 0.0
    waist_girths = [_g(f) for f in np.linspace(0.35, 0.50, 8)]
    meas.waist_girth = max(waist_girths) if any(waist_girths) else 0.0

    total_vol, zones = _compute_volume(pts, length_axis, scale)
    meas.volume = round(total_vol, 1)
    meas.volume_hindfoot = round(zones.get("hindfoot", 0.0), 1)
    meas.volume_midfoot = round(zones.get("midfoot", 0.0), 1)
    meas.volume_forefoot = round(zones.get("forefoot", 0.0), 1)

    meas.toe_shape = _classify_toe_shape(pts, length_axis, scale)
    meas.arch_type = _classify_arch_type(pts, length_axis, scale)
    meas.heel_shape = _classify_heel_shape(pts, length_axis, scale)

    l = meas.length
    if l >= 10.0:
        ea = (l - 14.0) / 1.5
        meas.eu_size = round(ea, 1)
        meas.uk_size = round(ea - 0.5, 1)
        meas.us_men_size = round(ea - 0.5, 1)
        meas.us_women_size = round(ea - 1.0, 1)
        meas.mondopoint_cm = round(l / 10.0, 1)

    meas.last_length = round(l + 6.0, 1)
    meas.last_ball_width = round(meas.ball_width + 3.0, 1)
    meas.last_instep_height = round(meas.instep_height + 4.0, 1)
    meas.last_volume = meas.volume
    meas.last_formula_note = (
        "Last = foot + 6.0 mm length, + 3.0 mm ball width, instep + 4.0 mm."
    )

    meas.sections = _sample_sections(pts, length_axis, scale, n_sections=60)

    return meas

# ============================================================================
# Pair comparison
# ============================================================================
PAIR_ADVICE = {
    "size_difference": (
        "If foot lengths differ by > 3 mm, build the last for the larger foot "
        "and add a custom insole for the smaller foot."
    ),
    "width_difference": (
        "Ball widths > 5 mm apart warrant different width lasts "
        "(e.g., E vs F for UK sizes)."
    ),
    "girth_difference": (
        "Instep girths > 8 mm apart suggest a custom last for the high-instep foot."
    ),
    "general": (
        "Always build the last for the LARGER foot. "
        "If asymmetry > 3% on key dimensions, consider two separate lasts."
    ),
}

def pair_advice(m1, m2):
    advice = {}
    if m1.length > 0 and m2.length > 0:
        dl = abs(m1.length - m2.length)
        advice["length_difference_mm"] = round(dl, 1)
        advice["size_difference_note"] = PAIR_ADVICE["size_difference"]
    if m1.ball_width > 0 and m2.ball_width > 0:
        dw = abs(m1.ball_width - m2.ball_width)
        advice["width_difference_mm"] = round(dw, 1)
        advice["width_difference_note"] = PAIR_ADVICE["width_difference"]
    if m1.instep_girth > 0 and m2.instep_girth > 0:
        dg = abs(m1.instep_girth - m2.instep_girth)
        advice["girth_difference_mm"] = round(dg, 1)
        advice["girth_difference_note"] = PAIR_ADVICE["girth_difference"]
    advice["general_note"] = PAIR_ADVICE["general"]
    return advice

def auto_label_sides(left_pts, right_pts):
    if left_pts is None or right_pts is None:
        return "left", "right"

    left_proj = left_pts[:, 1]
    right_proj = right_pts[:, 1]
    if left_proj.mean() < right_proj.mean():
        return "left", "right"
    return "right", "left"

# ============================================================================
# Split two feet
# ============================================================================
def split_feet(pts, faces):
    n = len(pts)
    if n < 50:
        return None, None, None, None

    centroid = pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts - centroid)
    long_dir = Vt[0].copy()

    if abs(long_dir[0]) < 0.9:
        lateral = np.array([1.0, 0.0, 0.0])
    else:
        lateral = np.array([0.0, 1.0, 0.0])
    lateral -= long_dir * np.dot(lateral, long_dir)
    lateral /= np.linalg.norm(lateral)

    lat_proj = pts @ lateral
    median = np.median(lat_proj)

    left_mask = lat_proj < median
    right_mask = lat_proj >= median

    if left_mask.sum() < 20 or right_mask.sum() < 20:
        hist, edges = np.histogram(lat_proj, bins=50)
        gaps = np.diff(hist)
        gap_idx = np.argmin(gaps)
        split_val = (edges[gap_idx] + edges[gap_idx + 1]) / 2
        left_mask = lat_proj < split_val
        right_mask = lat_proj >= split_val

    def _sub(pts_mask):
        if pts_mask.sum() < 20:
            return None, None
        old_idx = np.where(pts_mask)[0]
        old_to_new = -np.ones(n, dtype=np.int64)
        old_to_new[old_idx] = np.arange(len(old_idx))
        new_pts = pts[old_idx].copy()
        if faces is not None and len(faces) > 0:
            fmask = pts_mask[faces[:, 0]] & pts_mask[faces[:, 1]] & pts_mask[faces[:, 2]]
            if fmask.any():
                sf = old_to_new[faces[fmask]]
                if sf.min() >= 0:
                    return new_pts, sf
        return new_pts, None

    return _sub(left_mask), _sub(right_mask)

# ============================================================================
# Quality report  (replaces trimesh.is_watertight)
# ============================================================================
def quality_report(pts, faces, scale):
    n_v = len(pts)
    n_f = len(faces) if faces is not None else 0

    try:
        tr = trimesh.Trimesh(vertices=pts, faces=faces, process=True)
        is_wt = tr.is_watertight
        is_man = tr.is_manifold
        vol = float(tr.volume) * (scale ** 3) if tr.is_watertight else None
    except Exception:
        is_wt = False
        is_man = False
        vol = None

    bmin = pts.min(axis=0)
    bmax = pts.max(axis=0)
    ext_xyz = (bmax - bmin) * scale

    return {
        "summary": {
            "n_vertices": n_v,
            "n_faces": n_f,
            "watertight": is_wt,
            "manifold": is_man,
            "volume_mm3": round(vol, 1) if vol is not None else None,
        },
        "bounding_box_mm": {
            "x": round(float(ext_xyz[0]), 1),
            "y": round(float(ext_xyz[1]), 1),
            "z": round(float(ext_xyz[2]), 1),
        },
        "notes": [],
    }

def _print_quality(q):
    s = q["summary"]
    print("\n=== SCAN QUALITY REPORT ===")
    print(f"Vertices   : {s['n_vertices']:,}")
    print(f"Faces      : {s['n_faces']:,}")
    print(f"Water-tight: {s['watertight']}")
    print(f"Manifold   : {s['manifold']}")
    if s["volume_mm3"] is not None:
        print(f"Volume     : {s['volume_mm3']:,.1f} mm³")
    bb = q["bounding_box_mm"]
    print(f"Bounding box: {bb['x']:.1f} × {bb['y']:.1f} × {bb['z']:.1f} mm")
    for note in q["notes"]:
        print(f"  NOTE: {note}")
    print()

# ============================================================================
# Visual report
# ============================================================================
def _visual_report(meas, png_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        sec = meas.sections
        if len(sec) < 4:
            return

        fracs = [s.z for s in sec]
        widths = [s.width for s in sec]
        girths = [s.perimeter for s in sec]

        ax1.fill_between(fracs, widths, alpha=0.3, color="#3399ff")
        ax1.plot(fracs, widths, color="#0055cc", linewidth=2)
        ax1.axvline(0.08, color="#ff4444", linestyle="--", alpha=0.7, label="base limit")
        ax1.axvline(0.10, color="#ff8800", linestyle="--", alpha=0.7, label="heel zone")
        ax1.axvline(0.45, color="#44aa00", linestyle="--", alpha=0.7, label="ball zone start")
        ax1.axvline(0.70, color="#aa00ff", linestyle="--", alpha=0.7, label="ball zone end")
        ax1.set_xlabel("Fraction of foot length (0=heel, 1=toe)")
        ax1.set_ylabel("Width (mm)")
        ax1.set_title(f"Cross-section widths — {meas.side} foot\n"
                      f"Length: {meas.length} mm, Ball: {meas.ball_width} mm, "
                      f"Heel: {meas.heel_width} mm")
        ax1.legend(fontsize=8, loc="upper right")
        ax1.grid(True, alpha=0.3)

        ax2.fill_between(fracs, girths, alpha=0.3, color="#cc3399")
        ax2.plot(fracs, girths, color="#880066", linewidth=2)
        ax2.set_xlabel("Fraction of foot length")
        ax2.set_ylabel("Girth / perimeter (mm)")
        ax2.set_title(f"Cross-section perimeters\n"
                      f"Ball girth: {meas.ball_girth} mm, "
                      f"Instep girth: {meas.instep_girth} mm")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(png_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        LOG.info("Visual report → %s", png_path)
    except Exception as exc:
        LOG.warning("Could not render visual report: %s", exc)

# ============================================================================
# Main
# ============================================================================
def main():
    p = argparse.ArgumentParser(
        description="foot2last.py — Shoe-making measurements from 3D foot scans")
    p.add_argument("inputs", nargs="+", type=Path,
                   help="One or more 3D scan files (.obj .stl .gltf .glb)")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="JSON output path (default: stdout)")
    p.add_argument("-s", "--single-foot", action="store_true",
                   help="Treat each input as a single foot (skip auto-split)")
    p.add_argument("--left-label", type=str, default=None,
                   help="Label for first foot when processing 2 files with --single-foot")
    p.add_argument("--right-label", type=str, default=None,
                   help="Label for second foot when processing 2 files with --single-foot")
    p.add_argument("--unit", choices=["mm", "cm", "m"], default="mm",
                   help="Input mesh unit (default: auto)")
    p.add_argument("--force-scale", type=float, default=None,
                   help="Override auto-detected scale (mm per mesh unit)")
    p.add_argument("--plots-dir", type=Path, default=None,
                   help="Directory for PNG visual reports")
    p.add_argument("--quality", action="store_true",
                   help="Print scan quality report instead of measurements")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Verbose logging")

    args = p.parse_args()
    if args.verbose:
        LOG.setLevel(logging.DEBUG)

    results = {}
    input_paths = args.inputs

    if args.single_foot:
        for i, inp in enumerate(input_paths):
            if not inp.exists():
                LOG.error("Input file not found: %s", inp)
                return 1

            try:
                mesh = trimesh.load(inp)
            except Exception as exc:
                LOG.error("Could not load %s: %s", inp, exc)
                return 1

            pts = mesh.vertices.astype(np.float64)
            faces = mesh.faces
            n_v = len(pts)
            n_f = len(faces) if faces is not None else 0

            if args.force_scale:
                scale = args.force_scale
            else:
                scale = detect_scale(pts)

            LOG.info("Loaded %s: %s vertices, %s faces",
                     inp.name, f"{n_v:,}", f"{n_f:,}")

            base_axis = _detect_base_axis(pts)
            if base_axis >= 0:
                LOG.info("Detected scanning base on axis %d → length axis = %d, vertical axis = Z",
                         base_axis, base_axis)
                length_axis = base_axis
            else:
                LOG.info("No clear base; using Y-axis as length (Z-up for vertical)")
                length_axis = 1

            if args.quality:
                q = quality_report(pts, faces, scale)
                _print_quality(q)
                return 0

            label = f"foot_{i}"
            if args.left_label and i == 0:
                label = args.left_label
            elif args.right_label and i == 1:
                label = args.right_label

            meas = measure_foot(pts, length_axis, scale, label,
                                n_v, n_f)
            results[label] = meas

            print(f"\n{'='*80}")
            print(f"  {meas.side.upper()} FOOT → LAST MEASUREMENTS")
            print(f"{'='*80}")
            _print_report(meas)

            if args.plots_dir:
                _visual_report(meas,
                               args.plots_dir / f"foot_{label}_sections.png")

        if len(results) == 2:
            k0, k1 = list(results.keys())
            advice = pair_advice(results[k0], results[k1])
            print(f"\n{'='*80}")
            print("  PAIR COMPARISON ADVICE")
            print(f"{'='*80}")
            for key, value in advice.items():
                print(f"  • {key}: {value}")
            print("  PAIR COMPARISON ADVICE")
            print(f"{'='*80}")
            for key, value in advice.items():
                print(f"  • {key}: {value}")

    else:
        # Try to auto-split a single combined scan
        inp = input_paths[0]
        if not inp.exists():
            LOG.error("Input file not found: %s", inp)
            return 1

        try:
            mesh = trimesh.load(inp)
        except Exception as exc:
            LOG.error("Could not load %s: %s", inp, exc)
            return 1

        pts = mesh.vertices.astype(np.float64)
        faces = mesh.faces
        n_v = len(pts)
        n_f = len(faces) if faces is not None else 0
        scale = args.force_scale if args.force_scale else detect_scale(pts)

        LOG.info("Loaded %s: %s vertices, %s faces",
                 inp.name, f"{n_v:,}", f"{n_f:,}")
        LOG.info("Normalised mesh (max extent %.3f units). Scale=%.1f mm/unit",
                 pts.max() - pts.min(), scale)

        left_pts, left_faces, right_pts, right_faces = split_feet(pts, faces)
        if left_pts is None or right_pts is None:
            LOG.warning(
                "Auto-split failed (likely single foot). "
                "Re-run with --single-foot to measure directly."
            )
            meas = measure_foot(pts, length_axis, scale, "foot",
                                n_v, n_f)
            results["foot"] = meas
        else:
            lbl_l, lbl_r = auto_label_sides(left_pts, right_pts)
            LOG.info("Auto-split: %s (%s verts) / %s (%s verts)",
                     lbl_l, f"{len(left_pts):,}", lbl_r, f"{len(right_pts):,}")

            meas_l = measure_foot(left_pts, length_axis, scale, lbl_l,
                                  len(left_pts), len(left_faces) if left_faces is not None else 0)
            meas_r = measure_foot(right_pts, length_axis, scale, lbl_r,
                                  len(right_pts), len(right_faces) if right_faces is not None else 0)
            results[lbl_l] = meas_l
            results[lbl_r] = meas_r

            print(f"\n{'='*80}")
            print(f"  LEFT FOOT → LAST MEASUREMENTS")
            print(f"{'='*80}")
            _print_report(meas_l)
            print(f"\n{'='*80}")
            print(f"  RIGHT FOOT → LAST MEASUREMENTS")
            print(f"{'='*80}")
            _print_report(meas_r)

            advice = pair_advice(meas_l, meas_r)
            print(f"\n{'='*80}")
            print("  PAIR COMPARISON ADVICE")
            print(f"{'='*80}")
            for key, value in advice.items():
                print(f"  • {key}: {value}")

        if args.plots_dir:
            for label, meas in results.items():
                _visual_report(meas,
                               args.plots_dir / f"foot_{label}_sections.png")

    if args.output:
        out = {}
        for label, meas in results.items():
            out[label] = meas.to_dict()
        with open(args.output, "w") as fh:
            json.dump(out, fh, indent=2)
        LOG.info("Results → %s", args.output)
    else:
        out = {}
        for label, meas in results.items():
            out[label] = meas.to_dict()
        print(json.dumps(out, indent=2))

    return 0

def _print_report(meas):
    print(f"\n  ── {meas.side.upper()} ──")
    print(f"\n  Foot length (heel→toe)      : {meas.length:6.1f} mm")
    print(f"  Length to big toe            : {meas.length_to_big_toe:6.1f} mm")
    print(f"  Length to 2nd toe            : {meas.length_to_second_toe:6.1f} mm")
    print(f"\n  Ball width                   : {meas.ball_width:6.1f} mm")
    print(f"  Heel width                   : {meas.heel_width:6.1f} mm")
    print(f"  Waist (arch) width           : {meas.waist_width:6.1f} mm")
    print(f"\n  Instep height (floor→top)    : {meas.instep_height:6.1f} mm")
    print(f"  Heel height                  : {meas.heel_height:6.1f} mm")
    print(f"  Toe-box height              : {meas.toe_box_height:6.1f} mm")
    print(f"  Ankle height                 : {meas.ankle_height:6.1f} mm")
    print(f"\n  Ball girth                   : {meas.ball_girth:6.1f} mm")
    print(f"  Instep girth                 : {meas.instep_girth:6.1f} mm")
    print(f"  Heel girth                   : {meas.heel_girth:6.1f} mm")
    print(f"  Waist girth                  : {meas.waist_girth:6.1f} mm")
    print(f"\n  Total volume                 : {meas.volume:7.1f} mm³")
    print(f"    Hindfoot                   : {meas.volume_hindfoot:7.1f} mm³")
    print(f"    Midfoot                    : {meas.volume_midfoot:7.1f} mm³")
    print(f"    Forefoot                   : {meas.volume_forefoot:7.1f} mm³")
    print(f"\n  Toe shape                    : {meas.toe_shape}")
    print(f"  Arch type                    : {meas.arch_type}")
    print(f"  Heel shape                   : {meas.heel_shape}")
    print(f"\n  Estimated sizes:")
    print(f"    EU                         : {meas.eu_size:5.1f}")
    print(f"    UK                         : {meas.uk_size:5.1f}")
    print(f"    US Men's                   : {meas.us_men_size:5.1f}")
    print(f"    US Women's                 : {meas.us_women_size:5.1f}")
    print(f"    Mondopoint                 : {meas.mondopoint_cm:5.1f} cm")
    print(f"\n  LAST DESIGN PARAMETERS:")
    print(f"    Last length                : {meas.last_length:6.1f} mm")
    print(f"    Last ball width            : {meas.last_ball_width:6.1f} mm")
    print(f"    Last instep height         : {meas.last_instep_height:6.1f} mm")
    print(f"    Last internal volume       : {meas.last_volume:7.1f} mm³")
    if meas.heel_width > 0:
        print(f"    Heel cup width             : {meas.heel_width + 3.0:6.1f} mm")
        print(f"    Heel cup height            : {meas.heel_height + 5.0:6.1f} mm")
    print(f"    Toe spring                 : {12.0:6.1f} mm")
    if meas.arch_type == "high":
        print(f"    Arch support note          : High arch: raise medial arch of last and add cushioning.")
    elif meas.arch_type == "flat":
        print(f"    Arch support note          : Flat arch: add arch support to prevent over-pronation.")
    print(f"    Formula note               : {meas.last_formula_note}")

def detect_scale(pts):
    max_ext = float((pts.max(axis=0) - pts.min(axis=0)).max())
    if max_ext > 100.0:
        LOG.info("Large mesh (max extent %.1f) — assuming mm, scale=1.0", max_ext)
        return 1.0
    elif max_ext > 2.0:
        LOG.info("Medium mesh (max extent %.1f) — checking for cm... ", max_ext)
        return 10.0 if max_ext < 50.0 else 1.0
    else:
        LOG.info("Normalised mesh (max extent %.3f units). Scale=%.1f mm/unit → ~%.0f mm foot.",
                 max_ext, DEFAULT_NORMALISED_SCALE,
                 max_ext * DEFAULT_NORMALISED_SCALE)
        return DEFAULT_NORMALISED_SCALE

if __name__ == "__main__":
    sys.exit(main())
