import cv2
import numpy as np
import os
import glob
import math
import csv
import json
import threading
import traceback
import tkinter as tk
from tkinter import messagebox, simpledialog, filedialog
from tkinter import ttk
import concurrent.futures
import multiprocessing

# =====================================================================
# Utilities: Shared & Flake Analyzer
# =====================================================================
def deep_merge_config(default_dict, loaded_dict):
    for key, val in loaded_dict.items():
        if key in default_dict and isinstance(default_dict[key], dict) and isinstance(val, dict):
            deep_merge_config(default_dict[key], val)
        else:
            default_dict[key] = val

def calculate_circular_hue_mean(hue_array):
    if len(hue_array) == 0: return 0.0
    rad_array = hue_array * (np.pi / 90.0)
    mean_cos = np.mean(np.cos(rad_array))
    mean_sin = np.mean(np.sin(rad_array))
    mean_rad = np.arctan2(mean_sin, mean_cos)
    if mean_rad < 0: mean_rad += 2.0 * np.pi
    return mean_rad * (90.0 / np.pi)

def get_wrapped_hsv_mask(hsv_img, lower_bound, upper_bound):
    h_low, s_low, v_low = lower_bound
    h_up, s_up, v_up = upper_bound

    if h_low <= h_up:
        return cv2.inRange(hsv_img, lower_bound, upper_bound)
    else:
        mask1 = cv2.inRange(hsv_img, np.array([h_low, s_low, v_low], dtype=np.uint8), np.array([179, s_up, v_up], dtype=np.uint8))
        mask2 = cv2.inRange(hsv_img, np.array([0, s_low, v_low], dtype=np.uint8), np.array([h_up, s_up, v_up], dtype=np.uint8))
        return cv2.bitwise_or(mask1, mask2)

def get_dilated_mask(base_mask, expand_px):
    if expand_px <= 0: return base_mask.copy()
    k_size = int(expand_px * 2) | 1 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    return cv2.dilate(base_mask, kernel, iterations=1)

# =====================================================================
# Utilities: EBL Aligner
# =====================================================================
def parse_grid_to_xy(grid_str):
    if len(grid_str) == 2:
        return int(grid_str[0]), int(grid_str[1])
    elif len(grid_str) == 4:
        return int(grid_str[0:2]), int(grid_str[2:4])
    elif len(grid_str) == 3:
        if grid_str.startswith('10'):
            return 10, int(grid_str[2])
        else:
            return int(grid_str[0]), 10
    return 0, 0

def refine_mark_centroid(roi_gray, guess_x, guess_y, win=30):
    # Intensity-weighted centroid of the bright cross blob nearest the guess.
    # Robust to nearby dust (connected-component selection), but slightly biased
    # when the cross arms are asymmetric. Returns (x, y) in ROI coords or None.
    rh, rw = roi_gray.shape
    x0, y0 = int(round(guess_x)), int(round(guess_y))
    l, r = max(0, x0 - win), min(rw, x0 + win + 1)
    t, b = max(0, y0 - win), min(rh, y0 + win + 1)
    patch = roi_gray[t:b, l:r].astype(np.float32)
    if patch.size == 0:
        return None
    thr, _ = cv2.threshold(patch.astype(np.uint8), 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m8 = ((patch >= thr) * 255).astype(np.uint8)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(m8, connectivity=8)
    if n <= 1:
        return None
    gxl, gyl = guess_x - l, guess_y - t
    best, bestd = None, 1e9
    for i in range(1, n):
        d = math.hypot(cents[i][0] - gxl, cents[i][1] - gyl)
        if d < bestd:
            bestd, best = d, i
    if best is None:
        return None
    comp = (lab == best).astype(np.float32)
    wmask = comp * np.clip(patch - patch.min(), 0, None)
    if wmask.sum() < 1e-6:
        return None
    ys, xs = np.mgrid[0:patch.shape[0], 0:patch.shape[1]]
    return (l + (xs * wmask).sum() / wmask.sum(),
            t + (ys * wmask).sum() / wmask.sum())


def refine_mark_line_intersection(roi_gray, guess_x, guess_y, win=40):
    # Fit a line to each arm of the cross and return their intersection.
    # Independent of arm thickness and asymmetry, so it tracks the true
    # geometric center better than a centroid. Returns (x, y) or None.
    rh, rw = roi_gray.shape
    x0, y0 = int(round(guess_x)), int(round(guess_y))
    l, r = max(0, x0 - win), min(rw, x0 + win + 1)
    t, b = max(0, y0 - win), min(rh, y0 + win + 1)
    patch = roi_gray[t:b, l:r].astype(np.uint8)
    if patch.size == 0:
        return None
    thr, _ = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (patch >= thr).astype(np.uint8)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(mask * 255, connectivity=8)
    if n <= 1:
        return None
    gxl, gyl = guess_x - l, guess_y - t
    best, bestd = None, 1e9
    for i in range(1, n):
        d = math.hypot(cents[i][0] - gxl, cents[i][1] - gyl)
        if d < bestd:
            bestd, best = d, i
    ys, xs = np.where(lab == best)
    if len(xs) < 20:
        return None
    cxl, cyl = xs.mean(), ys.mean()
    dx, dy = xs - cxl, ys - cyl
    horiz = np.abs(dx) >= np.abs(dy)   # pixels belonging to the horizontal arm
    vert = ~horiz                       # pixels belonging to the vertical arm
    if horiz.sum() < 10 or vert.sum() < 10:
        return None

    def fit(xp, yp, vertical):
        if vertical:
            # nearly vertical: x = m*y + k  ->  1*x + (-m)*y = k
            A = np.vstack([yp, np.ones_like(yp)]).T
            m, k = np.linalg.lstsq(A, xp, rcond=None)[0]
            return (1.0, -m, k)
        else:
            # nearly horizontal: y = m*x + k  ->  (-m)*x + 1*y = k
            A = np.vstack([xp, np.ones_like(xp)]).T
            m, k = np.linalg.lstsq(A, yp, rcond=None)[0]
            return (-m, 1.0, k)

    a1, b1, c1 = fit(xs[horiz].astype(float), ys[horiz].astype(float), False)
    a2, b2, c2 = fit(xs[vert].astype(float), ys[vert].astype(float), True)
    A = np.array([[a1, b1], [a2, b2]])
    C = np.array([c1, c2])
    try:
        sol = np.linalg.solve(A, C)
    except np.linalg.LinAlgError:
        return None
    return (l + sol[0], t + sol[1])


def get_mark_nominal_uv(filename, suffix="_ShiftN"):
    base_name = os.path.basename(filename).split(suffix)[0]
    grids = base_name.split('_')
    x1, y1 = parse_grid_to_xy(grids[0])
    
    ll_x = -4200 + (x1 * 800)
    ll_y = -4200 + (y1 * 800)
    
    if len(grids) == 2:
        x2, y2 = parse_grid_to_xy(grids[1])
        if x2 > x1: ll_x += 400
        if y2 > y1: ll_y += 400
            
    marks_uv = {
        'LL': [ll_x, ll_y],
        'LR': [ll_x + 400, ll_y],
        'UL': [ll_x, ll_y + 400]
    }
    return marks_uv

# =====================================================================
# Core Processing Logic: Flake Analyzer (For Workers)
# =====================================================================
def process_single_image(args):
    orig_threads = cv2.getNumThreads()
    cv2.setNumThreads(0) 
    
    img_path, out_dir_path, px_per_um, flake_bounds, bg_bounds, config = args
    filename = os.path.basename(img_path)
    
    try:
        img = cv2.imread(img_path)
        if img is None:
            return (filename, False, 0, "Could not read image file or file is corrupted.")

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        min_area = config["flake_target"]["min_area_um2"]
        max_area = config["flake_target"]["max_area_um2"]
        max_aspect_ratio = config["flake_target"]["max_aspect_ratio"]
        
        clearance_um = config["isolation_rules"]["clearance_radius_um"]
        ignore_debris_um2 = config["isolation_rules"]["ignore_micro_debris_um2"]
        noise_tol_percent = config["isolation_rules"]["noise_tolerance_percent"] / 100.0
        clr_method = config["isolation_rules"].get("clearance_method", "dilation")
        halo_buffer_um = config["isolation_rules"].get("halo_buffer_um", 0.5)
        
        strict_color_filter = config["advanced_filters"]["strict_blob_color_verification"]
        strict_color_tol = config["advanced_filters"]["strict_color_tolerance"]
        morph_kernel_um = config["advanced_filters"]["morphology_kernel_um"]
        color_erosion_um = config["advanced_filters"]["color_check_erosion_um"]
        
        circle_scale = config["display"]["circle_scale_percent"]
        # Overlay display toggles and sizes (backward-compatible defaults).
        disp = config["display"]
        show_circle = disp.get("show_circle", True)
        show_label = disp.get("show_label", True)
        show_clearance = disp.get("show_clearance", True)
        line_thickness = int(disp.get("line_thickness", 3))
        font_scale = float(disp.get("font_scale", 0.5))
        output_suffix = config["file_io"]["output_suffix"]
        jpeg_quality = config["file_io"]["jpeg_quality"]
        save_debug = config["file_io"].get("save_debug_images", False)

        clearance_px = int(clearance_um * px_per_um)
        halo_buffer_px = int(halo_buffer_um * px_per_um)
        
        flake_lower, flake_upper = flake_bounds
        bg_lower, bg_upper = bg_bounds
        
        k_px = max(3, int(morph_kernel_um * px_per_um))
        if k_px % 2 == 0: k_px += 1
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_px, k_px))

        flake_mask = get_wrapped_hsv_mask(hsv, flake_lower, flake_upper)
        bg_mask = get_wrapped_hsv_mask(hsv, bg_lower, bg_upper)

        f_ex_min, f_ex_max = config["flake_target"].get("exclude_h_range", [0, 0])
        if f_ex_max > 0 or f_ex_min > 0:
            f_ex_mask = get_wrapped_hsv_mask(hsv, np.array([f_ex_min, 0, 0], dtype=np.uint8), np.array([f_ex_max, 255, 255], dtype=np.uint8))
            flake_mask = cv2.bitwise_and(flake_mask, cv2.bitwise_not(f_ex_mask))
            
        b_ex_min, b_ex_max = config["background"].get("exclude_h_range", [0, 0])
        if b_ex_max > 0 or b_ex_min > 0:
            b_ex_mask = get_wrapped_hsv_mask(hsv, np.array([b_ex_min, 0, 0], dtype=np.uint8), np.array([b_ex_max, 255, 255], dtype=np.uint8))
            bg_mask = cv2.bitwise_and(bg_mask, cv2.bitwise_not(b_ex_mask))

        flake_mask = cv2.morphologyEx(flake_mask, cv2.MORPH_OPEN, morph_kernel, iterations=1)
        flake_mask = cv2.morphologyEx(flake_mask, cv2.MORPH_CLOSE, morph_kernel, iterations=1)
        bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_OPEN, morph_kernel, iterations=1)
        bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_CLOSE, morph_kernel, iterations=1)
        
        raw_obstacles = cv2.bitwise_not(bg_mask)
        obs_contours, _ = cv2.findContours(raw_obstacles, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        obstacles_mask = np.zeros_like(raw_obstacles)
        ignore_obs_area_px = ignore_debris_um2 * (px_per_um ** 2)

        for obs_cnt in obs_contours:
            if cv2.contourArea(obs_cnt) > ignore_obs_area_px:
                cv2.drawContours(obstacles_mask, [obs_cnt], -1, 255, -1)
        
        contours, _ = cv2.findContours(flake_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        min_pixels = min_area * (px_per_um ** 2)
        max_pixels = max_area * (px_per_um ** 2)
        
        img_result = img.copy()
        debug_img = img.copy() if save_debug else None
        
        flake_count = 0
        csv_rows = []

        for cnt in contours:
            area_pixels = cv2.contourArea(cnt)
            if not (min_pixels <= area_pixels <= max_pixels): continue

            (_, _), (rect_w, rect_h), _ = cv2.minAreaRect(cnt)
            short_side = max(min(rect_w, rect_h), 1e-6)
            if max(rect_w, rect_h) / short_side > max_aspect_ratio: continue

            blob_fill = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.drawContours(blob_fill, [cnt], -1, 255, -1)
            
            if strict_color_filter:
                e_px = max(1, int(color_erosion_um * px_per_um))
                e_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (e_px, e_px))
                eroded_blob = cv2.erode(blob_fill, e_kernel, iterations=1)
                if cv2.countNonZero(eroded_blob) == 0: eroded_blob = blob_fill

                pixels_hsv = hsv[eroded_blob == 255]
                m_h = calculate_circular_hue_mean(pixels_hsv[:, 0])
                m_s, m_v = pixels_hsv[:, 1].mean(), pixels_hsv[:, 2].mean()
                
                f_low, f_up = flake_lower.astype(int), flake_upper.astype(int)
                
                if f_low[0] <= f_up[0]: h_valid = (f_low[0] - strict_color_tol <= m_h <= f_up[0] + strict_color_tol)
                else: h_valid = (m_h >= f_low[0] - strict_color_tol) or (m_h <= f_up[0] + strict_color_tol)
                s_valid = (f_low[1] - strict_color_tol <= m_s <= f_up[1] + strict_color_tol)
                v_valid = (f_low[2] - strict_color_tol <= m_v <= f_up[2] + strict_color_tol)
                
                is_excluded = False
                if f_ex_max > 0 or f_ex_min > 0:
                    if f_ex_min <= f_ex_max: is_excluded = (f_ex_min <= m_h <= f_ex_max)
                    else: is_excluded = (m_h >= f_ex_min) or (m_h <= f_ex_max)
                
                if not (h_valid and s_valid and v_valid) or is_excluded: continue

            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            else:
                (cx, cy), _ = cv2.minEnclosingCircle(cnt)
                cx, cy = int(cx), int(cy)
            
            center = (cx, cy)
            _, radius = cv2.minEnclosingCircle(cnt)
            
            halo_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            clearance_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            
            if clr_method == "circle":
                cv2.circle(halo_mask, center, int(radius + halo_buffer_px), 255, -1)
                cv2.circle(clearance_mask, center, int(radius + clearance_px + halo_buffer_px), 255, -1)
            else:
                halo_mask = get_dilated_mask(blob_fill, halo_buffer_px)
                clearance_mask = get_dilated_mask(blob_fill, clearance_px + halo_buffer_px)
                
            ring_mask = cv2.bitwise_and(clearance_mask, cv2.bitwise_not(halo_mask))
            violation = cv2.bitwise_and(ring_mask, obstacles_mask)
            violation_area = cv2.countNonZero(violation)
            noise_tolerance_px = int(cv2.countNonZero(ring_mask) * noise_tol_percent) 
            
            if violation_area > noise_tolerance_px:
                if save_debug:
                    ring_cnts, _ = cv2.findContours(ring_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(debug_img, ring_cnts, -1, (0, 0, 255), 1) 
                    cv2.drawContours(debug_img, [cnt], -1, (0, 165, 255), 1) 
                    debug_img[violation > 0] = [255, 0, 255] 
                    cv2.putText(debug_img, "REJECT", (center[0]-20, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                continue
                
            flake_count += 1
            area_um2 = area_pixels / (px_per_um ** 2)
            radius_um = radius / px_per_um
            blob_bgr = img[blob_fill == 255].mean(axis=0)
            
            drawn_radius_px = int(radius * (circle_scale / 100.0))
            if show_circle:
                cv2.circle(img_result, center, max(drawn_radius_px, 3),
                           (0, 0, 255), line_thickness)
            if show_label:
                cv2.putText(img_result, f"#{flake_count} ({area_um2:.1f})",
                            (center[0] + drawn_radius_px, center[1] - drawn_radius_px),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255),
                            max(1, line_thickness - 1))

            if show_clearance:
                if clr_method == "circle":
                    cv2.circle(img_result, center,
                               int(radius + clearance_px + halo_buffer_px),
                               (0, 255, 255), 1)
                else:
                    clr_cnts, _ = cv2.findContours(clearance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(img_result, clr_cnts, -1, (0, 255, 255), 1)
                        
            if save_debug:
                cv2.circle(debug_img, center, max(drawn_radius_px, 3), (255, 0, 0), 2)
                cv2.putText(debug_img, "PASS", (center[0]-15, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            csv_rows.append([filename, f"Flake_{flake_count}", f"{area_um2:.3f}", center[0], center[1], f"{radius_um:.3f}",
                              f"{blob_bgr[0]:.0f}", f"{blob_bgr[1]:.0f}", f"{blob_bgr[2]:.0f}"])

        base_name, _ = os.path.splitext(filename)
        
        if flake_count > 0:
            out_name = f"{base_name}{output_suffix}.jpg"
            out_path = os.path.join(out_dir_path, out_name)
            if out_path.lower().endswith('.jpg') or out_path.lower().endswith('.jpeg'):
                cv2.imwrite(out_path, img_result, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
            else:
                cv2.imwrite(out_path, img_result)
                
        if save_debug:
            debug_name = f"{base_name}_debug.jpg"
            debug_path = os.path.join(out_dir_path, debug_name)
            cv2.imwrite(debug_path, debug_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        return (filename, True, flake_count, csv_rows)
        
    except Exception as e:
        full_traceback = traceback.format_exc()
        return (filename, False, 0, full_traceback)
        
    finally:
        cv2.setNumThreads(orig_threads)

# =====================================================================
# OpenCV Interactive UI Functions (With Zoom/Pan)
# =====================================================================
def interactive_calibrate(image_path):
    img = cv2.imread(image_path)
    if img is None: return None
    window_name = 'Step 1: Calibration (Scroll to Zoom)'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1200, 800)

    pt1, pt2 = None, None
    drawing = False
    zoom_scale = 1.0
    offset_x, offset_y = 0, 0
    is_panning = False
    pan_start_x, pan_start_y = 0, 0
    current_mouse = None

    def draw_view():
        nonlocal offset_x, offset_y
        h, w = img.shape[:2]
        view_w, view_h = int(w / zoom_scale), int(h / zoom_scale)
        offset_x = max(0, min(offset_x, w - view_w))
        offset_y = max(0, min(offset_y, h - view_h))

        temp_img = img.copy()
        thick = max(1, int(2 / zoom_scale))
        if pt1: cv2.circle(temp_img, pt1, max(2, thick*2), (0, 0, 255), -1)
        if pt1 and pt2: cv2.line(temp_img, pt1, pt2, (0, 255, 0), thick)
        elif drawing and pt1 and current_mouse: cv2.line(temp_img, pt1, current_mouse, (0, 0, 255), thick)

        cropped = temp_img[offset_y:offset_y+view_h, offset_x:offset_x+view_w]
        cv2.putText(cropped, "Drag line over scale bar. Press ENTER to finish.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(window_name, cropped)

    def mouse_callback(event, x, y, flags, param):
        nonlocal pt1, pt2, drawing, zoom_scale, offset_x, offset_y, is_panning, pan_start_x, pan_start_y, current_mouse
        h, w = img.shape[:2]
        orig_x, orig_y = int(x + offset_x), int(y + offset_y)
        current_mouse = (orig_x, orig_y)

        if event == cv2.EVENT_MOUSEWHEEL:
            old_zoom = zoom_scale
            if flags > 0: zoom_scale = min(zoom_scale * 1.25, 20.0)
            elif flags < 0: zoom_scale = max(zoom_scale / 1.25, 1.0)
            if zoom_scale != old_zoom:
                offset_x = int(orig_x - (x / (w / old_zoom)) * (w / zoom_scale))
                offset_y = int(orig_y - (y / (h / old_zoom)) * (h / zoom_scale))
                draw_view()
        elif event == cv2.EVENT_MBUTTONDOWN:
            is_panning, pan_start_x, pan_start_y = True, x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if is_panning:
                offset_x -= (x - pan_start_x)
                offset_y -= (y - pan_start_y)
                pan_start_x, pan_start_y = x, y
            draw_view()
        elif event == cv2.EVENT_MBUTTONUP: is_panning = False
        elif event == cv2.EVENT_LBUTTONDOWN:
            drawing, pt1, pt2 = True, (orig_x, orig_y), None
            draw_view()
        elif event == cv2.EVENT_LBUTTONUP:
            drawing, pt2 = False, (orig_x, orig_y)
            draw_view()

    cv2.setMouseCallback(window_name, mouse_callback)
    draw_view()
    while True:
        if cv2.waitKey(10) & 0xFF in [13, 10]: break
    cv2.destroyAllWindows()
    cv2.waitKey(1) 
    if pt1 and pt2: return math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
    return None

def interactive_color_picker(image_path, title):
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    window_name = title
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1600, 700)

    clicked_hsv = []
    lower_bound = np.array([0, 0, 0], dtype=np.uint8)
    upper_bound = np.array([179, 255, 255], dtype=np.uint8)
    stacked = np.hstack((img, img))
    zoom_scale = 1.0
    offset_x, offset_y = 0, 0
    is_panning = False
    pan_start_x, pan_start_y = 0, 0
    is_initialized = False

    def get_gradient_bar(h_min, h_max, ex_min, ex_max, width, height=50):
        bar_hsv = np.zeros((height, width, 3), dtype=np.uint8)
        if not clicked_hsv:
            bar_bgr = np.full((height, width, 3), (50, 50, 50), dtype=np.uint8)
            cv2.putText(bar_bgr, "Click to start", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            return bar_bgr

        for x in range(width):
            ratio = x / width
            h_val = int(h_min + (h_max - h_min) * ratio)
            is_excluded = (ex_min <= h_val <= ex_max) if ex_min <= ex_max else (h_val >= ex_min or h_val <= ex_max)
            if is_excluded: bar_hsv[:, x] = [0, 200, 50] 
            else: bar_hsv[:, x] = [h_val, 255, 255] 
                
        bar_bgr = cv2.cvtColor(bar_hsv, cv2.COLOR_HSV2BGR)
        cv2.rectangle(bar_bgr, (0, 0), (width-1, height-1), (255, 255, 255), 2)
        cv2.putText(bar_bgr, f"Hue: {h_min}-{h_max}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(bar_bgr, f"Hue: {h_min}-{h_max}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return bar_bgr

    def draw_view():
        nonlocal offset_x, offset_y
        h, w = stacked.shape[:2]
        view_w, view_h = int(w / zoom_scale), int(h / zoom_scale)
        offset_x = max(0, min(offset_x, w - view_w))
        offset_y = max(0, min(offset_y, h - view_h))

        cropped = stacked[offset_y:offset_y+view_h, offset_x:offset_x+view_w].copy()
        ex_min = cv2.getTrackbarPos('Exclude H Min', window_name)
        ex_max = cv2.getTrackbarPos('Exclude H Max', window_name)
        
        color_bar = get_gradient_bar(lower_bound[0], upper_bound[0], ex_min, ex_max, int(cropped.shape[1] * 0.4), 60)
        y_start = cropped.shape[0] - 60 - 20
        x_start = cropped.shape[1] - int(cropped.shape[1] * 0.4) - 20
        cropped[y_start:y_start+60, x_start:x_start+int(cropped.shape[1] * 0.4)] = color_bar
        
        cv2.putText(cropped, f"Points: {len(clicked_hsv)} | Exclude: {ex_min}-{ex_max}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(window_name, cropped)

    def update_mask(*args):
        nonlocal lower_bound, upper_bound, stacked
        if not is_initialized: return
        tol = cv2.getTrackbarPos('Tolerance', window_name)
        ex_min = cv2.getTrackbarPos('Exclude H Min', window_name)
        ex_max = cv2.getTrackbarPos('Exclude H Max', window_name)

        if not clicked_hsv:
            result = np.zeros_like(img)
        else:
            min_h, max_h = int(min(p[0] for p in clicked_hsv)), int(max(p[0] for p in clicked_hsv))
            min_s, max_s = int(min(p[1] for p in clicked_hsv)), int(max(p[1] for p in clicked_hsv))
            min_v, max_v = int(min(p[2] for p in clicked_hsv)), int(max(p[2] for p in clicked_hsv))

            lower_bound = np.array([max(0, min_h - tol), max(0, min_s - tol), max(0, min_v - tol)], dtype=np.uint8)
            upper_bound = np.array([min(179, max_h + tol), min(255, max_s + tol), min(255, max_v + tol)], dtype=np.uint8)

            mask = get_wrapped_hsv_mask(hsv, lower_bound, upper_bound)
            if ex_max > 0 or ex_min > 0:
                exclude_mask = get_wrapped_hsv_mask(hsv, np.array([ex_min, 0, 0], dtype=np.uint8), np.array([ex_max, 255, 255], dtype=np.uint8))
                mask = cv2.bitwise_and(mask, cv2.bitwise_not(exclude_mask))
            result = cv2.bitwise_and(img, img, mask=mask)
        stacked = np.hstack((img, result))
        draw_view()

    def mouse_callback(event, x, y, flags, param):
        nonlocal zoom_scale, offset_x, offset_y, is_panning, pan_start_x, pan_start_y
        h, w = stacked.shape[:2]
        orig_x, orig_y = int(x + offset_x), int(y + offset_y)

        if event == cv2.EVENT_MOUSEWHEEL:
            old_zoom = zoom_scale
            if flags > 0: zoom_scale = min(zoom_scale * 1.25, 20.0)
            elif flags < 0: zoom_scale = max(zoom_scale / 1.25, 1.0)
            if zoom_scale != old_zoom:
                offset_x = int(orig_x - (x / (w / old_zoom)) * (w / zoom_scale))
                offset_y = int(orig_y - (y / (h / old_zoom)) * (h / zoom_scale))
                draw_view()
        elif event == cv2.EVENT_MBUTTONDOWN: is_panning, pan_start_x, pan_start_y = True, x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if is_panning:
                offset_x -= (x - pan_start_x)
                offset_y -= (y - pan_start_y)
                pan_start_x, pan_start_y = x, y
            draw_view()
        elif event == cv2.EVENT_MBUTTONUP: is_panning = False
        elif event == cv2.EVENT_LBUTTONDOWN:
            if orig_x < img.shape[1] and 0 <= orig_y < hsv.shape[0]: clicked_hsv.append(hsv[orig_y, orig_x])
            elif img.shape[1] <= orig_x < stacked.shape[1] and 0 <= orig_y < hsv.shape[0]: clicked_hsv.append(hsv[orig_y, orig_x - img.shape[1]])
            update_mask()
        elif event == cv2.EVENT_RBUTTONDOWN:
            clicked_hsv.clear()
            update_mask()

    cv2.setMouseCallback(window_name, mouse_callback)
    cv2.createTrackbar('Tolerance', window_name, 20, 100, lambda x: update_mask())
    cv2.createTrackbar('Exclude H Min', window_name, 0, 179, lambda x: update_mask())
    cv2.createTrackbar('Exclude H Max', window_name, 0, 179, lambda x: update_mask())
    is_initialized = True
    update_mask()
    while True:
        if cv2.waitKey(10) & 0xFF in [13, 10]: break
    ex_bounds = [cv2.getTrackbarPos('Exclude H Min', window_name), cv2.getTrackbarPos('Exclude H Max', window_name)]
    cv2.destroyAllWindows()
    cv2.waitKey(1) 
    return lower_bound, upper_bound, ex_bounds

# =====================================================================
# GUI Class: Tab 1 - Flake Analyzer
# =====================================================================
class FlakeAnalyzerTab:
    def __init__(self, parent_frame, root):
        self.container = parent_frame
        self.root = root
        
        self.app_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = os.path.join(self.app_dir, "config.json")
        self.input_dir = self.app_dir
        
        self.sample_image = None
        self.px_per_um = None
        self.flake_bounds = None
        self.bg_bounds = None
        
        self.config = self.load_default_config()
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    deep_merge_config(self.config, json.load(f))
            except Exception as e:
                print(f"Failed to load config: {e}")
                
        self.build_ui()
        self.update_ui_from_config()
        self.refresh_sample_image()

    def load_default_config(self):
        return {
            "system_settings": {"use_multiprocessing": True, "max_workers": max(1, multiprocessing.cpu_count() - 2)},
            "file_io": {"input_pattern": "*.jpg", "output_folder": "results", "output_suffix": "_analyzed", "jpeg_quality": 100, "save_debug_images": False},
            "calibration": {"px_per_um": None},
            "flake_target": {"min_area_um2": 2.0, "max_area_um2": 200.0, "max_aspect_ratio": 15.0, "hsv_lower": None, "hsv_upper": None, "exclude_h_range": [0,0]},
            "background": {"hsv_lower": None, "hsv_upper": None, "exclude_h_range": [0,0]},
            "isolation_rules": {"clearance_method": "dilation", "clearance_radius_um": 5.0, "halo_buffer_um": 0.5, "ignore_micro_debris_um2": 0.5, "noise_tolerance_percent": 5.0},
            "advanced_filters": {"strict_blob_color_verification": True, "strict_color_tolerance": 10, "morphology_kernel_um": 0.5, "color_check_erosion_um": 0.2},
            "display": {"circle_scale_percent": 100.0, "show_circle": True, "show_label": True, "show_clearance": True, "line_thickness": 3, "font_scale": 0.5}
        }

    def refresh_sample_image(self):
        input_pat = self.config["file_io"].get("input_pattern", "*.jpg")
        out_suf = self.config["file_io"].get("output_suffix", "_analyzed")
        search_path = os.path.join(self.input_dir, input_pat)
        imgs = [f for f in glob.glob(search_path) if out_suf not in f]
        if imgs: self.sample_image = imgs[0]
        else: self.sample_image = None

    def build_ui(self):
        main_frame = ttk.Frame(self.container, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Button(top_frame, text="Load Config", command=self.load_config_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(top_frame, text="Save Config", command=self.save_config_file).pack(side=tk.LEFT, padx=2)
        self.lbl_preset = ttk.Label(top_frame, text="Preset: Default", foreground="gray")
        self.lbl_preset.pack(side=tk.LEFT, padx=10)

        dir_frame = ttk.LabelFrame(main_frame, text="Directories")
        dir_frame.pack(fill=tk.X, pady=5)
        self.var_in_dir = tk.StringVar(value=self.input_dir)
        ttk.Label(dir_frame, text="Input Folder:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(dir_frame, textvariable=self.var_in_dir, width=40, state='readonly').grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(dir_frame, text="Browse", command=self.browse_input_dir).grid(row=0, column=2, padx=5, pady=2)
        self.var_out_dir = tk.StringVar(value=self.config["file_io"].get("output_folder", "results"))
        ttk.Label(dir_frame, text="Output Folder:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(dir_frame, textvariable=self.var_out_dir, width=40).grid(row=1, column=1, padx=5, pady=2)

        sys_frame = ttk.LabelFrame(main_frame, text="System & Debug")
        sys_frame.pack(fill=tk.X, pady=5)
        self.var_use_mp = tk.BooleanVar(value=self.config.get("system_settings", {}).get("use_multiprocessing", True))
        ttk.Checkbutton(sys_frame, text="Multiprocessing", variable=self.var_use_mp).grid(row=0, column=0, padx=5, pady=2)
        ttk.Label(sys_frame, text="Cores:").grid(row=0, column=1, padx=5, pady=2, sticky=tk.E)
        max_cpus = multiprocessing.cpu_count()
        self.var_cores = tk.IntVar(value=self.config.get("system_settings", {}).get("max_workers", max(1, max_cpus - 2)))
        ttk.Spinbox(sys_frame, from_=1, to=max_cpus, textvariable=self.var_cores, width=5).grid(row=0, column=2, padx=5, pady=2)
        self.var_debug = tk.BooleanVar(value=self.config["file_io"].get("save_debug_images", False))
        ttk.Checkbutton(sys_frame, text="Save Debug Images", variable=self.var_debug).grid(row=0, column=3, padx=10, pady=2)

        set_frame = ttk.LabelFrame(main_frame, text="Analysis Parameters")
        set_frame.pack(fill=tk.X, pady=5)
        self.vars = {}

        def add_entry(parent, label_text, key, dict_ref, row, col, options=None):
            ttk.Label(parent, text=label_text).grid(row=row, column=col*2, sticky=tk.W, padx=5, pady=2)
            var = tk.StringVar()
            if options:
                cb = ttk.Combobox(parent, textvariable=var, values=options, width=12, state="readonly")
                cb.grid(row=row, column=col*2+1, padx=5, pady=2)
            else:
                ttk.Entry(parent, textvariable=var, width=14).grid(row=row, column=col*2+1, padx=5, pady=2)
            self.vars[key] = (var, dict_ref)

        add_entry(set_frame, "Min Area (um2):", "min_area", ("flake_target", "min_area_um2"), 0, 0)
        add_entry(set_frame, "Max Area (um2):", "max_area", ("flake_target", "max_area_um2"), 0, 1)
        add_entry(set_frame, "Max Aspect Ratio:", "max_aspect", ("flake_target", "max_aspect_ratio"), 1, 0)
        add_entry(set_frame, "Circle Scale (%):", "circle_scale", ("display", "circle_scale_percent"), 1, 1)
        add_entry(set_frame, "Clearance (um):", "clearance", ("isolation_rules", "clearance_radius_um"), 2, 0)
        add_entry(set_frame, "Ignore Debris (um2):", "ignore_debris", ("isolation_rules", "ignore_micro_debris_um2"), 2, 1)
        add_entry(set_frame, "Strict Tol (0-255):", "strict_tol", ("advanced_filters", "strict_color_tolerance"), 3, 0)
        add_entry(set_frame, "Kernel Size (um):", "morph_kernel", ("advanced_filters", "morphology_kernel_um"), 3, 1)
        add_entry(set_frame, "Color Erosion (um):", "color_erosion", ("advanced_filters", "color_check_erosion_um"), 4, 0)
        add_entry(set_frame, "Strict Check (True/False):", "strict_bool", ("advanced_filters", "strict_blob_color_verification"), 4, 1)
        add_entry(set_frame, "Output Suffix:", "out_suffix", ("file_io", "output_suffix"), 5, 0)
        add_entry(set_frame, "JPEG Quality (0-100):", "jpg_qual", ("file_io", "jpeg_quality"), 5, 1)
        add_entry(set_frame, "Clearance Method:", "clr_method", ("isolation_rules", "clearance_method"), 6, 0, ["circle", "dilation"])
        add_entry(set_frame, "Halo Buffer (um):", "halo_buffer", ("isolation_rules", "halo_buffer_um"), 6, 1)
        add_entry(set_frame, "Line Thickness (px):", "line_thickness", ("display", "line_thickness"), 7, 0)
        add_entry(set_frame, "Font Scale:", "font_scale", ("display", "font_scale"), 7, 1)

        # Overlay display toggles: control what is drawn on the _analyzed image.
        # Turn these off to export a clean image for placement work in Raith.
        disp_frame = ttk.LabelFrame(main_frame, text="Overlay Display")
        disp_frame.pack(fill=tk.X, pady=5)
        self.var_show_circle = tk.BooleanVar(value=self.config["display"].get("show_circle", True))
        self.var_show_label = tk.BooleanVar(value=self.config["display"].get("show_label", True))
        self.var_show_clearance = tk.BooleanVar(value=self.config["display"].get("show_clearance", True))
        ttk.Checkbutton(disp_frame, text="Show flake circle", variable=self.var_show_circle).grid(row=0, column=0, sticky=tk.W, padx=10, pady=2)
        ttk.Checkbutton(disp_frame, text="Show number/area label", variable=self.var_show_label).grid(row=0, column=1, sticky=tk.W, padx=10, pady=2)
        ttk.Checkbutton(disp_frame, text="Show clearance ring", variable=self.var_show_clearance).grid(row=0, column=2, sticky=tk.W, padx=10, pady=2)

        act_frame = ttk.LabelFrame(main_frame, text="Interactive Setup and Execution")
        act_frame.pack(fill=tk.X, pady=5)
        
        self.lbl_cal = ttk.Label(act_frame, text="Calibration: Not Set", foreground="red")
        self.lbl_cal.pack(pady=2)
        ttk.Button(act_frame, text="1. Calibrate Scale", command=self.do_calibrate).pack(fill=tk.X, pady=2)
        self.lbl_col = ttk.Label(act_frame, text="Color Bounds: Not Set", foreground="red")
        self.lbl_col.pack(pady=2)
        ttk.Button(act_frame, text="2. Pick Target Flake Color", command=lambda: self.do_color("flake")).pack(fill=tk.X, pady=2)
        ttk.Button(act_frame, text="3. Pick Background Color", command=lambda: self.do_color("bg")).pack(fill=tk.X, pady=2)
        self.btn_run = ttk.Button(act_frame, text="4. START BATCH PROCESSING", command=self.run_batch_thread)
        self.btn_run.pack(fill=tk.X, pady=5)

        log_frame = ttk.LabelFrame(main_frame, text="Execution Log")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.log_text = tk.Text(log_frame, height=8, state='disabled', bg='#f4f4f4')
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        prog_frame = ttk.Frame(main_frame)
        prog_frame.pack(fill=tk.X, pady=2)
        self.lbl_status = ttk.Label(prog_frame, text="Ready.")
        self.lbl_status.pack(side=tk.TOP, anchor=tk.W)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(prog_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=2)

    def log(self, message):
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')

    def update_ui_from_config(self):
        for key, (var, (dict_key, sub_key)) in self.vars.items():
            val = self.config[dict_key].get(sub_key, "")
            var.set(str(val))
            
        if "system_settings" in self.config:
            self.var_use_mp.set(self.config["system_settings"].get("use_multiprocessing", True))
            self.var_cores.set(self.config["system_settings"].get("max_workers", max(1, multiprocessing.cpu_count() - 2)))
            
        self.var_out_dir.set(self.config["file_io"].get("output_folder", "results"))
        self.var_debug.set(self.config["file_io"].get("save_debug_images", False))
        self.var_show_circle.set(self.config["display"].get("show_circle", True))
        self.var_show_label.set(self.config["display"].get("show_label", True))
        self.var_show_clearance.set(self.config["display"].get("show_clearance", True))
            
        self.px_per_um = self.config["calibration"].get("px_per_um")
        if self.px_per_um: self.lbl_cal.config(text=f"Calibration: {self.px_per_um:.2f} px/um", foreground="blue")
        else: self.lbl_cal.config(text="Calibration: Not Set", foreground="red")
            
        self.update_color_label_state()

    def update_color_label_state(self):
        has_f = self.config["flake_target"].get("hsv_lower") is not None
        has_b = self.config["background"].get("hsv_lower") is not None
        
        if has_f and has_b:
            self.flake_bounds = (np.array(self.config["flake_target"]["hsv_lower"]), np.array(self.config["flake_target"]["hsv_upper"]))
            self.bg_bounds = (np.array(self.config["background"]["hsv_lower"]), np.array(self.config["background"]["hsv_upper"]))
            self.lbl_col.config(text="Color Bounds: Both Targets Saved", foreground="blue")
        elif has_f:
            self.flake_bounds = (np.array(self.config["flake_target"]["hsv_lower"]), np.array(self.config["flake_target"]["hsv_upper"]))
            self.bg_bounds = None
            self.lbl_col.config(text="Color Bounds: Flake Set, Background Missing", foreground="orange")
        elif has_b:
            self.flake_bounds = None
            self.bg_bounds = (np.array(self.config["background"]["hsv_lower"]), np.array(self.config["background"]["hsv_upper"]))
            self.lbl_col.config(text="Color Bounds: Background Set, Flake Missing", foreground="orange")
        else:
            self.flake_bounds, self.bg_bounds = None, None
            self.lbl_col.config(text="Color Bounds: Not Set", foreground="red")

    def save_ui_to_config(self):
        try:
            for key, (var, (dict_key, sub_key)) in self.vars.items():
                val = var.get().strip()
                if key in ["out_suffix", "clr_method"]: self.config[dict_key][sub_key] = val
                elif key == "strict_bool": self.config[dict_key][sub_key] = (val.lower() == 'true')
                elif key in ["jpg_qual", "strict_tol"]: self.config[dict_key][sub_key] = int(val)
                else: self.config[dict_key][sub_key] = float(val)
                    
            if "system_settings" not in self.config: self.config["system_settings"] = {}
            self.config["system_settings"]["use_multiprocessing"] = self.var_use_mp.get()
            self.config["system_settings"]["max_workers"] = int(self.var_cores.get())
            self.config["file_io"]["output_folder"] = self.var_out_dir.get().strip()
            self.config["file_io"]["save_debug_images"] = self.var_debug.get()
            self.config["display"]["show_circle"] = self.var_show_circle.get()
            self.config["display"]["show_label"] = self.var_show_label.get()
            self.config["display"]["show_clearance"] = self.var_show_clearance.get()
            return True
        except ValueError:
            messagebox.showerror("Error", "Invalid numeric input in parameters.")
            return False

    def browse_input_dir(self):
        folder = filedialog.askdirectory(title="Select Input Image Folder")
        if folder:
            self.input_dir = folder
            self.var_in_dir.set(self.input_dir)
            self.refresh_sample_image()
            if self.sample_image: 
                messagebox.showinfo("Folder Selected", f"Found images in folder.\nSample ready.")
                self.log(f"Loaded input directory: {self.input_dir}")
            else: 
                messagebox.showwarning("Warning", "No valid JPG images found in the selected folder.")
                self.log("Warning: No valid JPG images found in selected folder.")

    def load_config_file(self):
        file_path = filedialog.askopenfilename(title="Select Preset JSON", filetypes=[("JSON Files", "*.json")])
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    deep_merge_config(self.config, json.load(f))
                self.update_ui_from_config()
                self.lbl_preset.config(text=f"Preset: {os.path.basename(file_path)}", foreground="blue")
                self.refresh_sample_image()
                self.log(f"Loaded configuration: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load config: {str(e)}")
                self.log(f"Error loading config: {str(e)}")

    def save_config_file(self):
        if not self.save_ui_to_config(): return
        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON Files", "*.json")], title="Save Preset As")
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, indent=4)
                self.lbl_preset.config(text=f"Preset: {os.path.basename(file_path)}", foreground="blue")
                messagebox.showinfo("Success", "Configuration saved successfully.")
                self.log(f"Saved configuration to: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save config: {str(e)}")

    def do_calibrate(self):
        if not self.sample_image:
            messagebox.showerror("Error", "No JPG images found in input directory.")
            return
        self.root.withdraw() 
        self.root.update()
        try: dist = interactive_calibrate(self.sample_image)
        finally: self.root.deiconify()
        if dist:
            real = float(simpledialog.askstring("Input", "Enter physical length in um:", parent=self.root) or 0)
            if real > 0:
                self.px_per_um = dist / real
                self.config["calibration"]["px_per_um"] = round(self.px_per_um, 3)
                self.lbl_cal.config(text=f"Calibration: {self.px_per_um:.2f} px/um", foreground="blue")
                self.log(f"Scale calibrated: {self.px_per_um:.2f} px/um")

    def do_color(self, target):
        if not self.sample_image: 
            messagebox.showerror("Error", "No JPG images found in input directory.")
            return
        self.root.withdraw()
        self.root.update()
        try: low, up, ex = interactive_color_picker(self.sample_image, f"Pick {target.upper()} Color")
        finally: self.root.deiconify()
            
        if target == "flake":
            self.flake_bounds = (low, up)
            self.config["flake_target"]["hsv_lower"] = low.tolist()
            self.config["flake_target"]["hsv_upper"] = up.tolist()
            self.config["flake_target"]["exclude_h_range"] = ex
            self.log("Flake target color bounds updated.")
        else:
            self.bg_bounds = (low, up)
            self.config["background"]["hsv_lower"] = low.tolist()
            self.config["background"]["hsv_upper"] = up.tolist()
            self.config["background"]["exclude_h_range"] = ex
            self.log("Background color bounds updated.")
            
        self.update_color_label_state()

    def run_batch_thread(self):
        if not self.px_per_um or not self.flake_bounds or not self.bg_bounds:
            messagebox.showerror("Error", "Please complete Steps 1 to 3 or Load a Config.")
            return
        if not self.save_ui_to_config(): return

        input_pattern = self.config["file_io"]["input_pattern"]
        out_suffix = self.config["file_io"]["output_suffix"]
        search_path = os.path.join(self.input_dir, input_pattern)
        image_files = [f for f in glob.glob(search_path) if out_suffix not in f]
        
        if not image_files:
            messagebox.showinfo("Done", "No new images found to process in the input folder.")
            return

        self.btn_run.config(state=tk.DISABLED)
        self.progress_var.set(0)
        self.lbl_status.config(text="Starting batch process...")
        self.log(f"--- Starting Batch Analysis for {len(image_files)} images ---")
        threading.Thread(target=self._batch_process_worker, args=(image_files,), daemon=True).start()

    def _batch_process_worker(self, image_files):
        out_dir_name = self.config["file_io"]["output_folder"]
        if os.path.isabs(out_dir_name): out_dir_path = out_dir_name
        else: out_dir_path = os.path.join(self.input_dir, out_dir_name)
        os.makedirs(out_dir_path, exist_ok=True)
        
        tasks = [(img_path, out_dir_path, self.px_per_um, self.flake_bounds, self.bg_bounds, self.config) for img_path in image_files]
        total_flakes = 0
        total_images = len(tasks)
        completed_images = 0
        csv_data = [['Filename', 'Flake_ID', 'Area_um2', 'Center_X_px', 'Center_Y_px', 'Isolated_Radius_um', 'Avg_Color_B', 'Avg_Color_G', 'Avg_Color_R']]
        
        use_mp = self.config.get("system_settings", {}).get("use_multiprocessing", True)
        workers = self.config.get("system_settings", {}).get("max_workers", 1)
        
        failed_files = []

        if use_mp and workers > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(process_single_image, task): task for task in tasks}
                for future in concurrent.futures.as_completed(futures):
                    filename, success, count, payload = future.result()
                    if success:
                        total_flakes += count
                        csv_data.extend(payload)
                        self.root.after(0, self.log, f"[OK] {filename} -> Found {count} flake(s)")
                    else:
                        failed_files.append((filename, payload))
                        self.root.after(0, self.log, f"[ERROR] {filename} -> Failed to process")
                    
                    completed_images += 1
                    self.root.after(0, self._update_progress_ui, (completed_images / total_images) * 100, completed_images, total_images)
        else:
            for task in tasks:
                filename, success, count, payload = process_single_image(task)
                if success:
                    total_flakes += count
                    csv_data.extend(payload)
                    self.root.after(0, self.log, f"[OK] {filename} -> Found {count} flake(s)")
                else:
                    failed_files.append((filename, payload))
                    self.root.after(0, self.log, f"[ERROR] {filename} -> Failed to process")
                    
                completed_images += 1
                self.root.after(0, self._update_progress_ui, (completed_images / total_images) * 100, completed_images, total_images)

        csv_path = os.path.join(out_dir_path, 'isolated_flakes_report.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(csv_data)
            
        if failed_files:
            log_path = os.path.join(out_dir_path, 'failed_files_log.txt')
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write("=== 2D Flake Analyzer Pro - Processing Failure Report ===\n\n")
                for fname, err_msg in failed_files:
                    f.write(f"File: {fname}\nReason: {err_msg}\n-----------------------------------------------------\n")
            
        self.root.after(0, self._finish_batch_ui, total_flakes, out_dir_path, len(failed_files))

    def _update_progress_ui(self, val, completed, total):
        self.progress_var.set(val)
        self.lbl_status.config(text=f"Processing... {completed} / {total} images done.")

    def _finish_batch_ui(self, total_flakes, out_dir, failed_count):
        self.progress_var.set(100)
        self.btn_run.config(state=tk.NORMAL)
        
        if failed_count > 0:
            self.lbl_status.config(text=f"Complete with warnings! Found {total_flakes} flakes. ({failed_count} files failed)")
            self.log(f"--- Completed: {total_flakes} flakes found, {failed_count} failures ---")
            messagebox.showwarning("Batch Complete with Warnings", 
                                  f"Batch execution complete.\n- Found {total_flakes} flakes.\n- {failed_count} files failed extraction.\n\nCheck 'failed_files_log.txt' inside the results folder for details.")
        else:
            self.lbl_status.config(text=f"Complete! Found {total_flakes} flakes.")
            self.log(f"--- Completed: {total_flakes} flakes found successfully ---")
            messagebox.showinfo("Success", f"Batch Complete!\n{total_flakes} flakes found.\nResults saved to folder:\n{out_dir}")

# =====================================================================
# GUI Class: Tab 2 - EBL Aligner
# =====================================================================
class EBLAlignerTab:
    def __init__(self, parent_frame, root):
        self.container = parent_frame
        self.root = root
        self.work_dir = os.path.dirname(os.path.abspath(__file__))
        self.build_ui()

    def build_ui(self):
        main_frame = ttk.Frame(self.container, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        dir_frame = ttk.LabelFrame(main_frame, text="Working Directory")
        dir_frame.pack(fill=tk.X, pady=5)
        
        self.var_work_dir = tk.StringVar(value=self.work_dir)
        ttk.Label(dir_frame, text="Target Folder:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(dir_frame, textvariable=self.var_work_dir, width=45, state='readonly').grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(dir_frame, text="Browse", command=self.browse_dir).grid(row=0, column=2, padx=5, pady=2)

        self.var_suffix = tk.StringVar(value="_ShiftN")
        ttk.Label(dir_frame, text="Target Suffix:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Entry(dir_frame, textvariable=self.var_suffix, width=15).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

        # Detection methods: user checks any combination; one output set is written
        # per checked method so results can be compared per-image.
        self.method_vars = {
            "template": tk.BooleanVar(value=False),
            "centroid": tk.BooleanVar(value=True),
            "line": tk.BooleanVar(value=True),
            "median": tk.BooleanVar(value=True),
        }
        ttk.Label(dir_frame, text="Detection Methods:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        method_frame = ttk.Frame(dir_frame)
        method_frame.grid(row=2, column=1, columnspan=2, sticky=tk.W, padx=5, pady=2)
        for m in ["template", "centroid", "line", "median"]:
            ttk.Checkbutton(method_frame, text=m, variable=self.method_vars[m]).pack(side=tk.LEFT, padx=3)

        # Output file organization: A = per-method subfolders, B = method suffix in
        # filenames, C = single shared BMP with per-method .ssc files.
        self.var_file_org = tk.StringVar(value="A")
        ttk.Label(dir_frame, text="File Organization:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        org_frame = ttk.Frame(dir_frame)
        org_frame.grid(row=3, column=1, columnspan=2, sticky=tk.W, padx=5, pady=2)
        for val, lbl in [("A", "A: per-method folders"),
                         ("B", "B: method in filename"),
                         ("C", "C: shared BMP")]:
            ttk.Radiobutton(org_frame, text=lbl, value=val, variable=self.var_file_org).pack(side=tk.LEFT, padx=3)

        self.var_save_debug = tk.BooleanVar(value=True)
        ttk.Checkbutton(dir_frame, text="Save debug overlay (compare all methods)",
                        variable=self.var_save_debug).grid(row=4, column=1, columnspan=2, sticky=tk.W, padx=5, pady=2)

        inst_frame = ttk.LabelFrame(main_frame, text="Instructions")
        inst_frame.pack(fill=tk.X, pady=5)
        inst_text = (
            "1. Select the folder containing alignment files.\n"
            "2. Define the Target Suffix (default is '_ShiftN').\n"
            "3. Select the alignment method. \n (Tip: Usually Median works best but it depends on your images.)\n"
            "4. Click 'Run Alignment Pipeline'.\n"
            "5. An OpenCV window will pop up showing the first image.\n"
            "6. Drag a bounding box around the SMALL crosshair mark.\n"
            "7. Press SPACE or ENTER to confirm the ROI.\n"
            "8. The system will automatically align all images and generate .ssc files."
        )
        ttk.Label(inst_frame, text=inst_text, justify=tk.LEFT).pack(padx=10, pady=10, anchor=tk.W)

        self.btn_run = ttk.Button(main_frame, text="RUN ALIGNMENT PIPELINE", command=self.start_alignment)
        self.btn_run.pack(fill=tk.X, pady=15)

        log_frame = ttk.LabelFrame(main_frame, text="Execution Log")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.log_text = tk.Text(log_frame, height=12, state='disabled', bg='#f4f4f4')
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=2)

    def log(self, message):
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')

    def browse_dir(self):
        folder = filedialog.askdirectory(title="Select Alignment Image Folder")
        if folder:
            self.work_dir = folder
            self.var_work_dir.set(self.work_dir)
            suffix = self.var_suffix.get().strip()
            files = glob.glob(os.path.join(self.work_dir, f'*{suffix}.jpg'))
            self.log(f"Selected directory. Found {len(files)} image(s) ending with '{suffix}.jpg'.")

    def start_alignment(self):
        target_dir = self.var_work_dir.get()
        suffix = self.var_suffix.get().strip()
        
        if not suffix:
            messagebox.showerror("Error", "Please enter a Target Suffix (e.g. _ShiftN).")
            return
            
        image_files = glob.glob(os.path.join(target_dir, f'*{suffix}.jpg'))
        
        if not image_files:
            messagebox.showerror("Error", f"No '*{suffix}.jpg' files found in:\n{target_dir}")
            return

        self.log("="*40)
        self.log(f"Found {len(image_files)} image(s). Starting...")
        self.btn_run.config(state=tk.DISABLED)

        self.root.withdraw()
        first_img = cv2.imread(image_files[0])
        img_gray_full = cv2.cvtColor(first_img, cv2.COLOR_BGR2GRAY)
        
        try:
            roi = cv2.selectROI("Select TARGET Crosshair roughly", first_img, showCrosshair=True, fromCenter=False)
            cv2.destroyAllWindows()
        finally:
            self.root.deiconify()

        if roi == (0, 0, 0, 0):
            self.log("ROI selection cancelled by user.")
            self.btn_run.config(state=tk.NORMAL)
            return

        x, y, w_roi, h_roi = roi
        rough_crop = img_gray_full[y:y+h_roi, x:x+w_roi]
        
        _, thresh = cv2.threshold(rough_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find the cross's true mathematical center via 1D projections (eliminates
        # the offset that asymmetric arm lengths would otherwise introduce).
        proj_x = np.sum(thresh, axis=0)
        proj_y = np.sum(thresh, axis=1)
        
        max_x, max_y = np.max(proj_x), np.max(proj_y)
        
        if max_x > 0 and max_y > 0:
            peak_x_indices = np.where(proj_x > max_x * 0.5)[0]
            peak_y_indices = np.where(proj_y > max_y * 0.5)[0]
            cx = int(np.mean(peak_x_indices))
            cy = int(np.mean(peak_y_indices))
        else:
            # Fallback just in case (original bounding-box logic)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                cx_bound, cy_bound, w_bound, h_bound = cv2.boundingRect(largest_contour)
                cx = cx_bound + w_bound // 2
                cy = cy_bound + h_bound // 2
            else:
                cx, cy = w_roi // 2, h_roi // 2
                
        true_center_x = x + cx
        true_center_y = y + cy
        
        # Set the template size
        box_size = max(w_roi, h_roi) // 2
        top = max(0, true_center_y - box_size)
        bottom = min(img_gray_full.shape[0], true_center_y + box_size)
        left = max(0, true_center_x - box_size)
        right = min(img_gray_full.shape[1], true_center_x + box_size)
        
        template = img_gray_full[top:bottom, left:right]
        tw, th = template.shape[::-1]
        
        # KEY: remember where the true cross intersection sits (in pixels) inside the template image
        template_cx = true_center_x - left
        template_cy = true_center_y - top

        self.log(f"Template created: {tw}x{th} pixels.")

        # Run heavy processing in background thread
        methods = [m for m, v in self.method_vars.items() if v.get()]
        if not methods:
            messagebox.showerror("Error", "At least one detection method must be selected.")
            self.btn_run.config(state=tk.NORMAL)
            return
        # Keep a stable, sensible order regardless of dict iteration.
        order = ["template", "centroid", "line", "median"]
        methods = [m for m in order if m in methods]
        file_org = self.var_file_org.get()
        save_debug_flag = self.var_save_debug.get()
        threading.Thread(target=self._process_align_worker,
                         args=(image_files, target_dir, template, template_cx, template_cy,
                               suffix, methods, file_org, save_debug_flag), daemon=True).start()

    def _process_align_worker(self, image_files, work_dir, template, template_cx, template_cy,
                              suffix, methods, file_org, save_debug_flag):
        total_files = len(image_files)
        
        for idx, img_path in enumerate(image_files):
            filename = os.path.basename(img_path)
            self.root.after(0, self.log, f"Processing: {filename} ...")
            
            try:
                img = cv2.imread(img_path)
                if img is None:
                    raise ValueError("Could not read image file.")
                    
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                h, w = img_gray.shape
                
                # Original marker-detection logic (ignores debris & detects the exact shape)
                def find_mark_in_roi(roi_img, offset_x, offset_y):
                    res = cv2.matchTemplate(roi_img, template, cv2.TM_CCOEFF_NORMED)
                    _, _, _, max_loc = cv2.minMaxLoc(res)
                    rx, ry = max_loc
                    
                    dx, dy = 0.0, 0.0
                    if 0 < rx < res.shape[1] - 1 and 0 < ry < res.shape[0] - 1:
                        num_x = res[ry, rx-1] - res[ry, rx+1]
                        den_x = 2.0 * (res[ry, rx-1] - 2.0 * res[ry, rx] + res[ry, rx+1])
                        if den_x != 0: dx = num_x / den_x
                        
                        num_y = res[ry-1, rx] - res[ry+1, rx]
                        den_y = 2.0 * (res[ry-1, rx] - 2.0 * res[ry, rx] + res[ry+1, rx])
                        if den_y != 0: dy = num_y / den_y
                    
                    peak_x = rx + dx
                    peak_y = ry + dy

                    # Template-match center (baseline): uses the recorded true cross
                    # position inside the template, not the template's geometric middle.
                    tm_x = peak_x + template_cx
                    tm_y = peak_y + template_cy

                    # Two refinements, both seeded from the template-match location.
                    cen = refine_mark_centroid(roi_img, tm_x, tm_y)
                    lin = refine_mark_line_intersection(roi_img, tm_x, tm_y)

                    # Sanity clamp: a refinement is only trusted if it stays within a
                    # few px of the template match (otherwise it latched onto junk).
                    def ok(p):
                        return p is not None and math.hypot(p[0] - tm_x, p[1] - tm_y) <= 40

                    cen_xy = cen if ok(cen) else (tm_x, tm_y)
                    lin_xy = lin if ok(lin) else (tm_x, tm_y)

                    # Median of the three candidates (component-wise). Robust default:
                    # if any single method is an outlier, the median ignores it.
                    med_x = sorted([tm_x, cen_xy[0], lin_xy[0]])[1]
                    med_y = sorted([tm_y, cen_xy[1], lin_xy[1]])[1]
                    med_xy = (med_x, med_y)

                    # 'sel' is informational only now (all candidates are returned and
                    # the worker writes one output per selected method). Default to the
                    # first requested method for any incidental use.
                    primary = methods[0] if methods else "line"
                    if primary == "line":
                        sel = lin_xy
                    elif primary == "centroid":
                        sel = cen_xy
                    elif primary == "median":
                        sel = med_xy
                    else:
                        sel = (tm_x, tm_y)

                    # Return the selected point plus all candidates (ROI-local, for
                    # side-by-side comparison in the debug overlay / log).
                    return {
                        "sel": [sel[0] + offset_x, sel[1] + offset_y],
                        "template": [tm_x + offset_x, tm_y + offset_y],
                        "centroid": [cen_xy[0] + offset_x, cen_xy[1] + offset_y],
                        "line": [lin_xy[0] + offset_x, lin_xy[1] + offset_y],
                        "median": [med_xy[0] + offset_x, med_xy[1] + offset_y],
                    }

                y_cut = int(h * 0.3)
                x_cut = int(w * 0.3)

                roi_LL = img_gray[h - y_cut : h, 0 : x_cut]
                mark_LL = find_mark_in_roi(roi_LL, offset_x=0, offset_y=h - y_cut)

                roi_LR = img_gray[h - y_cut : h, w - x_cut : w]
                mark_LR = find_mark_in_roi(roi_LR, offset_x=w - x_cut, offset_y=h - y_cut)

                roi_UL = img_gray[0 : y_cut, 0 : x_cut]
                mark_UL = find_mark_in_roi(roi_UL, offset_x=0, offset_y=0)

                nominal_uv = get_mark_nominal_uv(filename, suffix)
                base_name, _ = os.path.splitext(filename)

                # Compute an .ssc coordinate set from one triple of detected points.
                # (Same math as before; now parameterized by which detection method
                # supplied the LL/LR/UL pixel positions.)
                def compute_ssc(pLL, pLR, pUL):
                    src_pts = np.float32([pLL, pLR, pUL])
                    dst_pts = np.float32([nominal_uv['LL'], nominal_uv['LR'], nominal_uv['UL']])
                    M = cv2.getAffineTransform(src_pts, dst_pts)
                    ox, oy = np.dot(M, np.array([w / 2.0, h / 2.0, 1.0]))[:2]
                    dpx = math.hypot(pLR[0] - pLL[0], pLR[1] - pLL[1])
                    dpy = math.hypot(pUL[0] - pLL[0], pUL[1] - pLL[1])
                    sx, sy = 400.0 / dpx, 400.0 / dpy
                    wu, hu = w * sx, h * sy
                    ll = ((ox - wu / 2.0) / 1000.0, (oy - hu / 2.0) / 1000.0)
                    ur = ((ox + wu / 2.0) / 1000.0, (oy + hu / 2.0) / 1000.0)
                    orr = (ox / 1000.0, oy / 1000.0)
                    tilt = math.degrees(math.atan2(pLR[1] - pLL[1], pLR[0] - pLL[0]))
                    v1 = np.array([pLR[0] - pLL[0], pLR[1] - pLL[1]])
                    v2 = np.array([pUL[0] - pLL[0], pUL[1] - pLL[1]])
                    perp = 90.0 - math.degrees(math.acos(
                        abs(np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)))
                    return ll, ur, orr, tilt, perp

                def ssc_text(bmp_ref, ll, ur, orr):
                    return (f"[SLOWSCAN]\r\nBitmap={bmp_ref}\r\n"
                            f"LowerLeftUV={ll[0]:.7f},{ll[1]:.7f}\r\n"
                            f"UpperRightUV={ur[0]:.7f},{ur[1]:.7f}\r\n"
                            f"LocalOriginUV={orr[0]:.7f},{orr[1]:.7f}\r\n"
                            f"StageUV=0.000000,0.000000\r\nRotationUV=0.00000\r\n")

                # Disagreement metric (kept for the log/debug comparison).
                def _disp(m):
                    return math.hypot(m["centroid"][0] - m["line"][0],
                                      m["centroid"][1] - m["line"][1])
                cmp_msg = (f"cen-vs-line disp px: LL={_disp(mark_LL):.2f} "
                           f"LR={_disp(mark_LR):.2f} UL={_disp(mark_UL):.2f}")

                # Write one .ssc/.bmp set per selected detection method, organized
                # according to file_org: "A"=per-method subfolder, "B"=method suffix
                # in filename, "C"=shared single BMP + per-method .ssc.
                shared_bmp_written = False
                first_tilt, first_perp = 0.0, 0.0
                for mi, method in enumerate(methods):
                    pLL, pLR, pUL = mark_LL[method], mark_LR[method], mark_UL[method]
                    ll, ur, orr, tilt, perp = compute_ssc(pLL, pLR, pUL)
                    if mi == 0:
                        first_tilt, first_perp = tilt, perp

                    if file_org == "A":
                        # results/<method>/<name>.ssc + .bmp
                        out_sub = os.path.join(work_dir, method)
                        os.makedirs(out_sub, exist_ok=True)
                        bmp_ref = f"{base_name}.bmp"
                        cv2.imwrite(os.path.join(out_sub, bmp_ref), img)
                        with open(os.path.join(out_sub, f"{base_name}.ssc"), 'w') as f:
                            f.write(ssc_text(bmp_ref, ll, ur, orr))
                    elif file_org == "B":
                        # <name>_<method>.ssc + _<method>.bmp in work_dir
                        bmp_ref = f"{base_name}_{method}.bmp"
                        cv2.imwrite(os.path.join(work_dir, bmp_ref), img)
                        with open(os.path.join(work_dir, f"{base_name}_{method}.ssc"), 'w') as f:
                            f.write(ssc_text(bmp_ref, ll, ur, orr))
                    else:  # "C": single shared BMP, per-method .ssc referencing it
                        bmp_ref = f"{base_name}.bmp"
                        if not shared_bmp_written:
                            cv2.imwrite(os.path.join(work_dir, bmp_ref), img)
                            shared_bmp_written = True
                        with open(os.path.join(work_dir, f"{base_name}_{method}.ssc"), 'w') as f:
                            f.write(ssc_text(bmp_ref, ll, ur, orr))

                sample_tilt_deg, perp_dev = first_tilt, first_perp

                # Debug overlay comparing all detection methods per mark, so the
                # best method can be judged per-image on real data.
                # Colors: template=red, centroid=yellow, line=green, median=cyan.
                if save_debug_flag:
                    dbg = img.copy()
                    for m, name in zip([mark_LL, mark_LR, mark_UL], ['LL', 'LR', 'UL']):
                        for key, col in (("template", (0, 0, 255)),
                                         ("centroid", (0, 255, 255)),
                                         ("line", (0, 255, 0)),
                                         ("median", (255, 255, 0))):
                            px, py = int(round(m[key][0])), int(round(m[key][1]))
                            cv2.drawMarker(dbg, (px, py), col, cv2.MARKER_CROSS, 30, 1)
                        cv2.putText(dbg, name, (int(m["line"][0]) + 14, int(m["line"][1]) - 14),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(dbg, f"methods={'+'.join(methods)}  {cmp_msg}",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(dbg, "red=template yellow=centroid green=line cyan=median",
                                (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    dbg_dir = os.path.join(work_dir, 'debug_align')
                    os.makedirs(dbg_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(dbg_dir, f"{base_name}_debug.jpg"), dbg,
                                [int(cv2.IMWRITE_JPEG_QUALITY), 92])

                self.root.after(0, self.log,
                                f"[OK] {filename} -> tilt={sample_tilt_deg:+.4f} deg, "
                                f"perp_dev={perp_dev:+.4f} deg, {cmp_msg}")
                
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, self.log, f"[ERROR] {filename} -> {error_msg}")
                
            self.root.after(0, self._update_progress, (idx + 1) / total_files * 100)

        self.root.after(0, self._finish_alignment)

    def _update_progress(self, val):
        self.progress_var.set(val)

    def _finish_alignment(self):
        self.progress_var.set(100)
        self.log("All processing completed successfully!")
        self.btn_run.config(state=tk.NORMAL)
        messagebox.showinfo("Complete", "EBL Alignment files (.ssc and .bmp) successfully generated.")

# =====================================================================
# Main Application Entry Point
# =====================================================================
class LithoToolkitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Lithography Vision Toolkit (Flake + EBL Align)")
        self.root.geometry("600x900")

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.tab_flake_frame = ttk.Frame(self.notebook)
        self.tab_align_frame = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_flake_frame, text="🔍 Flake Analyzer")
        self.notebook.add(self.tab_align_frame, text="🎯 EBL Aligner")

        self.flake_app = FlakeAnalyzerTab(self.tab_flake_frame, self.root)
        self.align_app = EBLAlignerTab(self.tab_align_frame, self.root)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    root = tk.Tk()
    app = LithoToolkitApp(root)
    root.mainloop()