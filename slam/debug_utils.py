import os
import cv2
import numpy as np
import torch
from typing import List, Dict
from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT


def frame_tensor_to_bgr_image(img_tensor: torch.Tensor) -> np.ndarray:
    if img_tensor is None:
        raise ValueError("Empty image tensor in frame buffer.")
    img = img_tensor.detach().cpu()
    if img.ndim == 4:
        img = img[0]
    img_norm = IMAGE_NORMALIZATION_DICT["dinov2"]
    mean = torch.tensor(img_norm.mean).view(3, 1, 1)
    std = torch.tensor(img_norm.std).view(3, 1, 1)
    img = (img * std + mean).clamp(0.0, 1.0)
    img = (img * 255.0).byte().permute(1, 2, 0).numpy()  # HWC, RGB
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return bgr


def save_localization_start_end_debug(database_path: str, reference_db: List[Dict], frame_buffer: List[torch.Tensor], start_idx: int, end_idx: int):
    if not reference_db or 'image_path' not in reference_db[0] or 'image_path' not in reference_db[-1]:
        return
    out_dir = os.path.join(os.path.dirname(database_path), "debug_matches")
    os.makedirs(out_dir, exist_ok=True)

    q_start_bgr = frame_tensor_to_bgr_image(frame_buffer[start_idx])
    q_end_bgr = frame_tensor_to_bgr_image(frame_buffer[end_idx])

    r_start_bgr = cv2.imread(reference_db[0]['image_path'])
    r_end_bgr = cv2.imread(reference_db[-1]['image_path'])
    if r_start_bgr is None or r_end_bgr is None:
        return

    def _stack_h(img_left, img_right, target_h=240):
        def _resize_h(img, h):
            ih, iw = img.shape[:2]
            w = int(iw * (h / ih))
            return cv2.resize(img, (w, h))
        l = _resize_h(img_left, target_h)
        r = _resize_h(img_right, target_h)
        return cv2.hconcat([l, r])

    start_pair = _stack_h(q_start_bgr, r_start_bgr)
    end_pair = _stack_h(q_end_bgr, r_end_bgr)

    start_path = os.path.join(out_dir, f"localization_start_pair_q{start_idx:06d}.jpg")
    end_path = os.path.join(out_dir, f"localization_end_pair_q{end_idx:06d}.jpg")
    cv2.imwrite(start_path, start_pair)
    cv2.imwrite(end_path, end_pair)


def save_query_db_match_debug(database_path: str, frame_buffer: List[torch.Tensor], chunk_idx: int, query_image_idx_in_chunk: int, query_image_identifier: str, db_image_path: str):
    try:
        import os
        base = os.path.basename(query_image_identifier)
        name, _sep, _ext = base.partition('.')
        digits = ''.join([c for c in name if c.isdigit()])
        frame_idx = int(digits)

        query_bgr = frame_tensor_to_bgr_image(frame_buffer[frame_idx])
        ref_bgr = cv2.imread(db_image_path)
        if ref_bgr is None:
            return
        target_h = 240
        def _resize_h(img, h):
            ih, iw = img.shape[:2]
            w = int(iw * (h / ih))
            return cv2.resize(img, (w, h))
        q_res = _resize_h(query_bgr, target_h)
        r_res = _resize_h(ref_bgr, target_h)
        stacked = cv2.hconcat([q_res, r_res])

        out_dir = os.path.join(os.path.dirname(database_path), "debug_matches")
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, f"chunk_{chunk_idx:04d}_match_q{frame_idx:06d}_idx{query_image_idx_in_chunk:03d}.jpg")
        cv2.imwrite(save_path, stacked)
    except Exception:
        pass


def create_debug_match_image(chunk_idx, query_image_idx, query_image_path, reference_image_path, output_dir="debug_matches"):
    """Legacy helper kept for compatibility; writes side-by-side of two file paths."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    query_img = cv2.imread(query_image_path)
    if query_img is None:
        return

    h, w, _ = query_img.shape
    target_h = 240
    target_w = int(w * (target_h / h))
    query_img_resized = cv2.resize(query_img, (target_w, target_h))

    match_img = cv2.imread(reference_image_path)
    if match_img is None:
        return
    h, w, _ = match_img.shape
    target_w = int(w * (target_h / h))
    match_img_resized = cv2.resize(match_img, (target_w, target_h))

    combined_image = cv2.hconcat([query_img_resized, match_img_resized])
    save_path = os.path.join(output_dir, f"chunk_{chunk_idx}_matches_{query_image_idx}.jpg")
    cv2.imwrite(save_path, combined_image)

