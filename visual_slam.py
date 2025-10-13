import torch
import os
import shutil
import time
import cv2
import numpy as np
from typing import List, Dict, Tuple

from mapanything.models import MapAnything
from mapanything.utils.image import load_images, find_closest_aspect_ratio
from slam_config import SLAMConfig
from mapanything.utils.image import rgb
import open3d as o3d
from PIL import Image
from mapanything.utils.inference import preprocess_input_views_for_inference, postprocess_model_outputs_for_inference
from utils.telemetry_converter import TelemetryImporter
from utils.undistortion import create_undistortion_maps_from_file
from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT
import torchvision.transforms as tvf
from scipy.spatial.transform import Rotation as RS
import glob
from slam.debug_utils import save_query_db_match_debug
from slam.trajectory_utils import save_query_trajectory_and_normals
from slam.alignment_utils import align_reconstruction_to_gravity, align_reconstruction_to_gps_direction
from slam.geo_utils import ecef_to_ned
from utils.math import scale_intrinsics
from slam.plane_utils import estimate_ground_plane
from slam.chunk_utils import (
    save_chunk_with_interpolation,
    save_chunks_for_visualization, 
    save_debug_chunk_visualization
)
from slam.matcher import Matcher

MS_TO_NS = 1_000_000

class VisualSLAM:
    def __init__(self, config: SLAMConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._setup_paths()
        self.map_anything = None
        self.intrinsics = None
        self.telemetry = None
        self.frame_buffer = []
        self.frametimes_ns = []
        self.undistortion_maps = None
        
        self._initialize_models()

        if self.config.video_path:
            self._cache_frames()

    def _setup_paths(self):
        """Creates output paths relative to the video path (video-only)."""
        if not self.config.video_path:
            raise ValueError("Video-only mode: 'video_path' must be provided.")
        results_dir_base = self.config.video_path
        results_dir = os.path.join(os.path.dirname(results_dir_base), "_slam_results")
        
        if self.config.output_dir is None or self.config.database_path is None:
            os.makedirs(results_dir, exist_ok=True)
            
            if self.config.output_dir is None:
                self.config.output_dir = os.path.join(results_dir, "chunks")
            
            if self.config.database_path is None:
                db_name = os.path.basename(os.path.normpath(results_dir_base)) + "_db.pt"
                self.config.database_path = os.path.join(results_dir, db_name)
        
        # Setup database path, making it absolute if it's relative
        if not os.path.isabs(self.config.database_path):
            if self.config.database_path is None:
                self.config.database_path = os.path.join(results_dir, "slam_database.pt")
            else:
                db_name = os.path.basename(self.config.database_path)
                self.config.database_path = os.path.join(results_dir, db_name)
        
        # Clean existing output directory to prevent conflicts with old chunks
        if os.path.exists(self.config.output_dir):
            print(f"Removing existing output directory: {self.config.output_dir}")
            shutil.rmtree(self.config.output_dir)
        
        # Create fresh output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Ensure the directory for the database exists
        db_dir = os.path.dirname(self.config.database_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        
        # Clean existing database-related files to prevent conflicts
        if os.path.exists(self.config.database_path):
            print(f"Removing existing database file: {self.config.database_path}")
            os.remove(self.config.database_path)
        
        # Clean related output files in the database directory
        if db_dir and os.path.exists(db_dir):
            # Remove reference images directory if it exists
            ref_images_dir = os.path.join(db_dir, "reference_images")
            if os.path.exists(ref_images_dir):
                print(f"Removing existing reference images: {ref_images_dir}")
                shutil.rmtree(ref_images_dir)
            
            # Remove debug directories if they exist
            debug_keypoint_dir = os.path.join(db_dir, "debug_keypoint_matches")
            if os.path.exists(debug_keypoint_dir):
                print(f"Removing existing debug keypoint matches: {debug_keypoint_dir}")
                shutil.rmtree(debug_keypoint_dir)
            
            # Remove alignment transform if it exists
            alignment_path = os.path.join(db_dir, "alignment.pt")
            if os.path.exists(alignment_path):
                print(f"Removing existing alignment file: {alignment_path}")
                os.remove(alignment_path)
            
            # Remove trajectory files if they exist
            traj_path = os.path.join(db_dir, "query_trajectory_only.ply")
            if os.path.exists(traj_path):
                print(f"Removing existing trajectory file: {traj_path}")
                os.remove(traj_path)
            
            normals_path = os.path.join(db_dir, "query_normals.ply")
            if os.path.exists(normals_path):
                print(f"Removing existing normals file: {normals_path}")
                os.remove(normals_path)

        print(f"✓ Output directory: {self.config.output_dir}")
        print(f"✓ Database path: {self.config.database_path}")

    def _cache_frames(self):
        """Cache video frames and associated telemetry data with memory management."""
        print("Start caching frames.")
        
        print("Setting up undistortion maps.")
        self.undistortion_maps = create_undistortion_maps_from_file(self.config.cam_dist_path)
        self.camera_intrinsics_undist = self.undistortion_maps.cam_undist.get_camera()
        self.camera_intrinsics_dist = self.undistortion_maps.cam_dist.get_camera()
        self.intrinsics = self._setup_intrinsics()

        if self.config.telemetry_file:
            self.telemetry = TelemetryImporter()
            self.telemetry.read_generic_json(self.config.telemetry_file)
            self.llh0 = self.telemetry.telemetry["gps_llh"][0]
            print(f"Telemetry file read. LLH0: {self.llh0}")

        self.cap = cv2.VideoCapture(self.config.video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate start and end frames from time in seconds
        start_frame = int(self.config.start_time_seconds * self.fps)
        if self.config.end_time_seconds <= 0:
            end_frame = total_frames
        else:
            end_frame = min(int(self.config.end_time_seconds * self.fps), total_frames)
        
        print(f"Video FPS: {self.fps}, Total frames: {total_frames}")
        print(f"Time range: {self.config.start_time_seconds}s - {self.config.end_time_seconds}s")
        print(f"Frame range: {start_frame} - {end_frame}")
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        expected_frames = end_frame - start_frame
        self.frame_buffer = [None] * expected_frames
        self.frame_pos_buffer = [0] * expected_frames
        self.frametimes_ns = [0] * expected_frames
        
        img_norm = IMAGE_NORMALIZATION_DICT["dinov2"]
        self.img_transform = tvf.Compose(
            [tvf.ToTensor(), tvf.Normalize(mean=img_norm.mean, std=img_norm.std)]
        )

        frame_count = 0
        invalid_cnt = 0
        for frame_idx in range(start_frame, end_frame):
            ret, img = self.cap.read()

            if not ret:
                invalid_cnt += 1
                if invalid_cnt > 100:
                    print("Too many invalid frames. Stopping.")
                    break
                continue
            
            if self.undistortion_maps:
                # First, resize to calibration size and undistort.
                # The output image will be at the calibration resolution.
                img = self.undistortion_maps.undistort_image(img, target_size=None)

            if self.target_size:
                # Then, resize to the neural network's target size.
                img = cv2.resize(img, self.target_size)
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img_pil = Image.fromarray(img_rgb)
            normalized_img = self.img_transform(img_pil)

            self.frame_buffer[frame_count] = normalized_img
            self.frame_pos_buffer[frame_count] = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.frametimes_ns[frame_count] = int(self.cap.get(cv2.CAP_PROP_POS_MSEC) * MS_TO_NS)

            frame_count += 1

        self.cap.release()
        
        actual_frames = frame_count
        self.frame_buffer = self.frame_buffer[:actual_frames]
        self.frame_pos_buffer = self.frame_pos_buffer[:actual_frames]
        self.frametimes_ns = self.frametimes_ns[:actual_frames]
            
        print(f"Finished caching {actual_frames} frames.")
        
        if self.telemetry:
            self.gps_ecef, self.gps_prec, self.gps_vel3d, self.interp_ftns = (
                self.telemetry.get_gps_pos_at_frametimes(self.frametimes_ns))
            
            # imu to camera transformation
            R_c_i = RS.from_quat([-0.005614731940060652, 0.7154606008027313, -0.6985867170157808, 0.00782318946143889]).as_matrix()

            self.gravity = self.telemetry.get_gravity_at_times(self.interp_ftns, R_c_i)
            self.gps_ned = ecef_to_ned(self.gps_ecef, self.llh0, self.interp_ftns)
        
            if not self.interp_ftns.any():
                print("No overlapping telemetry found for the cached frames.")
                return True

            first_id = self.frametimes_ns.index(self.interp_ftns[0])
            last_id = self.frametimes_ns.index(self.interp_ftns[-1])

            print("Setting valid frame id window to {}({}) and {}({}) (where we have GPS).".format(  
                self.frame_pos_buffer[first_id], self.frame_pos_buffer[0], 
                self.frame_pos_buffer[last_id], self.frame_pos_buffer[-1]))
            
            self.frame_buffer = self.frame_buffer[first_id:last_id+1]
            self.frame_pos_buffer = self.frame_pos_buffer[first_id:last_id+1]
            self.frametimes_ns = self.frametimes_ns[first_id:last_id+1]
        
        self.max_frames = len(self.frame_buffer)
        return True

    def _setup_intrinsics(self):
        """Calculates and scales the camera intrinsics."""
        self.intrinsics = self.camera_intrinsics_undist
        org_width, org_height = self.camera_intrinsics_undist.ImageWidth(), self.camera_intrinsics_undist.ImageHeight()
        
        average_aspect_ratio = org_width / org_height
        target_width, target_height = find_closest_aspect_ratio(average_aspect_ratio, self.config.resolution_set)
        self.target_size = (target_width, target_height)

        scale_x = target_width / org_width
        scale_y = target_height / org_height

        intrinsics_tensor = torch.tensor(self.camera_intrinsics_undist.GetCalibrationMatrix()).float()
        return scale_intrinsics(intrinsics_tensor, scale_x, scale_y)

    def _initialize_models(self):
        """Initializes and loads the required models."""
        print("Initializing MapAnything model...")
        self.map_anything = MapAnything.from_pretrained("facebook/map-anything-apache").to(self.device).eval()
        print("MapAnything model initialized.")

        # from torchao.quantization.quant_api import (
        #     Int8DynamicActivationInt8WeightConfig,
        #     Int4WeightOnlyConfig,
        #     Int8DynamicActivationInt4WeightConfig,
        #     quantize_,
        # )

        # quantize_(self.map_anything, Int8DynamicActivationInt4WeightConfig(),)
        #self.map_anything = torch.compile(self.map_anything) #, backend="tensorrt"
        
        # Initialize matcher for keypoint matching and verification
        # max_resolution=512 for faster matching (default is 640)
        self.matcher = Matcher(
            device=self.device, 
            keypoint_matching_threshold=self.config.keypoint_matching_threshold,
            max_resolution=320  # Lower resolution = faster matching
        )
    
    def run_slam_for_sequence(self):
        """Runs the standard sequential SLAM pipeline for a new video sequence."""
        print("\n--- Running New SLAM Sequence ---")
        self._run_pipeline()

    def extend_map_with_sequence(self):
        """Runs the SLAM pipeline to extend an existing map with a new video sequence."""
        if not self.config.relocalization_db_path or not os.path.exists(self.config.relocalization_db_path):
            raise FileNotFoundError("Relocalization database path not provided or file not found.")
        
        print(f"\n--- Extending Map using Database: {self.config.relocalization_db_path} ---")
        
        # Load the existing database for relocalization
        print("Loading relocalization database...")
        db_raw = torch.load(self.config.relocalization_db_path, weights_only=False)
        db_data = self._unpack_reference_db(db_raw)
        
        # --- Find optimal start/end frames using Place Recognition ---
        processed_image_paths = self._find_and_prepare_localization_frames(db_data)

        if not processed_image_paths:
            print("Could not find a suitable sequence for localization. Aborting.")
            return

        self._run_pipeline(relocalization_db=db_data, localization_frames=processed_image_paths)

    def _unpack_reference_db(self, db_data):
        """Unpack reference database from saved data.
        Returns just the reference_db list (no place vectors needed anymore).
        """
        try:
            if isinstance(db_data, tuple) and len(db_data) == 2:
                reference_db, _ = db_data  # Ignore place vectors
                return reference_db
            # Fallback: legacy format as list of entries
            if isinstance(db_data, list):
                return db_data
        except Exception as e:
            print(f"Warning: Failed to unpack reference DB: {e}")
        # As a last resort, treat as empty
        return []

    def _find_and_prepare_localization_frames(self, db_data):
        """
        Localize a query sub-sequence to the reference by:
        1) Comparing query place vectors to the first and last reference vectors to get start/end query indices
        2) Expanding by a small buffer
        3) Trimming internal query buffers to this window to save memory
        4) Returning the trimmed query frame identifiers (e.g., 'frame_0', ...)
        """
        # 1. Unpack database
        reference_db = self._unpack_reference_db(db_data)
        print(f"Loaded reference database with {len(reference_db)} entries.")

        # 2. Validate we have enough frames
        if not reference_db or len(self.frame_buffer) < 25:
            print("ERROR: Not enough frames for localization!")
            return []

        # Get first and last reference images
        ref_start_img_path = reference_db[0]['image_path']
        ref_end_img_path = reference_db[-1]['image_path']

        # 3. Match first 25 query frames against first reference frame using LOFTR
        print("Finding best start frame using LOFTR matching on first 25 frames...")
        num_candidates = min(25, len(self.frame_buffer))
        best_start_matches = 0
        start_query_frame_idx = 0
        
        for i in range(num_candidates):
            query_img = self.frame_buffer[i]
            num_matches = self.matcher.count_keypoint_matches(query_img, ref_start_img_path, return_details=False)
            
            print(f"  Frame {i}: {num_matches} LOFTR matches")
            
            if num_matches > best_start_matches:
                best_start_matches = num_matches
                start_query_frame_idx = i
        
        print(f"✓ Selected start frame {start_query_frame_idx} with {best_start_matches} LOFTR matches")
        
        # 4. Match last 25 query frames against last reference frame using LOFTR
        print("\nFinding best end frame using LOFTR matching on last 25 frames...")
        start_idx = max(0, len(self.frame_buffer) - 25)
        best_end_matches = 0
        end_query_frame_idx = len(self.frame_buffer) - 1
        
        for i in range(start_idx, len(self.frame_buffer)):
            query_img = self.frame_buffer[i]
            num_matches = self.matcher.count_keypoint_matches(query_img, ref_end_img_path, return_details=False)
            
            print(f"  Frame {i}: {num_matches} LOFTR matches")
            
            if num_matches > best_end_matches:
                best_end_matches = num_matches
                end_query_frame_idx = i
        
        print(f"✓ Selected end frame {end_query_frame_idx} with {best_end_matches} LOFTR matches")
        
        if start_query_frame_idx > end_query_frame_idx:
            start_query_frame_idx, end_query_frame_idx = end_query_frame_idx, start_query_frame_idx
        
        print("="*70)
        print(f"FINAL LOCALIZATION RESULTS:")
        print(f"  ✓ Start frame: {start_query_frame_idx} with {best_start_matches} LOFTR matches")
        print(f"  ✓ End frame:   {end_query_frame_idx} with {best_end_matches} LOFTR matches")
        print("="*70 + "\n")

        # 4. Add buffer around start/end and trim buffers
        overlap = self.config.relocalization_frame_overlap
        q_start = max(0, start_query_frame_idx - overlap)
        q_end = min(len(self.frame_buffer), end_query_frame_idx + overlap + 1)
        print(f"Trimming query to [{q_start}, {q_end}) with overlap {overlap}.")
        if q_start >= q_end:
            print("Warning: Computed empty query window. Aborting localization selection.")
            return []

        # Store initial reference indices for spatial matching in subsequent chunks
        self.initial_ref_start_idx = 0  # Reference DB start index
        self.initial_ref_end_idx = len(reference_db) - 1  # Reference DB end index
        self.current_ref_idx = 0  # Will be updated as we process chunks

        # Trim internal buffers to reduce memory
        self.frame_buffer = self.frame_buffer[q_start:q_end]
        self.frame_pos_buffer = self.frame_pos_buffer[q_start:q_end]
        self.frametimes_ns = self.frametimes_ns[q_start:q_end]
        print(f"Query buffers trimmed to {len(self.frame_buffer)} frames.")

        # Build identifiers for the trimmed frames, respecting stride
        # Apply stride to get the actual frames to process
        selected_query_paths = [f"frame_{i}" for i in range(0, len(self.frame_buffer), self.config.stride)]
        print(f"Selected {len(selected_query_paths)} query frames for localization (stride={self.config.stride}).")
        return selected_query_paths

    def _preprocess_views(self, views):
        """Preprocesses the views for the MapAnything model."""

        ignore_keys = set(
            [
                "instance",
                "idx",
                "true_shape",
                "data_norm_type"
            ]
        )
        for view in views:
            for name in view.keys():
                if name in ignore_keys:
                    continue
                view[name] = view[name].to(self.device, non_blocking=True)
        return views

    def _run_pipeline(self, relocalization_db=None, localization_frames=None):
        """The core SLAM processing pipeline, adaptable for sequential or relocalization modes."""
        overall_start_time = time.time()

        if localization_frames:
            # For relocalization, localization_frames contains the pre-selected frames (may already be strided)
            processed_image_paths = localization_frames
            # Create full frame list for proper interpolation (ALL frames from buffer)
            all_frame_paths = [f"frame_{i}" for i in range(len(self.frame_buffer))]
            print(f"Running localization on a pre-selected sequence of {len(processed_image_paths)} frames.")
            print(f"  Full frame range for interpolation: frame_0 to frame_{len(all_frame_paths)-1}")
        elif self.config.video_path:
            # Create full frame list (ALL frames from buffer, not just strided ones)
            all_frame_paths = [f"frame_{i}" for i in range(len(self.frame_buffer))]
            # Process only every Nth frame (with stride)
            processed_image_paths = all_frame_paths[::self.config.stride]
            print(f"Processing {len(self.frame_buffer)} frames from video. With stride={self.config.stride}, processing {len(processed_image_paths)} frames.")
            print(f"  Full frame range for interpolation: frame_0 to frame_{len(all_frame_paths)-1}")
        else:
            raise ValueError("Video-only mode: image list input is not supported.")

        if len(processed_image_paths) < 2: return

        self.last_chunk_poses_for_fallback = {} # Used for sequential fallback
        self.last_chunk_depth_z_for_fallback = {} # Depth maps for overlap priors
        database = []

        chunk_step = self.config.chunk_size - self.config.overlap_size
        num_chunks = len(range(0, len(processed_image_paths), chunk_step))
        for i, chunk_start_idx in enumerate(range(0, len(processed_image_paths), chunk_step)):
            self.chunk_idx_for_debug = i # Store for debug image naming
            chunk_end_idx = min(chunk_start_idx + self.config.chunk_size, len(processed_image_paths))
            current_chunk_paths = processed_image_paths[chunk_start_idx:chunk_end_idx]
            
            if len(current_chunk_paths) < 2: continue
            
            print(f"\n--- Processing Chunk {i+1}/{num_chunks} ---")
            
            place_vectors_for_chunk = None
            last_ref_pose = None
            last_ref_idx = -1
            if relocalization_db:
                view_set, paths_in_view_set, first_overlap_pose, place_vectors_for_chunk, last_ref_pose, last_ref_idx = self._prepare_relocalization_views(current_chunk_paths, relocalization_db)
            else:
                view_set, paths_in_view_set, first_overlap_pose = self._prepare_sequential_views(current_chunk_paths, self.last_chunk_poses_for_fallback, self.last_chunk_depth_z_for_fallback)

            if not view_set or len(view_set) < 2:
                print("Not enough views in chunk to process.")
                continue

            # preprocess views
            view_set_pre = self._preprocess_views(view_set)
            view_set_pre = preprocess_input_views_for_inference(view_set_pre)

            with torch.no_grad():
                with torch.autocast("cuda", enabled=True, dtype=torch.float16):
                    raw_predictions = self.map_anything(view_set_pre, True)

            predictions = postprocess_model_outputs_for_inference(
                raw_outputs=raw_predictions,
                input_views=view_set_pre,
                apply_mask=True,
                mask_edges=True,
                edge_normal_threshold=5.0,
                edge_depth_threshold=0.03,
                apply_confidence_mask=True,
                confidence_percentile=20,
            )
            
            # Align query reconstruction to global frame
            # Strategy: Always use FIRST pose in view_set (index 0)
            # - Chunk 1: First pose is first reference frame
            # - Chunk 2+: First pose is first query overlap frame
            if first_overlap_pose is not None:
                is_first_chunk = i == 0
                chunk_label = f"Chunk {i+1}"
                print(f"  [{chunk_label}] Aligning using first pose in view_set (index 0)")
                
                new_first_pose = predictions[0]['camera_poses']
                transform = first_overlap_pose.to(self.device) @ torch.inverse(new_first_pose)
                predictions = self._apply_transform_to_chunk(predictions, transform)
            else:
                print(f"  Warning: No overlap pose found for chunk {i+1}, skipping alignment")

            # Filter for query sequence data
            query_predictions, query_paths = [], []
            for j, pred in enumerate(predictions):
                path = paths_in_view_set[j]
                if path in current_chunk_paths:
                    query_predictions.append(pred)
                    query_paths.append(path)

            # --- Estimate Ground Plane for the Chunk ---
            try:
                ground_normal, plane_d = estimate_ground_plane(query_predictions, distance_threshold=0.2, ransac_n=3, num_iterations=1000, device=self.device)
            except Exception as e:
                print(f"Chunk {i}: Could not estimate ground plane, will use fallback. {e}")

            # --- Trim Overlap from End (except for last chunk) to avoid duplicates ---
            is_last_chunk = (i == num_chunks - 1)
            if not is_last_chunk and len(query_predictions) > self.config.overlap_size:
                # Drop the last overlap_size frames - they'll be saved with the next chunk
                trim_idx = len(query_predictions) - self.config.overlap_size
                query_predictions_to_save = query_predictions[:trim_idx]
                query_paths_to_save = query_paths[:trim_idx]
                print(f"  Trimming last {self.config.overlap_size} overlap frames from chunk (will be in next chunk)")
            else:
                query_predictions_to_save = query_predictions
                query_paths_to_save = query_paths

            # Save chunk data and update database
            all_normals_for_chunk = [ground_normal] * len(query_predictions_to_save) if ground_normal is not None else [None] * len(query_predictions_to_save)
            all_plane_ds_for_chunk = [plane_d] * len(query_predictions_to_save) if plane_d is not None else [None] * len(query_predictions_to_save)
            
            # Get intrinsics and image shape for rendering
            intrinsics_np = self.intrinsics.cpu().numpy() if isinstance(self.intrinsics, torch.Tensor) else np.array(self.intrinsics)
            image_shape = (self.target_size[1], self.target_size[0])  # (height, width)
            
            # Calculate full frame range for interpolation
            # Extract frame indices from first and last keyframe paths
            first_frame_idx = int(''.join(c for c in query_paths_to_save[0] if c.isdigit()))
            last_frame_idx = int(''.join(c for c in query_paths_to_save[-1] if c.isdigit()))
            
            # These indices correspond to positions in all_frame_paths
            full_chunk_start_idx = first_frame_idx
            full_chunk_end_idx = last_frame_idx + 1  # +1 for exclusive end
            
            save_chunk_with_interpolation(
                i, 
                full_chunk_start_idx, 
                full_chunk_end_idx, 
                all_frame_paths, 
                query_predictions_to_save, 
                query_paths_to_save, 
                all_normals_for_chunk, 
                all_plane_ds_for_chunk,
                self.config.output_dir,
                self.config.max_depth_for_point_cloud,
                self.config.stride,
                self.config.video_path,
                self.frametimes_ns,
                self.frame_pos_buffer,
                intrinsics_np,
                image_shape
            )

            # Add only non-overlapping frames to database
            processed_chunk_data = self._process_chunk_and_update_db(
                query_predictions_to_save, query_paths_to_save, ground_normal, plane_d, place_vectors_for_chunk)
            database.extend(processed_chunk_data)

            # --- Update Last Chunk Poses for Next Overlap ---
            # Store ALL frames (including overlap) for next chunk to use as priors
            # Use original query paths as keys to preserve overlap identity across chunks
            self.last_chunk_poses_for_fallback.clear()
            self.last_chunk_depth_z_for_fallback.clear()
            for idx, pred in enumerate(query_predictions):
                if idx < len(query_paths):
                    key = query_paths[idx]
                    self.last_chunk_poses_for_fallback[key] = pred["camera_poses"]
                    self.last_chunk_depth_z_for_fallback[key] = pred["depth_z"]
                
            if self.config.create_debug_chunk_visualizations:
                save_debug_chunk_visualization(i, query_predictions, self.config.output_dir, self.config.max_depth_for_point_cloud)

        overall_proc_time = time.time() - overall_start_time
        
        # --- Trim Query Trajectory Based on Spatial Distance to Reference ---
        if relocalization_db is not None:
            print("\n--- Trimming Query Trajectory Based on Spatial Distance ---")
            database = self._trim_query_to_reference_coverage(database, relocalization_db)
        
        # --- Final Alignment and Database Save ---
        alignment_transform = None
        # if relocalization_db is None:
        #     # Only run GPS/gravity alignment for standalone query reconstructions.
        #     if self.telemetry and self.gravity is not None:
        #         # 1. Align to Gravity
        #         gravity_transform = align_reconstruction_to_gravity(database, self.gravity, self.frametimes_ns, self.device)
                
        #         if gravity_transform is not None:
        #             # Create a temporary database aligned to gravity for subsequent alignments
        #             db_grav_aligned = [{'camera_poses': gravity_transform.cpu() @ entry['camera_poses'], 'image_path': entry['image_path']} for entry in database]

        #             # 2. Align to GPS direction in the XY plane (zero_z=True)
        #             gps_xy_transform = align_reconstruction_to_gps_direction(db_grav_aligned, self.gps_ned, self.frametimes_ns, True, self.device)

        #             if gps_xy_transform is not None:
        #                 # Combine gravity and XY alignment
        #                 alignment_transform = gps_xy_transform @ gravity_transform
                        
        #                 # Create a temporary database also aligned in the XY plane for the final step
        #                 db_xy_aligned = [{'camera_poses': gps_xy_transform.cpu() @ entry['camera_poses'], 'image_path': entry['image_path']} for entry in db_grav_aligned]

        #                 # 3. Perform final full 3D alignment (zero_z=False)
        #                 final_gps_transform = align_reconstruction_to_gps_direction(db_xy_aligned, self.gps_ned, self.frametimes_ns, False, self.device)

        #                 if final_gps_transform is not None:
        #                     # Combine all transforms: T_final = T_gps_full @ T_gps_xy @ T_gravity
        #                     alignment_transform = final_gps_transform @ alignment_transform
        #             else:
        #                 # Fallback to just gravity alignment if GPS XY fails
        #                 alignment_transform = gravity_transform
        #     else:
        #         # Fallback for sequences without telemetry
        #         print("No telemetry or gravity data available, attempting simple GPS direction alignment.")
        #         alignment_transform = align_reconstruction_to_gps_direction(database, self.gps_ned, self.frametimes_ns, True, self.device)

        self._save_database(database, alignment_transform)
        
        # Save trajectory-only PLY for query reconstruction (only during relocalization)
        if relocalization_db is not None:
            try:
                save_query_trajectory_and_normals(database, self.config.database_path)
            except Exception as e:
                print(f"Warning: Could not save trajectory/normals PLY: {e}")

        if relocalization_db is None and alignment_transform is not None:
            # Save the final transform for the point cloud processor and align chunks
            alignment_path = os.path.join(os.path.dirname(self.config.database_path), "alignment.pt")
            torch.save(alignment_transform, alignment_path)
            print(f"Saved final alignment transform to {alignment_path}")
            save_chunks_for_visualization(self.config.output_dir, alignment_transform)


        print("\n--- Total Execution Summary ---")
        print(f"Processed {len(database)} frames in {overall_proc_time:.2f} seconds.")

    def _trim_query_to_reference_coverage(self, database: List[Dict], relocalization_db_data: Tuple) -> List[Dict]:
        """
        Trims the query database to precise spatial coverage of the reference trajectory.
        Finds the query frames closest to reference start/end poses and drops everything outside.
        
        Args:
            database: Query reconstruction database with poses
            relocalization_db_data: Reference database tuple (db, vectors)
            
        Returns:
            Trimmed database containing only frames within reference coverage
        """
        if not database:
            return database
        
        reference_db = relocalization_db_data
        
        # Get reference start and end poses
        ref_start_pose = reference_db[0]['camera_poses']
        ref_end_pose = reference_db[-1]['camera_poses']
        
        if isinstance(ref_start_pose, torch.Tensor):
            ref_start_pos = ref_start_pose[0, :3, 3]
            ref_end_pos = ref_end_pose[0, :3, 3]
        else:
            ref_start_pos = torch.tensor(ref_start_pose[0][:3, 3])
            ref_end_pos = torch.tensor(ref_end_pose[0][:3, 3])
        
        print(f"Reference trajectory: {len(reference_db)} frames")
        print(f"  Start position: {ref_start_pos.cpu().numpy()}")
        print(f"  End position: {ref_end_pos.cpu().numpy()}")
        
        # Calculate distances from reference start/end to all query poses
        min_start_dist = float('inf')
        min_end_dist = float('inf')
        closest_start_idx = 0
        closest_end_idx = len(database) - 1
        
        for idx, entry in enumerate(database):
            query_pose = entry['camera_poses']
            if isinstance(query_pose, torch.Tensor):
                query_pos = query_pose[0, :3, 3] if query_pose.ndim == 3 else query_pose[:3, 3]
            else:
                query_pos = torch.tensor(query_pose[0][:3, 3] if len(query_pose.shape) == 3 else query_pose[:3, 3])
            
            # Distance to reference start
            dist_to_start = torch.norm(query_pos - ref_start_pos).item()
            if dist_to_start < min_start_dist:
                min_start_dist = dist_to_start
                closest_start_idx = idx
            
            # Distance to reference end
            dist_to_end = torch.norm(query_pos - ref_end_pos).item()
            if dist_to_end < min_end_dist:
                min_end_dist = dist_to_end
                closest_end_idx = idx
        
        # Ensure start comes before end
        if closest_start_idx > closest_end_idx:
            closest_start_idx, closest_end_idx = closest_end_idx, closest_start_idx
            min_start_dist, min_end_dist = min_end_dist, min_start_dist
        
        print(f"\nQuery trajectory before trimming: {len(database)} frames")
        print(f"  Closest to ref start: frame {closest_start_idx} at distance {min_start_dist:.3f}m")
        print(f"  Closest to ref end: frame {closest_end_idx} at distance {min_end_dist:.3f}m")
        
        # Trim database
        trimmed_database = database[closest_start_idx:closest_end_idx + 1]
        
        num_dropped_start = closest_start_idx
        num_dropped_end = len(database) - closest_end_idx - 1
        
        print(f"\nQuery trajectory after trimming: {len(trimmed_database)} frames")
        print(f"  Dropped {num_dropped_start} frames from start")
        print(f"  Dropped {num_dropped_end} frames from end")
        print(f"  Kept frames [{closest_start_idx}, {closest_end_idx}]")
        
        return trimmed_database
    
    def _prepare_sequential_views(
        self,
        current_chunk_paths: List[str],
        last_chunk_poses: Dict[str, torch.Tensor],
        last_chunk_depth_z: Dict[str, torch.Tensor],
    ) -> Tuple[List[Dict], List[str], torch.Tensor]:
        """Prepares the view_set for a standard sequential chunk."""
        
        view_set = []
        paths_in_view_set = []
        first_overlap_pose = None
        
        # 1. Add overlapping views from the previous chunk as references
        if last_chunk_poses:
            # Determine which paths from the current chunk overlap with the last one
            overlap_paths = [path for path in current_chunk_paths if path in last_chunk_poses]
            if overlap_paths:
                print(f"Found {len(overlap_paths)} overlapping views from previous chunk to use as priors.")
                # Load overlap views from video buffer (video-only)
                overlap_views = self._get_views_from_buffer(overlap_paths)
                for i, view in enumerate(overlap_views):
                    path = overlap_paths[i]
                    pose = last_chunk_poses[path]
                    depth_z = last_chunk_depth_z[path]
                    view.update({
                        "camera_poses": pose,
                        "intrinsics": self.intrinsics,
                        "is_metric_scale": torch.tensor([True], device=self.device),
                        "depth_z": depth_z
                    })
                view_set.extend(overlap_views)
                paths_in_view_set.extend(overlap_paths)
                first_overlap_pose = last_chunk_poses[overlap_paths[0]]
        
        # 2. Add all new (unposed) views from the current chunk
        new_paths = [path for path in current_chunk_paths if path not in paths_in_view_set]
        new_views = self._get_views_from_buffer(new_paths)
        for view in new_views:
            view["intrinsics"] = self.intrinsics

        view_set.extend(new_views)
        paths_in_view_set.extend(new_paths)
        
        return view_set, paths_in_view_set, first_overlap_pose

    def _find_closest_reference_by_spatial_distance(self, query_pose, reference_db, search_start_idx=None, search_window=None):
        """
        Find the closest reference frame to a query pose based on spatial distance.
        
        Args:
            query_pose: 4x4 camera pose tensor
            reference_db: List of reference database entries
            search_start_idx: Starting index in reference DB to search from (None = search all)
            search_window: Number of frames to search around the start index (None = search all)
            
        Returns:
            Tuple of (best_ref_idx, best_distance) or (None, float('inf'))
        """
        query_position = query_pose[0,:3, 3]  # Extract translation
        
        best_ref_idx = None
        best_distance = float('inf')
        
        # Determine search range
        if search_start_idx is None or search_window is None:
            # Search all reference frames
            search_start = 0
            search_end = len(reference_db)
        else:
            # Search within window centered on search_start_idx
            search_start = max(0, search_start_idx - search_window // 2)
            search_end = min(search_start_idx + search_window, len(reference_db))
        
        for ref_idx in range(search_start, search_end):
            ref_pose = reference_db[ref_idx]['camera_poses'][0]
            if isinstance(ref_pose, torch.Tensor):
                ref_position = ref_pose[:3, 3].cpu()
            else:
                ref_position = torch.tensor(ref_pose[:3, 3])
            
            # Ensure both tensors are on CPU for distance calculation
            distance = torch.norm(query_position.cpu() - ref_position).item()
            
            if distance < best_distance:
                best_distance = distance
                best_ref_idx = ref_idx
        
        return best_ref_idx, best_distance
    
    def _visualize_matching_grid(
        self,
        query_img_tensor: torch.Tensor,
        reference_db: List[Dict],
        ref_indices: List[int],
        match_counts: List[int],
        best_ref_idx: int,
        save_path: str,
        title: str = "Reference Matching Results"
    ):
        """
        Visualize matching results as a grid showing query and top reference matches.
        
        Args:
            query_img_tensor: Query image tensor (C, H, W)
            reference_db: List of reference database entries
            ref_indices: Indices of reference frames that were matched
            match_counts: Number of matches for each reference frame
            best_ref_idx: Index of best matching reference frame
            save_path: Path to save visualization
            title: Title for the plot
        """
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        
        # Denormalize query image for visualization
        img_norm = IMAGE_NORMALIZATION_DICT["dinov2"]
        mean = torch.as_tensor(img_norm.mean, dtype=torch.float32).view(3, 1, 1).to(query_img_tensor.device)
        std = torch.as_tensor(img_norm.std, dtype=torch.float32).view(3, 1, 1).to(query_img_tensor.device)
        
        query_denorm = query_img_tensor * std + mean
        query_denorm = torch.clamp(query_denorm * 255, 0, 255).byte()
        query_img_np = query_denorm.permute(1, 2, 0).cpu().numpy()
        
        # Sort references by match count (descending)
        sorted_pairs = sorted(zip(ref_indices, match_counts), key=lambda x: x[1], reverse=True)
        top_k = min(5, len(sorted_pairs))  # Show top 5 matches
        
        # Create figure with grid: 1 query + top K references
        fig, axes = plt.subplots(1, top_k + 1, figsize=(4 * (top_k + 1), 4))
        if top_k == 0:
            axes = [axes]
        
        # Plot query image
        axes[0].imshow(query_img_np)
        axes[0].set_title(f'Query Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Plot top K reference matches
        for i, (ref_idx, match_count) in enumerate(sorted_pairs[:top_k]):
            ref_img_path = reference_db[ref_idx]['image_path']
            ref_img_bgr = cv2.imread(ref_img_path)
            if ref_img_bgr is not None:
                ref_img_rgb = cv2.cvtColor(ref_img_bgr, cv2.COLOR_BGR2RGB)
                axes[i + 1].imshow(ref_img_rgb)
                
                # Highlight best match
                border_color = 'green' if ref_idx == best_ref_idx else 'gray'
                border_width = 3 if ref_idx == best_ref_idx else 1
                
                axes[i + 1].set_title(
                    f'Ref #{ref_idx}\n{match_count} matches',
                    fontsize=12,
                    fontweight='bold' if ref_idx == best_ref_idx else 'normal',
                    color=border_color
                )
                
                # Add colored border
                for spine in axes[i + 1].spines.values():
                    spine.set_edgecolor(border_color)
                    spine.set_linewidth(border_width)
                    spine.set_visible(True)
            
            axes[i + 1].axis('off')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved matching visualization to {save_path}")
    
    def _find_best_reference_by_image_matching(
        self, 
        query_img_tensor: torch.Tensor, 
        reference_db: List[Dict], 
        search_center_idx: int = 0,
        search_window: int = 30,
        min_matches: int = 1000
    ) -> Tuple[int, int, List[int], List[int]]:
        """
        Find the best matching reference frame using LoFTR image matching.
        
        Args:
            query_img_tensor: Query image tensor (B, C, H, W) or (C, H, W) normalized
            reference_db: List of reference database entries
            search_center_idx: Center index for search window
            search_window: Number of frames to search on each side of center
            min_matches: Minimum number of matches required
            
        Returns:
            Tuple of (best_ref_idx, num_matches, all_ref_indices, all_match_counts)
        """
        # Squeeze batch dimension if present
        if query_img_tensor.ndim == 4:
            query_img_tensor = query_img_tensor.squeeze(0)
        
        # Define search range
        start_idx = max(0, search_center_idx - search_window // 2)
        end_idx = min(len(reference_db), search_center_idx + search_window // 2)
        
        best_ref_idx = -1
        best_match_count = 0
        all_ref_indices = list(range(start_idx, end_idx))
        
        print(f"  Searching for matches in reference frames [{start_idx}, {end_idx})")
        
        # Batch process all reference images for speed
        ref_img_paths = [reference_db[idx]['image_path'] for idx in all_ref_indices]
        
        try:
            all_match_counts = self.matcher.count_keypoint_matches_batch(
                query_img_tensor,
                ref_img_paths
            )
            
            # Find best match
            if all_match_counts:
                best_match_count = max(all_match_counts)
                if best_match_count > 0:
                    best_idx_in_list = all_match_counts.index(best_match_count)
                    best_ref_idx = all_ref_indices[best_idx_in_list]
                    
                    # Early termination if we found an excellent match
                    if best_match_count > 2000:
                        print(f"  Found excellent match (>{2000} matches) at reference frame {best_ref_idx}, terminating search early")
                    
        except Exception as e:
            print(f"  Warning: Batch matching failed: {e}, falling back to sequential")
            # Fallback to sequential matching with early termination
            all_match_counts = []
            for ref_idx in all_ref_indices:
                ref_img_path = reference_db[ref_idx]['image_path']
                try:
                    num_matches = self.matcher.count_keypoint_matches(
                        query_img_tensor, 
                        ref_img_path, 
                        return_details=False
                    )
                    all_match_counts.append(num_matches)
                    
                    if num_matches > best_match_count:
                        best_match_count = num_matches
                        best_ref_idx = ref_idx
                        
                        # Early termination if we found an excellent match
                        if best_match_count > 2000:
                            print(f"  Found excellent match (>{2000} matches) at reference frame {best_ref_idx}, terminating search early")
                            # Fill remaining with zeros for consistency
                            remaining = len(all_ref_indices) - len(all_match_counts)
                            all_match_counts.extend([0] * remaining)
                            break
                            
                except Exception as e2:
                    print(f"  Warning: Failed to match with reference frame {ref_idx}: {e2}")
                    all_match_counts.append(0)
        
        if best_match_count >= min_matches:
            print(f"  Found best match at reference frame {best_ref_idx} with {best_match_count} matches")
            return best_ref_idx, best_match_count, all_ref_indices, all_match_counts
        else:
            print(f"  No good match found (best: {best_match_count} < {min_matches} matches)")
            return -1, 0, all_ref_indices, all_match_counts
    
    def _prepare_relocalization_views(
        self, current_chunk_paths: List[str], db_data: List[Dict]
    ) -> Tuple[List[Dict], List[str], torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Prepares views by adding reference priors based on spatial proximity.
        
        Strategy:
        - First chunk: Add first N reference frames as priors (assumes query starts near beginning of reference)
        - Subsequent chunks: Use last query pose to find closest reference, add N consecutive reference frames if within threshold
        - Align using the LAST reference pose
        
        Returns:
            Tuple of (view_set, paths_in_view_set, first_overlap_pose, current_vectors, last_ref_pose, last_ref_idx_in_viewset)
        """
        reference_db = db_data
        view_set = []
        paths_in_view_set = []
        first_overlap_pose = None
        current_vectors = None
        last_ref_indices = []  # Will hold reference frames to add at the end
        last_ref_pose = None  # Ground truth pose of last reference frame for alignment
        last_ref_idx_in_viewset = -1  # Index of last reference view in view_set
        
        is_first_chunk = len(self.last_chunk_poses_for_fallback) == 0
        
        # Load current chunk images
        if self.config.video_path:
            current_views = self._get_views_from_buffer(current_chunk_paths)
        else:
            current_views = load_images(
                current_chunk_paths, 
                resize_mode="fixed_size", 
                size=self.target_size,
            )
        
        # ========== FIRST CHUNK: Use Image Matching to Find Best Reference Frames ==========
        if is_first_chunk:
            print(f"[Chunk 1] Using image matching to find best reference frames...")
            
            # Get first and last query images for matching
            first_query_img = current_views[0]['img']  # Already loaded
            last_query_img = current_views[-1]['img']
            
            # Find best matching reference frame for the FIRST query image
            print(f"  Matching first query image...")
            first_ref_idx, first_matches, first_ref_indices, first_match_counts = self._find_best_reference_by_image_matching(
                first_query_img,
                reference_db,
                search_center_idx=0,
                search_window=30,
                min_matches=1000
            )
            
            # Visualize first query matching
            if first_ref_indices:
                vis_path = os.path.join(self.config.output_dir, "debug_matches", "chunk_1_first_query_matches.png")
                self._visualize_matching_grid(
                    first_query_img if first_query_img.ndim == 3 else first_query_img.squeeze(0),
                    reference_db,
                    first_ref_indices,
                    first_match_counts,
                    first_ref_idx,
                    vis_path,
                    title="Chunk 1 - First Query Image Matches"
                )
            
            # Find best matching reference frame for the LAST query image
            print(f"  Matching last query image...")
            last_ref_idx, last_matches, last_ref_indices, last_match_counts = self._find_best_reference_by_image_matching(
                last_query_img,
                reference_db,
                search_center_idx=self.config.chunk_size,
                search_window=10,
                min_matches=1000
            )
            
            # Visualize last query matching
            if last_ref_indices:
                vis_path = os.path.join(self.config.output_dir, "debug_matches", "chunk_1_last_query_matches.png")
                self._visualize_matching_grid(
                    last_query_img if last_query_img.ndim == 3 else last_query_img.squeeze(0),
                    reference_db,
                    last_ref_indices,
                    last_match_counts,
                    last_ref_idx,
                    vis_path,
                    title="Chunk 1 - Last Query Image Matches"
                )
            
            # Build list of reference indices to add (clear previous match results)
            ref_indices_to_add_start = []
            ref_indices_to_add_end = []
            
            if first_ref_idx >= 0:
                ref_indices_to_add_start.append(first_ref_idx)
                # Also add the next frame for temporal continuity
                if first_ref_idx + 1 < len(reference_db):
                    ref_indices_to_add_start.append(first_ref_idx + 1)
                print(f"  Will add reference frames {ref_indices_to_add_start} at start")
            else:
                # Fallback to beginning of reference if no match
                ref_indices_to_add_start = [0]
                print(f"  No good match for first query, using reference frame 0")
            
            if last_ref_idx >= 0 and last_ref_idx not in ref_indices_to_add_start:
                ref_indices_to_add_end.append(last_ref_idx)
                print(f"  Will add reference frame {last_ref_idx} at end")
            else:
                # Fallback or skip if already added
                if last_ref_idx < 0:
                    # Use chunk_size as fallback
                    fallback_idx = min(self.config.chunk_size, len(reference_db) - 1)
                    if fallback_idx not in ref_indices_to_add_start:
                        ref_indices_to_add_end.append(fallback_idx)
                        print(f"  No good match for last query, using reference frame {fallback_idx}")
            
            # Load and add first 2 reference priors at the start
            for i, idx in enumerate(ref_indices_to_add_start):
                if idx >= len(reference_db):
                    continue
                db_entry = reference_db[idx]
                db_view = load_images([db_entry['image_path']], resize_mode="fixed_size", size=self.target_size)[0]
                ref_intrinsics = db_entry.get('intrinsics', self.intrinsics)
                if not isinstance(ref_intrinsics, torch.Tensor):
                    ref_intrinsics = torch.tensor(ref_intrinsics, dtype=torch.float32)
                
                # Load depth if available
                depth_z = None
                if 'depth_path' in db_entry and os.path.exists(db_entry['depth_path']):
                    try:
                        depth_np = np.load(db_entry['depth_path'])
                        depth_z = torch.from_numpy(depth_np).to(self.device)
                        print(f"  Loaded depth prior from {os.path.basename(db_entry['depth_path'])}")
                    except Exception as e:
                        print(f"  Warning: Could not load depth from {db_entry['depth_path']}: {e}")
                
                db_view.update({
                    "intrinsics": ref_intrinsics,
                    "camera_poses": db_entry['camera_poses'].to(self.device),
                    "is_metric_scale": torch.tensor([True], device=self.device)
                })
                
                if depth_z is not None:
                    db_view["depth_z"] = depth_z
                
                view_set.append(db_view)
                paths_in_view_set.append(db_entry['image_path'])
                
                if i == 0:
                    first_overlap_pose = db_entry['camera_poses'].to(self.device)
            
            # Store reference index for next chunk
            self.current_ref_idx = self.config.chunk_size + 1
            print(f"  ✓ Added {len(ref_indices_to_add_start)} reference priors at start for first chunk")
            
            # Store for later addition at end
            last_ref_indices = ref_indices_to_add_end
        
        # ========== SUBSEQUENT CHUNKS: Add Query Overlap First, Then Reference ==========
        else:
            print(f"[Chunk N] Using image matching to find best reference frames...")
            
            # Get last query pose from previous chunk for search window centering
            last_query_path = list(self.last_chunk_poses_for_fallback.keys())[-1]
            last_query_pose = self.last_chunk_poses_for_fallback[last_query_path]
            
            # Find closest reference by spatial distance to center the search window
            center_ref_idx, distance = self._find_closest_reference_by_spatial_distance(
                last_query_pose, reference_db, search_start_idx=None, search_window=None
            )
            
            print(f"  Using reference frame {center_ref_idx} (distance {distance:.3f}m) as search center")
            
            # Get first and last query images for matching
            first_query_img = current_views[0]['img']
            last_query_img = current_views[-1]['img']
            
            # Find best matching reference frame for the FIRST query image
            # Search around center_ref_idx since first query is from overlap with previous chunk
            print(f"  Matching first query image...")
            search_center_first = center_ref_idx
            print(f"  Searching for first query image in reference frames around {search_center_first} (window=5)")
            first_ref_idx, first_matches, first_ref_indices_match, first_match_counts_match = self._find_best_reference_by_image_matching(
                first_query_img,
                reference_db,
                search_center_idx=search_center_first,
                search_window=5,
                min_matches=1000
            )
            
            # Visualize first query matching
            if first_ref_indices_match:
                chunk_dirs = glob.glob(os.path.join(self.config.output_dir, "chunk_*"))
                chunk_num = len(chunk_dirs) + 1
                vis_path = os.path.join(self.config.output_dir, "debug_matches", f"chunk_{chunk_num}_first_query_matches.png")
                self._visualize_matching_grid(
                    first_query_img if first_query_img.ndim == 3 else first_query_img.squeeze(0),
                    reference_db,
                    first_ref_indices_match,
                    first_match_counts_match,
                    first_ref_idx,
                    vis_path,
                    title=f"Chunk {chunk_num} - First Query Image Matches"
                )
            
            # Find best matching reference frame for the LAST query image
            # Search center should be ahead of where the previous chunk ended
            print(f"  Matching last query image...")
            search_center_last = min(len(reference_db) - 1, center_ref_idx + self.config.chunk_size)
            print(f"  Searching for last query image in reference frames [{center_ref_idx}, {search_center_last + 5})")
            last_ref_idx, last_matches, last_ref_indices_match, last_match_counts_match = self._find_best_reference_by_image_matching(
                last_query_img,
                reference_db,
                search_center_idx=search_center_last,
                search_window=5,
                min_matches=1000
            )
            
            # Visualize last query matching
            if last_ref_indices_match:
                chunk_dirs = glob.glob(os.path.join(self.config.output_dir, "chunk_*"))
                chunk_num = len(chunk_dirs) + 1
                vis_path = os.path.join(self.config.output_dir, "debug_matches", f"chunk_{chunk_num}_last_query_matches.png")
                self._visualize_matching_grid(
                    last_query_img if last_query_img.ndim == 3 else last_query_img.squeeze(0),
                    reference_db,
                    last_ref_indices_match,
                    last_match_counts_match,
                    last_ref_idx,
                    vis_path,
                    title=f"Chunk {chunk_num} - Last Query Image Matches"
                )
            
            # Determine which reference frames to add
            first_ref_to_add = first_ref_idx if first_ref_idx >= 0 else max(0, center_ref_idx - self.config.chunk_size)
            last_ref_to_add = last_ref_idx if last_ref_idx >= 0 else min(center_ref_idx, len(reference_db) - 1)
            
            print(f"  Will add reference frame {first_ref_to_add} after overlap, and {last_ref_to_add} at end")
            
            # Store reference indices for later (will be added after overlap in correct order)
            ref_frame_to_add_after_overlap = first_ref_to_add
            last_ref_indices = [last_ref_to_add]
        
        # ========== Add Sequential Overlap from Previous Query Chunk ==========
        overlapping_sequential_paths = [
            p for p in current_chunk_paths 
            if p in self.last_chunk_poses_for_fallback and p not in paths_in_view_set
        ]
        
        if overlapping_sequential_paths:
            print(f"  Adding {len(overlapping_sequential_paths)} sequential overlap views from previous query chunk")
            overlap_views = self._get_views_from_buffer(overlapping_sequential_paths)
            for i, view in enumerate(overlap_views):
                path = overlapping_sequential_paths[i]
                pose = self.last_chunk_poses_for_fallback[path]
                depth_z = self.last_chunk_depth_z_for_fallback[path]
                view.update({
                    "camera_poses": pose,
                    "intrinsics": self.intrinsics,
                    "is_metric_scale": torch.tensor([True], device=self.device),
                    "depth_z": depth_z
                })
            view_set.extend(overlap_views)
            paths_in_view_set.extend(overlapping_sequential_paths)
            
            # For subsequent chunks, set first_overlap_pose to FIRST overlap frame (index 0 in view_set)
            if first_overlap_pose is None and overlapping_sequential_paths:
                first_overlap_pose = self.last_chunk_poses_for_fallback[overlapping_sequential_paths[0]]
                print(f"  Using first sequential overlap (index 0) as alignment anchor")
        
        # ========== Add First Reference Frame (Subsequent Chunks Only) ==========
        # For subsequent chunks, add first matching reference frame AFTER overlap
        if not is_first_chunk and 'ref_frame_to_add_after_overlap' in locals():
            print(f"  Adding first reference frame {ref_frame_to_add_after_overlap} after overlap")
            db_entry = reference_db[ref_frame_to_add_after_overlap]
            db_view = load_images([db_entry['image_path']], resize_mode="fixed_size", size=self.target_size)[0]
            ref_intrinsics = db_entry.get('intrinsics', self.intrinsics)
            if not isinstance(ref_intrinsics, torch.Tensor):
                ref_intrinsics = torch.tensor(ref_intrinsics, dtype=torch.float32)
            
            # Load depth if available
            depth_z = None
            if 'depth_path' in db_entry and os.path.exists(db_entry['depth_path']):
                try:
                    depth_np = np.load(db_entry['depth_path'])
                    depth_z = torch.from_numpy(depth_np).to(self.device)
                    print(f"  Loaded depth prior from {os.path.basename(db_entry['depth_path'])}")
                except Exception as e:
                    print(f"  Warning: Could not load depth from {db_entry['depth_path']}: {e}")
            
            db_view.update({
                "intrinsics": ref_intrinsics,
                "camera_poses": db_entry['camera_poses'].to(self.device),
                "is_metric_scale": torch.tensor([True], device=self.device)
            })
            
            if depth_z is not None:
                db_view["depth_z"] = depth_z
            
            view_set.append(db_view)
            paths_in_view_set.append(db_entry['image_path'])
        
        # ========== Add New (Unposed) Query Views ==========
        new_query_paths = [p for p in current_chunk_paths if p not in paths_in_view_set]
        for path in new_query_paths:
            original_idx = current_chunk_paths.index(path)
            view = current_views[original_idx]
            view["intrinsics"] = self.intrinsics
            view_set.append(view)
        paths_in_view_set.extend(new_query_paths)
        
        # ========== Add Last Reference Frame(s) at the End ==========
        if last_ref_indices:
            print(f"  Adding {len(last_ref_indices)} reference prior(s) at the end")
            for idx in last_ref_indices:
                if idx >= len(reference_db):
                    continue
                db_entry = reference_db[idx]
                db_view = load_images([db_entry['image_path']], resize_mode="fixed_size", size=self.target_size)[0]
                ref_intrinsics = db_entry.get('intrinsics', self.intrinsics)
                if not isinstance(ref_intrinsics, torch.Tensor):
                    ref_intrinsics = torch.tensor(ref_intrinsics, dtype=torch.float32)
                
                # Load depth if available
                depth_z = None
                if 'depth_path' in db_entry and os.path.exists(db_entry['depth_path']):
                    try:
                        depth_np = np.load(db_entry['depth_path'])
                        depth_z = torch.from_numpy(depth_np).to(self.device)
                        print(f"  Loaded depth prior from {os.path.basename(db_entry['depth_path'])}")
                    except Exception as e:
                        print(f"  Warning: Could not load depth from {db_entry['depth_path']}: {e}")
                
                db_view.update({
                    "intrinsics": ref_intrinsics,
                    "camera_poses": db_entry['camera_poses'].to(self.device),
                    "is_metric_scale": torch.tensor([True], device=self.device)
                })
                
                if depth_z is not None:
                    db_view["depth_z"] = depth_z
                
                # Track the LAST reference pose for alignment (this will be the final one added)
                last_ref_pose = db_entry['camera_poses'].to(self.device)
                last_ref_idx_in_viewset = len(view_set)
                
                view_set.append(db_view)
                paths_in_view_set.append(db_entry['image_path'])
        
        print(f"  Total views in set: {len(view_set)} ({len(view_set) - len(new_query_paths)} priors, {len(new_query_paths)} new)")
        if last_ref_pose is not None:
            print(f"  Will use last reference pose (index {last_ref_idx_in_viewset}) for alignment")
        
        return view_set, paths_in_view_set, first_overlap_pose, current_vectors, last_ref_pose, last_ref_idx_in_viewset

    def _align_saved_chunks(self, alignment_transform: torch.Tensor):
        """Iterates through all saved chunks and applies the final alignment transform to them."""
        print("Applying final alignment to all saved chunk files...")
        chunk_dirs = sorted(glob.glob(os.path.join(self.config.output_dir, "chunk_*")))
        transform_np = alignment_transform.cpu().numpy()

        for chunk_dir in chunk_dirs:
            try:
                metadata_file = os.path.join(chunk_dir, "metadata.pt")
                pcd_file = os.path.join(chunk_dir, "point_cloud.npz")

                if not (os.path.exists(metadata_file) and os.path.exists(pcd_file)):
                    continue
                
                # Load data
                metadata = torch.load(metadata_file, map_location="cpu")
                pcd_data = np.load(pcd_file)

                # --- Transform Poses ---
                original_poses = metadata['camera_poses']
                aligned_poses = [torch.from_numpy(transform_np) @ pose for pose in original_poses]
                metadata['camera_poses'] = aligned_poses

                # --- Transform Normals ---
                if 'ground_normals' in metadata:
                    R_align = transform_np[:3, :3]
                    original_normals = [n.numpy() if isinstance(n, torch.Tensor) else n for n in metadata['ground_normals']]
                    aligned_normals = [R_align @ n if n is not None else None for n in original_normals]
                    metadata['ground_normals'] = aligned_normals
                
                metadata['is_aligned'] = True
                torch.save(metadata, metadata_file)

                # --- Transform Points ---
                pts3d = pcd_data['pts3d']
                pts_hom = np.hstack((pts3d, np.ones((pts3d.shape[0], 1))))
                aligned_pts3d = (transform_np @ pts_hom.T).T[:, :3]

                np.savez_compressed(
                    pcd_file,
                    pts3d=aligned_pts3d,
                    colors=pcd_data['colors']
                )
            except Exception as e:
                print(f"Warning: Could not align chunk {os.path.basename(chunk_dir)}: {e}")
        print("Finished aligning chunk files.")

    

    def _get_views_from_buffer(self, paths: List[str]) -> List[Dict]:
        """Constructs view dictionaries from the frame buffer for a list of frame identifiers."""
        views = []
        for path in paths:
            try:
                # Paths are of the form 'frame_{idx}' or may point to saved files; handle both robustly
                base = os.path.basename(path)
                name, _sep, _ext = base.partition('.')
                # Expect prefix 'frame_'; extract trailing digits
                digits = ''.join([c for c in name if c.isdigit()])
                frame_idx = int(digits)
                img_tensor = self.frame_buffer[frame_idx]
                
                view = {
                    'img': img_tensor[None], # Add batch dim
                    'true_shape': np.int32([img_tensor.shape[1], img_tensor.shape[2]]),
                    'idx': frame_idx,
                    'instance': str(frame_idx),
                    'data_norm_type': ['dinov2'],
                }
                views.append(view)
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not retrieve frame for path '{path}'. Error: {e}")
        return views

    def _apply_transform_to_chunk(self, predictions, transform):
        for pred in predictions:
            pred['camera_poses'] = (transform @ pred['camera_poses'])
            pts3d_flat = pred['pts3d'].view(-1, 3)
            pts3d_hom = torch.cat([pts3d_flat, torch.ones(pts3d_flat.shape[0], 1, device=self.device)], dim=1)
            transformed_pts = (transform.squeeze() @ pts3d_hom.T).T
            pred['pts3d'] = transformed_pts[:, :3].view(pred['pts3d'].shape)
        return predictions

    def _process_chunk_and_update_db(self, predictions, paths, ground_normal, plane_d, place_vectors):
        """Processes predictions and updates the database list."""
        processed_data = []
        # Ensure a persistent images directory exists next to the database to store RGB frames
        images_dir = None
        try:
            if self.config.database_path:
                images_dir = os.path.join(os.path.dirname(self.config.database_path), "reference_images")
                os.makedirs(images_dir, exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not create images directory for database frames: {e}")
        for i, pred in enumerate(predictions):
            entry = {
                "camera_poses": pred['camera_poses'].cpu(),
                "image_path": paths[i],
                "ground_normal": ground_normal.cpu() if ground_normal is not None else None,
                "plane_d": plane_d.cpu() if plane_d is not None else None,
                "depth_z": pred['depth_z'].cpu(),
                "intrinsics": self.intrinsics.detach().cpu() if isinstance(self.intrinsics, torch.Tensor) else torch.tensor(self.intrinsics, dtype=torch.float32)
            }
            # Attach GPS NED to database if available (per frame)
            try:
                if hasattr(self, 'gps_ned') and self.gps_ned is not None and self.config.video_path:
                    base = os.path.basename(paths[i])
                    name, _sep, _ext = base.partition('.')
                    digits = ''.join([c for c in name if c.isdigit()])
                    frame_idx = int(digits)
                    if frame_idx < len(self.frametimes_ns):
                        tns = self.frametimes_ns[frame_idx]
                        if tns in self.gps_ned:
                            entry["gps_ned"] = torch.tensor(self.gps_ned[tns], dtype=torch.float32)
            except Exception as e:
                print(f"Warning: Could not attach GPS NED for {paths[i]}: {e}")
            # Persist the RGB frame and depth map used for reconstruction
            try:
                if images_dir is not None and 'img_no_norm' in pred:
                    # Derive a stable filename from the original path identifier
                    base = os.path.basename(paths[i])
                    name, _sep, _ext = base.partition('.')
                    if not name:
                        name = base
                    
                    # Save RGB image
                    rgb_filename = f"{name}.jpg"
                    rgb_save_path = os.path.join(images_dir, rgb_filename)

                    # Convert denormalized image [0,1] (H,W,3) to uint8 RGB
                    img = pred['img_no_norm']
                    if isinstance(img, torch.Tensor):
                        # Expect shape (B, H, W, 3) with B=1
                        if img.ndim == 4:
                            img = img[0]
                        img_np = img.detach().cpu().numpy()
                    else:
                        img_np = np.array(img)
                    img_uint8 = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)

                    # Write as JPEG (convert RGB->BGR for OpenCV)
                    if not os.path.exists(rgb_save_path):
                        cv2.imwrite(rgb_save_path, cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))

                    # Update entry to point to the saved RGB frame
                    entry["image_path"] = rgb_save_path
                    
                    # Save depth map as .npy file
                    if 'depth_z' in pred:
                        depth_filename = f"{name}_depth.npy"
                        depth_save_path = os.path.join(images_dir, depth_filename)
                        
                        # Extract depth_z tensor
                        depth_z = pred['depth_z']
                        if isinstance(depth_z, torch.Tensor):
                            # Remove batch dimension if present
                            if depth_z.ndim == 4:  # (B, 1, H, W)
                                depth_z = depth_z[0]
                            depth_np = depth_z.detach().cpu().numpy()
                        else:
                            depth_np = np.array(depth_z)
                        
                        # Save depth as numpy file
                        if not os.path.exists(depth_save_path):
                            np.save(depth_save_path, depth_np)
                        
                        # Store depth path in database entry
                        entry["depth_path"] = depth_save_path

            except Exception as e:
                print(f"Warning: Failed to save/reference RGB frame or depth for database entry {paths[i]}: {e}")
            
            processed_data.append(entry)
        return processed_data

    def _save_database(self, database, alignment_transform=None):
        """Saves the final processed database to a file."""
        if not self.config.database_path:
            return
        
        if alignment_transform is not None:
            print("Applying final GPS alignment to the database...")
            for entry in database:
                original_pose = entry['camera_poses']
                aligned_pose = alignment_transform.to(original_pose.device) @ original_pose
                entry['camera_poses'] = aligned_pose
        
        # --- Save database ---
        try:
            os.makedirs(os.path.dirname(self.config.database_path), exist_ok=True)
            torch.save(database, self.config.database_path)
            print(f"Saved SLAM database with {len(database)} entries to {self.config.database_path}")
        except Exception as e:
            print(f"Error saving database: {e}")

