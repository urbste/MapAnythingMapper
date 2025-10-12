import torch
import os
import shutil
import time
import cv2
import numpy as np
from typing import List, Dict, Tuple

from mapanything.models import MapAnything
from mapanything.utils.image import load_images, find_closest_aspect_ratio
from place_net_trt import PlaceNetTRT
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
        self.place_net = None
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
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.config.start_frame)

        end_frame = total_frames
        if self.config.max_frames != -1:
            end_frame = min(self.config.start_frame + self.config.max_frames, total_frames)

        expected_frames = end_frame - self.config.start_frame
        self.frame_buffer = [None] * expected_frames
        self.frame_pos_buffer = [0] * expected_frames
        self.frametimes_ns = [0] * expected_frames
        self.place_net_descriptors = [None] * expected_frames
        
        img_norm = IMAGE_NORMALIZATION_DICT["dinov2"]
        self.img_transform = tvf.Compose(
            [tvf.ToTensor(), tvf.Normalize(mean=img_norm.mean, std=img_norm.std)]
        )

        frame_count = 0
        invalid_cnt = 0
        for frame_idx in range(self.config.start_frame, end_frame):
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

            # run place net here
            self.frame_buffer[frame_count] = normalized_img
            self.frame_pos_buffer[frame_count] = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.frametimes_ns[frame_count] = int(self.cap.get(cv2.CAP_PROP_POS_MSEC) * MS_TO_NS)
            self.place_net_descriptors[frame_count] = self.place_net(img_rgb)

            frame_count += 1

        self.cap.release()
        
        actual_frames = frame_count
        self.frame_buffer = self.frame_buffer[:actual_frames]
        self.frame_pos_buffer = self.frame_pos_buffer[:actual_frames]
        self.frametimes_ns = self.frametimes_ns[:actual_frames]
        self.place_net_descriptors = self.place_net_descriptors[:actual_frames]
            
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
            self.place_net_descriptors = self.place_net_descriptors[first_id:last_id+1]
        
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
        
        print("Initializing PlaceNetTRT model...")
        self.place_net = PlaceNetTRT()
        self.place_net.to_device(torch.device("cuda"))
        print("PlaceNetTRT model initialized.")
        
        # Initialize matcher for keypoint matching and verification
        self.matcher = Matcher(device=self.device, keypoint_matching_threshold=self.config.keypoint_matching_threshold)
    
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
        """Return (reference_db, reference_vectors) from saved database in a backward-compatible way.
        - If db_data is a (list, tensor) tuple, return as-is
        - If db_data is a list of dicts, try to stack 'place_net_descriptor'; otherwise compute vectors from images
        """
        try:
            if isinstance(db_data, tuple) and len(db_data) == 2:
                reference_db, reference_vectors = db_data
                if not isinstance(reference_vectors, torch.Tensor):
                    reference_vectors = torch.tensor(reference_vectors)
                return reference_db, reference_vectors
            # Fallback: legacy format as list of entries
            if isinstance(db_data, list):
                reference_db = db_data
                if len(reference_db) > 0 and 'place_net_descriptor' in reference_db[0]:
                    vectors = []
                    for e in reference_db:
                        v = e['place_net_descriptor']
                        if isinstance(v, torch.Tensor):
                            vectors.append(v.squeeze().cpu())
                        else:
                            vectors.append(torch.tensor(v).squeeze())
                    reference_vectors = torch.stack(vectors, dim=0)
                    return reference_db, reference_vectors
                # Compute vectors from stored images
                vectors = []
                for e in reference_db:
                    p = e.get('image_path')
                    try:
                        img_bgr = cv2.imread(p)
                        if img_bgr is None:
                            vectors.append(torch.zeros(128))
                            continue
                        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                        vec = self.place_net(img_rgb)
                        vectors.append(torch.from_numpy(vec).squeeze())
                    except Exception:
                        vectors.append(torch.zeros(128))
                reference_vectors = torch.stack(vectors, dim=0)
                return reference_db, reference_vectors
        except Exception as e:
            print(f"Warning: Failed to unpack reference DB flexibly: {e}")
        # As a last resort, treat as empty
        return [], torch.empty(0)

    def _find_and_prepare_localization_frames(self, db_data):
        """
        Localize a query sub-sequence to the reference by:
        1) Comparing query place vectors to the first and last reference vectors to get start/end query indices
        2) Expanding by a small buffer
        3) Trimming internal query buffers to this window to save memory
        4) Returning the trimmed query frame identifiers (e.g., 'frame_0', ...)
        """
        # 1. Unpack database (supports legacy formats)
        reference_db, reference_vectors = self._unpack_reference_db(db_data)
        reference_vectors = reference_vectors.to(self.device)
        print(f"Loaded reference database with {len(reference_db)} entries.")

        # 2. Calculate place vectors for the query sequence (strided)
        print("Calculating place recognition vectors for the new sequence...")
        query_indices = [idx for idx in range(len(self.frame_buffer)) if idx % self.config.stride == 0]
        if not query_indices:
            return []
        query_vectors = [torch.from_numpy(self.place_net_descriptors[idx]).squeeze() for idx in query_indices]
        query_vectors = torch.stack(query_vectors).to(self.device)
        print(f"Calculated {len(query_vectors)} place vectors for the query video.")

        # 3. Find best matching query frames for reference first and last using place vectors + keypoint verification
        print("Finding best start frame match with keypoint verification...")
        print(f"Minimum required keypoint matches: {self.config.min_keypoint_matches}")
        sim_start = torch.cdist(query_vectors, reference_vectors[[0]])
        # Get top 5 candidates for start frame (sorted by place distance, best first)
        top_k = min(5, len(query_vectors))
        top_start_distances, top_start_indices = torch.topk(sim_start.squeeze(), top_k, largest=False)
        
        print(f"Testing {top_k} candidates sorted by place recognition score (best to worst):")
        
        # Verify with keypoint matching
        best_start_matches = 0
        start_query_vec_idx = None
        ref_start_img_path = reference_db[0]['image_path']
        
        for i, idx in enumerate(top_start_indices):
            query_vec_idx = idx.item()
            query_frame_idx = query_indices[query_vec_idx]
            query_img = self.frame_buffer[query_frame_idx]
            
            # Get detailed match info
            num_matches, query_img_np, ref_img_rgb, keypoints0, keypoints1, matching_scores = self.matcher.count_keypoint_matches(
                query_img, ref_start_img_path, return_details=True
            )
            
            meets_threshold = num_matches >= self.config.min_keypoint_matches
            status = "✓ VALID" if meets_threshold else "✗ BELOW THRESHOLD"
            print(f"  Candidate {i+1} (rank #{i+1}): place_dist={top_start_distances[i].item():.4f}, keypoint_matches={num_matches} {status}")
            
            # Only consider candidates with enough matches
            if meets_threshold and num_matches > best_start_matches:
                best_start_matches = num_matches
                start_query_vec_idx = query_vec_idx
        
        if start_query_vec_idx is None:
            print(f"ERROR: No start frame candidate met the minimum threshold of {self.config.min_keypoint_matches} matches!")
            return []
        
        start_query_frame_idx = query_indices[start_query_vec_idx]
        print(f"✓ Selected start frame {start_query_frame_idx} with {best_start_matches} keypoint matches")
        
        print("\nFinding best end frame match with keypoint verification...")
        sim_end = torch.cdist(query_vectors, reference_vectors[[-1]])
        # Get top 5 candidates for end frame (sorted by place distance, best first)
        top_end_distances, top_end_indices = torch.topk(sim_end.squeeze(), top_k, largest=False)
        
        print(f"Testing {top_k} candidates sorted by place recognition score (best to worst):")
        
        # Verify with keypoint matching
        best_end_matches = 0
        end_query_vec_idx = None
        ref_end_img_path = reference_db[-1]['image_path']
        
        for i, idx in enumerate(top_end_indices):
            query_vec_idx = idx.item()
            query_frame_idx = query_indices[query_vec_idx]
            query_img = self.frame_buffer[query_frame_idx]
            
            # Get detailed match info
            num_matches, query_img_np, ref_img_rgb, keypoints0, keypoints1, matching_scores = self.matcher.count_keypoint_matches(
                query_img, ref_end_img_path, return_details=True
            )
            
            meets_threshold = num_matches >= self.config.min_keypoint_matches
            status = "✓ VALID" if meets_threshold else "✗ BELOW THRESHOLD"
            print(f"  Candidate {i+1} (rank #{i+1}): place_dist={top_end_distances[i].item():.4f}, keypoint_matches={num_matches} {status}")
            
            # Only consider candidates with enough matches
            if meets_threshold and num_matches > best_end_matches:
                best_end_matches = num_matches
                end_query_vec_idx = query_vec_idx
        
        if end_query_vec_idx is None:
            print(f"ERROR: No end frame candidate met the minimum threshold of {self.config.min_keypoint_matches} matches!")
            return []
        
        end_query_frame_idx = query_indices[end_query_vec_idx]
        print(f"✓ Selected end frame {end_query_frame_idx} with {best_end_matches} keypoint matches")
        
        if start_query_frame_idx > end_query_frame_idx:
            start_query_frame_idx, end_query_frame_idx = end_query_frame_idx, start_query_frame_idx
        
        print("\n" + "="*70)
        print(f"FINAL LOCALIZATION RESULTS:")
        print(f"  ✓ Start frame: {start_query_frame_idx} with {best_start_matches} keypoint matches")
        print(f"  ✓ End frame:   {end_query_frame_idx} with {best_end_matches} keypoint matches")
        print(f"  Minimum threshold: {self.config.min_keypoint_matches} matches (both passed)")
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
        self.place_net_descriptors = self.place_net_descriptors[q_start:q_end]
        print(f"Query buffers trimmed to {len(self.frame_buffer)} frames.")

        # Build identifiers for the trimmed frames
        selected_query_paths = [f"frame_{i}" for i in range(len(self.frame_buffer))]
        print(f"Selected {len(selected_query_paths)} query frames for localization.")
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
            processed_image_paths = localization_frames
            print(f"Running localization on a pre-selected sequence of {len(processed_image_paths)} frames.")
        elif self.config.video_path:
            processed_image_paths = [f"frame_{i}" for i in range(len(self.frame_buffer))]
            processed_image_paths = processed_image_paths[::self.config.stride]
            print(f"Processing {len(self.frame_buffer)} frames from video. With stride={self.config.stride}, processing {len(processed_image_paths)} frames.")
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
            if relocalization_db:
                view_set, paths_in_view_set, first_overlap_pose, place_vectors_for_chunk = self._prepare_relocalization_views(current_chunk_paths, relocalization_db)
            else:
                view_set, paths_in_view_set, first_overlap_pose = self._prepare_sequential_views(current_chunk_paths, self.last_chunk_poses_for_fallback, self.last_chunk_depth_z_for_fallback)

            if not view_set or len(view_set) < 2:
                print("Not enough views in chunk to process.")
                continue

            # preprocess views
            ignore_keys = set(
                [
                    "instance",
                    "idx",
                    "true_shape",
                    "data_norm_type",
                    "place_net_descriptor",
                ]
            )
            for view in view_set:
                for name in view.keys():
                    if name in ignore_keys:
                        continue
                    view[name] = view[name].to(self.device, non_blocking=True)

            view_set_pre = preprocess_input_views_for_inference(view_set)

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
                confidence_percentile=50,
            )
            
            if first_overlap_pose is not None:
                new_first_pose = predictions[0]['camera_poses']
                transform = first_overlap_pose.to(self.device) @ torch.inverse(new_first_pose)
                predictions = self._apply_transform_to_chunk(predictions, transform)

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

            # Save chunk data and update database
            all_normals_for_chunk = [ground_normal] * len(query_predictions) if ground_normal is not None else [None] * len(query_predictions)
            all_plane_ds_for_chunk = [plane_d] * len(query_predictions) if plane_d is not None else [None] * len(query_predictions)
            save_chunk_with_interpolation(
                i, 
                chunk_start_idx, 
                chunk_end_idx, 
                processed_image_paths, 
                query_predictions, 
                query_paths, 
                all_normals_for_chunk, 
                all_plane_ds_for_chunk,
                self.config.output_dir,
                self.config.max_depth_for_point_cloud,
                self.config.stride,
                self.config.video_path,
                self.frametimes_ns,
                self.frame_pos_buffer
            )

            processed_chunk_data = self._process_chunk_and_update_db(
                query_predictions, query_paths, ground_normal, plane_d, place_vectors_for_chunk)
            database.extend(processed_chunk_data)

            # --- Update Last Chunk Poses for Next Overlap ---
            # Use original query paths as keys to preserve overlap identity across chunks
            self.last_chunk_poses_for_fallback.clear()
            self.last_chunk_depth_z_for_fallback.clear()
            for idx, entry in enumerate(processed_chunk_data):
                if idx < len(query_paths):
                    key = query_paths[idx]
                else:
                    key = entry["image_path"]
                self.last_chunk_poses_for_fallback[key] = entry["camera_poses"]
                self.last_chunk_depth_z_for_fallback[key] = entry["depth_z"]
                
            if self.config.create_debug_chunk_visualizations:
                save_debug_chunk_visualization(i, query_predictions, self.config.output_dir, self.config.max_depth_for_point_cloud)

        overall_proc_time = time.time() - overall_start_time
        
        # --- Trim Query Trajectory Based on Spatial Distance to Reference ---
        if relocalization_db is not None:
            print("\n--- Trimming Query Trajectory Based on Spatial Distance ---")
            database = self._trim_query_to_reference_coverage(database, relocalization_db)
        
        # --- Final Alignment and Database Save ---
        alignment_transform = None
        if relocalization_db is None:
            # Only run GPS/gravity alignment for standalone query reconstructions.
            if self.telemetry and self.gravity is not None:
                # 1. Align to Gravity
                gravity_transform = align_reconstruction_to_gravity(database, self.gravity, self.frametimes_ns, self.device)
                
                if gravity_transform is not None:
                    # Create a temporary database aligned to gravity for subsequent alignments
                    db_grav_aligned = [{'camera_poses': gravity_transform.cpu() @ entry['camera_poses'], 'image_path': entry['image_path']} for entry in database]

                    # 2. Align to GPS direction in the XY plane (zero_z=True)
                    gps_xy_transform = align_reconstruction_to_gps_direction(db_grav_aligned, self.gps_ned, self.frametimes_ns, True, self.device)

                    if gps_xy_transform is not None:
                        # Combine gravity and XY alignment
                        alignment_transform = gps_xy_transform @ gravity_transform
                        
                        # Create a temporary database also aligned in the XY plane for the final step
                        db_xy_aligned = [{'camera_poses': gps_xy_transform.cpu() @ entry['camera_poses'], 'image_path': entry['image_path']} for entry in db_grav_aligned]

                        # 3. Perform final full 3D alignment (zero_z=False)
                        final_gps_transform = align_reconstruction_to_gps_direction(db_xy_aligned, self.gps_ned, self.frametimes_ns, False, self.device)

                        if final_gps_transform is not None:
                            # Combine all transforms: T_final = T_gps_full @ T_gps_xy @ T_gravity
                            alignment_transform = final_gps_transform @ alignment_transform
                    else:
                        # Fallback to just gravity alignment if GPS XY fails
                        alignment_transform = gravity_transform
            else:
                # Fallback for sequences without telemetry
                print("No telemetry or gravity data available, attempting simple GPS direction alignment.")
                alignment_transform = align_reconstruction_to_gps_direction(database, self.gps_ned, self.frametimes_ns, True, self.device)

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
        
        reference_db, _ = relocalization_db_data
        
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
                ref_position = ref_pose[:3, 3]
            else:
                ref_position = torch.tensor(ref_pose[:3, 3])
            
            distance = torch.norm(query_position - ref_position).item()
            
            if distance < best_distance:
                best_distance = distance
                best_ref_idx = ref_idx
        
        return best_ref_idx, best_distance
    
    def _prepare_relocalization_views(
        self, current_chunk_paths: List[str], db_data: Tuple[List[Dict], torch.Tensor]
    ) -> Tuple[List[Dict], List[str], torch.Tensor, torch.Tensor]:
        """
        Prepares views by adding reference priors based on spatial proximity.
        
        Strategy:
        - First chunk: Add first N reference frames as priors (assumes query starts near beginning of reference)
        - Subsequent chunks: Use last query pose to find closest reference, add N consecutive reference frames if within threshold
        
        Returns:
            Tuple of (view_set, paths_in_view_set, first_overlap_pose, current_vectors)
        """
        reference_db, reference_vectors = db_data
        view_set = []
        paths_in_view_set = []
        first_overlap_pose = None
        current_vectors = None
        
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
        
        # ========== FIRST CHUNK: Add First N Reference Frames ==========
        if is_first_chunk:
            print(f"[Chunk 1] Adding first {self.config.num_reference_priors} reference frames as priors...")
            
            # Add first N consecutive reference frames as priors
            num_priors_to_add = min(self.config.num_reference_priors, len(reference_db))
            for i in range(num_priors_to_add):
                db_entry = reference_db[i]
                db_view = load_images([db_entry['image_path']], resize_mode="fixed_size", size=self.target_size)[0]
                ref_intrinsics = db_entry.get('intrinsics', self.intrinsics)
                if not isinstance(ref_intrinsics, torch.Tensor):
                    ref_intrinsics = torch.tensor(ref_intrinsics, dtype=torch.float32)
                db_view.update({
                    "intrinsics": ref_intrinsics,
                    "camera_poses": db_entry['camera_poses'].to(self.device)
                })
                view_set.append(db_view)
                paths_in_view_set.append(db_entry['image_path'])
                
                if i == 0:
                    first_overlap_pose = db_entry['camera_poses'].to(self.device)
            
            # Store reference index for next chunk
            self.current_ref_idx = num_priors_to_add - 1
            print(f"  ✓ Added reference frames 0 to {num_priors_to_add - 1}")
        
        # ========== SUBSEQUENT CHUNKS: Use Spatial Distance Matching ==========
        else:
            print(f"[Chunk N] Using spatial distance to find closest reference...")
            
            # Get last query pose from previous chunk
            last_query_path = list(self.last_chunk_poses_for_fallback.keys())[-1]
            last_query_pose = self.last_chunk_poses_for_fallback[last_query_path]
            
            # Find closest reference pose across ALL reference frames
            ref_idx, distance = self._find_closest_reference_by_spatial_distance(
                last_query_pose, reference_db, search_start_idx=None, search_window=None
            )
            
            print(f"  Closest reference frame: {ref_idx} at distance {distance:.3f}m")
            
            # If within threshold: add reference priors
            if distance <= self.config.spatial_distance_threshold_m:
                print(f"  ✓ Within threshold ({self.config.spatial_distance_threshold_m}m), adding {self.config.num_reference_priors} reference priors")
                
                # Add N consecutive reference frames as priors
                num_priors_to_add = min(self.config.num_reference_priors, len(reference_db) - ref_idx)
                for i in range(num_priors_to_add):
                    idx = ref_idx + i
                    db_entry = reference_db[idx]
                    db_view = load_images([db_entry['image_path']], resize_mode="fixed_size", size=self.target_size)[0]
                    ref_intrinsics = db_entry.get('intrinsics', self.intrinsics)
                    if not isinstance(ref_intrinsics, torch.Tensor):
                        ref_intrinsics = torch.tensor(ref_intrinsics, dtype=torch.float32)
                    db_view.update({
                        "intrinsics": ref_intrinsics,
                        "camera_poses": db_entry['camera_poses'].to(self.device)
                    })
                    view_set.append(db_view)
                    paths_in_view_set.append(db_entry['image_path'])
                    
                    if i == 0:
                        first_overlap_pose = db_entry['camera_poses'].to(self.device)
                
                self.current_ref_idx = ref_idx + num_priors_to_add - 1
                print(f"  Added reference frames {ref_idx} to {ref_idx + num_priors_to_add - 1}")
            else:
                print(f"  ✗ Distance {distance:.3f}m exceeds threshold, continuing sequential reconstruction")
        
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
            
            # If no reference priors were found, use first sequential overlap for alignment
            if first_overlap_pose is None and overlapping_sequential_paths:
                first_overlap_pose = self.last_chunk_poses_for_fallback[overlapping_sequential_paths[0]]
                print(f"  Using first sequential overlap as alignment anchor")
        
        # ========== Add New (Unposed) Query Views ==========
        new_query_paths = [p for p in current_chunk_paths if p not in paths_in_view_set]
        for path in new_query_paths:
            original_idx = current_chunk_paths.index(path)
            view = current_views[original_idx]
            view["intrinsics"] = self.intrinsics
            view_set.append(view)
        paths_in_view_set.extend(new_query_paths)
        
        print(f"  Total views in set: {len(view_set)} ({len(view_set) - len(new_query_paths)} priors, {len(new_query_paths)} new)")
        
        return view_set, paths_in_view_set, first_overlap_pose, current_vectors

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
                place_net_descriptor = self.place_net_descriptors[frame_idx]
                
                view = {
                    'img': img_tensor[None], # Add batch dim
                    'true_shape': np.int32([img_tensor.shape[1], img_tensor.shape[2]]),
                    'idx': frame_idx,
                    'instance': str(frame_idx),
                    'data_norm_type': ['dinov2'],
                    'place_net_descriptor': place_net_descriptor,
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
            if place_vectors is not None and i < len(place_vectors):
                entry["place_net_descriptor"] = place_vectors[i].cpu()
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
            # Persist the RGB frame used for reconstruction and repoint image_path to it
            try:
                if images_dir is not None and 'img_no_norm' in pred:
                    # Derive a stable filename from the original path identifier
                    base = os.path.basename(paths[i])
                    name, _sep, _ext = base.partition('.')
                    if not name:
                        name = base
                    filename = f"{name}.jpg"
                    save_path = os.path.join(images_dir, filename)

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
                    if not os.path.exists(save_path):
                        cv2.imwrite(save_path, cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))

                    # Update entry to point to the saved RGB frame
                    entry["image_path"] = save_path
            except Exception as e:
                print(f"Warning: Failed to save/reference RGB frame for database entry {paths[i]}: {e}")
            
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
        
        # --- Prepare data for saving ---
        # Separate place vectors from the main database if they exist
        main_db_for_saving = []
        place_vectors_for_saving = []
        has_place_vectors = 'place_net_descriptor' in database[0] if database else False
        
        if has_place_vectors:
            print("Separating place recognition vectors for optimized loading.")
            for entry in database:
                main_db_for_saving.append({k: v for k, v in entry.items() if k != 'place_net_descriptor'})
                place_vectors_for_saving.append(entry['place_net_descriptor'])
            
            # Stack vectors into a single tensor
            place_vectors_tensor = torch.stack(place_vectors_for_saving, dim=0)
            
            # Save the main database and the vectors as a tuple
            save_data = (main_db_for_saving, place_vectors_tensor)
        else:
            # Save only the main database if no vectors are present
            save_data = database

        try:
            os.makedirs(os.path.dirname(self.config.database_path), exist_ok=True)
            torch.save(save_data, self.config.database_path)
            print(f"Saved SLAM database with {len(database)} entries to {self.config.database_path}")
        except Exception as e:
            print(f"Error saving database: {e}")

