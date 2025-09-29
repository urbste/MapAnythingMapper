import torch
import os
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
from scipy.spatial.transform import Rotation, Slerp
from mapanything.utils.inference import preprocess_input_views_for_inference, postprocess_model_outputs_for_inference
from utils.telemetry_converter import TelemetryImporter
from utils.undistortion import create_undistortion_maps_from_file
from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT
import torchvision.transforms as tvf
from scipy.spatial.transform import Rotation as RS
import glob
from slam.debug_utils import save_localization_start_end_debug, save_query_db_match_debug
from slam.trajectory_utils import save_query_trajectory_and_normals
from slam.alignment_utils import align_reconstruction_to_gravity, align_reconstruction_to_gps_direction
from slam.geo_utils import ecef_to_ned
from utils.math import scale_intrinsics
from slam.plane_utils import estimate_ground_plane

MS_TO_NS = 1_000_000

class VisualSLAM:
    def __init__(self, config: SLAMConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._setup_paths()
        self.map_anything = None
        self.place_net = None
        self.intrinsics = self._setup_intrinsics()
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
        if self.config.output_dir is None or self.config.database_path is None:
            results_dir = os.path.join(os.path.dirname(results_dir_base), "_slam_results")
            os.makedirs(results_dir, exist_ok=True)
            
            if self.config.output_dir is None:
                self.config.output_dir = os.path.join(results_dir, "chunks")
            
            if self.config.database_path is None:
                db_name = os.path.basename(os.path.normpath(results_dir_base)) + "_db.pt"
                self.config.database_path = os.path.join(results_dir, db_name)
        
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Setup database path, making it absolute if it's relative
        if not os.path.isabs(self.config.database_path):
            if self.config.database_path is None:
                self.config.database_path = os.path.join(results_dir, "slam_database.pt")
            else:
                db_name = os.path.basename(self.config.database_path)
                self.config.database_path = os.path.join(results_dir, db_name)
        
        # Ensure the directory for the database exists
        db_dir = os.path.dirname(self.config.database_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        print(f"Using output directory: {self.config.output_dir}")
        print(f"Using database path: {self.config.database_path}")

    def _cache_frames(self):
        """Cache video frames and associated telemetry data with memory management."""
        print("Start caching frames.")
        
        if self.config.cam_dist_path:
            print("Setting up undistortion maps.")
            self.undistortion_maps = create_undistortion_maps_from_file(self.config.cam_dist_path)

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
            R_c_i = RS.from_quat([
                -0.005614731940060652,
                0.7154606008027313,
                -0.6985867170157808,
                0.00782318946143889]).as_matrix()

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
        org_width, org_height = self.config.org_image_size
        fx, fy, cx, cy = self.config.base_intrinsics
        
        average_aspect_ratio = org_width / org_height
        target_width, target_height = find_closest_aspect_ratio(average_aspect_ratio, self.config.resolution_set)
        self.target_size = (target_width, target_height)

        scale_x = target_width / org_width
        scale_y = target_height / org_height

        intrinsics_tensor = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32, device=self.device)
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
        #self.map_anything = torch.compile(self.map_anything, backend="tensorrt")
        
        print("Initializing PlaceNetTRT model...")
        self.place_net = PlaceNetTRT()
        self.place_net.to_device(torch.device("cuda"))
        print("PlaceNetTRT model initialized.")
    
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

        # 3. Find best matching query frames for reference first and last
        sim_start = torch.cdist(query_vectors, reference_vectors[[0]])
        sim_end = torch.cdist(query_vectors, reference_vectors[[-1]])
        start_query_vec_idx = torch.argmin(sim_start).item()
        end_query_vec_idx = torch.argmin(sim_end).item()
        start_query_frame_idx = query_indices[start_query_vec_idx]
        end_query_frame_idx = query_indices[end_query_vec_idx]
        if start_query_frame_idx > end_query_frame_idx:
            start_query_frame_idx, end_query_frame_idx = end_query_frame_idx, start_query_frame_idx
        print(f"Best query frame indices: start={start_query_frame_idx}, end={end_query_frame_idx}")

        # Save debug preview of the matched start/end pairs (query vs. reference)
        try:
            if getattr(self.config, 'create_debug_matches', False):
                reference_db, _reference_vectors = self._unpack_reference_db(db_data)
                save_localization_start_end_debug(self.config.database_path, reference_db, self.frame_buffer, start_query_frame_idx, end_query_frame_idx)
        except Exception as e:
            print(f"Warning: Could not save start/end relocalization debug previews: {e}")

        # 4. Add buffer around start/end and trim buffers
        overlap = self.config.relocalization_frame_overlap
        q_start = max(0, start_query_frame_idx - overlap)
        q_end = min(len(self.frame_buffer), end_query_frame_idx + overlap + 1)
        print(f"Trimming query to [{q_start}, {q_end}) with overlap {overlap}.")
        if q_start >= q_end:
            print("Warning: Computed empty query window. Aborting localization selection.")
            return []

        # Optional GPS gate to ensure rough consistency between query ends and reference ends
        if self.telemetry and hasattr(self, 'gps_ned') and self.gps_ned is not None and len(reference_db) >= 2:
            try:
                qs_tns = self.frametimes_ns[q_start]
                qe_tns = self.frametimes_ns[q_end - 1]
                if qs_tns in self.gps_ned and qe_tns in self.gps_ned:
                    q_start_ned = np.array(self.gps_ned[qs_tns])
                    q_end_ned = np.array(self.gps_ned[qe_tns])
                    r_start_ned = reference_db[0].get('gps_ned', None)
                    r_end_ned = reference_db[-1].get('gps_ned', None)
                    if r_start_ned is not None and r_end_ned is not None:
                        r_start_ned = r_start_ned.cpu().numpy() if isinstance(r_start_ned, torch.Tensor) else np.array(r_start_ned)
                        r_end_ned = r_end_ned.cpu().numpy() if isinstance(r_end_ned, torch.Tensor) else np.array(r_end_ned)
                        dist_start = np.linalg.norm(q_start_ned - r_start_ned)
                        dist_end = np.linalg.norm(q_end_ned - r_end_ned)
                        max_allowed = self.config.relocalization_gps_max_pair_distance_m
                        if dist_start > max_allowed or dist_end > max_allowed:
                            print(f"GPS gating failed: start {dist_start:.2f}m, end {dist_end:.2f}m exceed {max_allowed}m.")
                            return []
            except Exception as e:
                print(f"Warning: GPS gating failed due to error: {e}. Proceeding without GPS gate.")

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
            self._save_chunk_with_interpolation(i, chunk_start_idx, chunk_end_idx, processed_image_paths, query_predictions, query_paths, all_normals_for_chunk, all_plane_ds_for_chunk)

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
                if "depth_z" in entry:
                    self.last_chunk_depth_z_for_fallback[key] = entry["depth_z"]
                self.last_chunk_depth_z_for_fallback[key] = entry["depth_z"]
                
            if self.config.create_debug_chunk_visualizations:
                self._save_debug_chunk_visualization(i, query_predictions)

        overall_proc_time = time.time() - overall_start_time
        
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
        
        # Always save a trajectory-only PLY for the query reconstruction
        try:
            save_query_trajectory_and_normals(database, self.config.database_path)
        except Exception as e:
            print(f"Warning: Could not save trajectory/normals PLY: {e}")

        if relocalization_db is None and alignment_transform is not None:
            # Save the final transform for the point cloud processor and align chunks
            alignment_path = os.path.join(os.path.dirname(self.config.database_path), "alignment.pt")
            torch.save(alignment_transform, alignment_path)
            print(f"Saved final alignment transform to {alignment_path}")
            self._save_chunks_for_visualization(database, alignment_transform)


        print("\n--- Total Execution Summary ---")
        print(f"Processed {len(database)} frames in {overall_proc_time:.2f} seconds.")

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

    def _prepare_relocalization_views(
        self, current_chunk_paths: List[str], db_data: Tuple[List[Dict], torch.Tensor]
    ) -> Tuple[List[Dict], List[str], torch.Tensor, torch.Tensor]:
        """Prepares views by finding matches in an existing database."""
        reference_db, reference_vectors = db_data
        view_set, paths_in_view_set = [], []
        first_overlap_pose = None
        
        print(f"Finding relocalization priors for the current chunk...")

        # 1. Load current chunk images and get their place vectors
        if self.config.video_path:
            current_views = self._get_views_from_buffer(current_chunk_paths)
        else:
            current_views = load_images(
                current_chunk_paths, 
                resize_mode="fixed_size", 
                size=self.target_size,
            )
        
        # Reuse precomputed place vectors for video; compute only for image lists
        if self.config.video_path:
            current_vectors = []
            for path in current_chunk_paths:
                base = os.path.basename(path)
                name, _sep, _ext = base.partition('.')
                digits = ''.join([c for c in name if c.isdigit()])
                frame_idx = int(digits)
                current_vectors.append(torch.from_numpy(self.place_net_descriptors[frame_idx]).squeeze())
            current_vectors = torch.stack(current_vectors)
        else:
            current_vectors = []
            for view in current_views:
                img_for_place_net = rgb(view['img'], view['data_norm_type'][0])[0] * 255
                vector = torch.tensor(self.place_net(img_for_place_net).squeeze())
                current_vectors.append(vector.squeeze())
            current_vectors = torch.stack(current_vectors)

        # 2. Find best matches in the database using L2 distance
        similarities = torch.cdist(current_vectors, reference_vectors.cpu())
        best_match_indices_for_all = torch.argmin(similarities, dim=1)
        best_match_distances = similarities[torch.arange(similarities.shape[0]), best_match_indices_for_all]
        
        # 3. Filter matches based on thresholds
        min_distance = best_match_distances.min() 
        print("Min distance:", min_distance)
        relative_threshold = min_distance * (1 + self.config.relocalization_threshold_percent / 100)
        absolute_threshold = self.config.max_relocalization_threshold
        
        valid_indices_mask = (best_match_distances < relative_threshold) & (best_match_distances < absolute_threshold)
        
        valid_query_indices = torch.where(valid_indices_mask)[0].tolist()
        
        print(f"Found {len(valid_query_indices)} potential high-quality matches for this chunk.")

        # 4. Strategically select the first and last valid matches to use as priors
        priors_to_use = []
        if len(valid_query_indices) > 0:
            first_match_idx = valid_query_indices[0]
            last_match_idx = valid_query_indices[-1]
            
            priors_to_use.append(first_match_idx)
            if first_match_idx != last_match_idx:
                priors_to_use.append(last_match_idx)
        
        # 5. Add the selected database entries as priors
        added_db_indices = set()
        if priors_to_use:
            print(f"Selected {len(priors_to_use)} priors (first and last valid matches) for alignment.")
            for query_idx in priors_to_use:
                db_idx = best_match_indices_for_all[query_idx].item()
                if db_idx not in added_db_indices:
                    db_entry = reference_db[db_idx]
                    db_image_path = db_entry['image_path']
                    db_pose = db_entry['camera_poses'].to(self.device)
                    
                    if self.config.video_path:
                        # This part is tricky, relocalization with video requires a way to map db paths to frames
                        # For now, we assume db paths are image files and load them.
                        # A more robust solution would be needed for pure video-to-video relocalization.
                        print(f"Warning: Relocalization from a video sequence against a database with file paths ({db_image_path}).")
                    
                    db_view = load_images([db_image_path], resize_mode="fixed_size", size=self.target_size)[0]
                    # Prefer stored intrinsics from the reference DB if present
                    ref_intrinsics = db_entry.get('intrinsics', None)
                    if ref_intrinsics is not None:
                        if isinstance(ref_intrinsics, torch.Tensor):
                            intr = ref_intrinsics
                        else:
                            intr = torch.tensor(ref_intrinsics, dtype=torch.float32)
                    else:
                        intr = self.intrinsics
                    db_view.update({"intrinsics": intr, "camera_poses": db_pose})
                    
                    if first_overlap_pose is None:
                        first_overlap_pose = db_pose
                    
                    view_set.append(db_view)
                    paths_in_view_set.append(db_image_path)
                    added_db_indices.add(db_idx)
                    
                    # Create debug image for the selected priors
                    if self.config.create_debug_matches:
                        query_image_id = current_chunk_paths[query_idx]
                        save_query_db_match_debug(self.config.database_path, self.frame_buffer, 
                            self.chunk_idx_for_debug, query_idx, query_image_id, db_image_path)
        else:
            print("No valid relocalization matches found for this chunk. Proceeding with sequential alignment.")
            # Fallback to sequential alignment if no matches are found
            view_set, paths_in_view_set, first_overlap_pose = self._prepare_sequential_views(current_chunk_paths, self.last_chunk_poses_for_fallback, self.last_chunk_depth_z_for_fallback)
            return view_set, paths_in_view_set, first_overlap_pose, current_vectors
            
        # If no valid matches are found, we'll try to align sequentially
        if not valid_query_indices:
            print("No valid relocalization matches found for this chunk. Will attempt sequential alignment.")
        else:
             # Use the pose of the very first valid database match as the anchor for alignment
            first_db_pose = view_set[0]["camera_poses"]
            first_overlap_pose = first_db_pose.to(self.device)
            print(f"Added {len(view_set)} unique database views as priors.")

        # --- 2. Add Priors from Previous Query Chunk (Sequential Overlap) ---
        overlapping_sequential_paths = [
            p for p in current_chunk_paths 
            if p in self.last_chunk_poses_for_fallback and p not in paths_in_view_set
        ]

        if overlapping_sequential_paths:
            print(f"Adding {len(overlapping_sequential_paths)} sequential views from previous query chunk as priors.")
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
            
            # If no global DB priors were found, use the first sequential one for alignment.
            if first_overlap_pose is None:
                print("Using first sequential overlap view as alignment anchor.")
                first_overlap_pose = self.last_chunk_poses_for_fallback[overlapping_sequential_paths[0]]

        # --- 3. Add all new (unposed) views from the current chunk ---
        new_query_paths = [p for p in current_chunk_paths if p not in paths_in_view_set]
        
        # We need to find the original views for these paths to add them
        for path in new_query_paths:
            original_idx = current_chunk_paths.index(path)
            view = current_views[original_idx]
            view["intrinsics"] = self.intrinsics
            view_set.append(view)
        paths_in_view_set.extend(new_query_paths)
            
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

    def _get_up_vector_residuals(self, aligned_database: List[Dict]) -> float:
        """Calculates the average angular residual between measured gravity and the world's up-vector after alignment."""
        all_residuals_deg = []
        grav_dir_w_target = np.array([0, 0, -1])

        for entry in aligned_database:
            path = entry.get('image_path')
            if not path:
                continue
            try:
                base = os.path.basename(path)
                name, _sep, _ext = base.partition('.')
                digits = ''.join([c for c in name if c.isdigit()])
                frame_idx = int(digits)
                if frame_idx >= len(self.frametimes_ns):
                    continue
                tns = self.frametimes_ns[frame_idx]

                if tns in self.gravity:
                    g_c_measured = np.array(self.gravity[tns])

                    # Final aligned pose T_w_c
                    T_w_c = entry['camera_poses'].cpu().numpy()
                    R_w_c = T_w_c[0, :3, :3]

                    g_c_aligned = R_w_c.T @ grav_dir_w_target

                    # Calculate the difference in the z-component
                    all_residuals_deg.append(np.sqrt((g_c_measured[2] - g_c_aligned[2])**2))

            except (ValueError, KeyError, IndexError) as e:
                continue

        if not all_residuals_deg:
            return -1.0

        return np.mean(all_residuals_deg)

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

    def _save_chunks_for_visualization(self, database, alignment_transform):
        """Saves the point clouds for each frame for later visualization."""
        print("Saving point clouds for visualization...")
        chunk_dirs = sorted(glob.glob(os.path.join(self.config.output_dir, "chunk_*")))
        transform_np = alignment_transform.cpu().numpy() if alignment_transform is not None else np.eye(4)

        for chunk_dir in chunk_dirs:
            try:
                metadata_file = os.path.join(chunk_dir, "metadata.pt")
                pcd_file = os.path.join(chunk_dir, "point_cloud.npz")

                if not (os.path.exists(metadata_file) and os.path.exists(pcd_file)):
                    continue
                
                # Load data
                metadata = torch.load(metadata_file, map_location="cpu", weights_only=False)
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

    def _save_chunk(self, chunk_idx, predictions, all_poses, all_paths, is_reconstructed_flags, all_normals, all_plane_ds, frametimes_ns=None, frame_pos=None):
        """Saves the processed data for a chunk to a .pt file."""
        if not predictions:
            return

        # Points, colors, and confidences are only from the *reconstructed* views
        all_pts3d = torch.cat([p['pts3d'].reshape(-1, 3) for p in predictions], dim=0)
        all_pts3d_cam = torch.cat([p['pts3d_cam'].reshape(-1, 3) for p in predictions], dim=0)
        all_colors = torch.cat([p['img_no_norm'].reshape(-1, 3) for p in predictions], dim=0)
        all_masks = torch.cat([p['mask'].reshape(-1) for p in predictions], dim=0)

        # Apply masks to filter points before saving
        all_pts3d = all_pts3d[all_masks]
        all_pts3d_cam = all_pts3d_cam[all_masks]
        all_colors = all_colors[all_masks]

        # Apply depth filter
        depth_mask = all_pts3d_cam[:, 2] < self.config.max_depth_for_point_cloud
        all_pts3d = all_pts3d[depth_mask]
        all_colors = all_colors[depth_mask]
        
        chunk_dir = os.path.join(self.config.output_dir, f"chunk_{chunk_idx:04d}")
        os.makedirs(chunk_dir, exist_ok=True)
        
        # Save point cloud data as a compressed numpy array for efficiency
        pcd_path = os.path.join(chunk_dir, "point_cloud.npz")
        np.savez_compressed(
            pcd_path,
            pts3d=all_pts3d.cpu().numpy(),
            colors=(all_colors * 255).byte().cpu().numpy()
        )

        # Save metadata separately
        metadata = {
            'camera_poses': [p.cpu() for p in all_poses],
            'is_reconstructed': is_reconstructed_flags,
            'ground_normals': all_normals,
            'plane_ds': all_plane_ds,
        }
        if not self.config.video_path:
            metadata['image_paths'] = all_paths
        if frametimes_ns:
            metadata['frametimes_ns'] = frametimes_ns
        if frame_pos:
            metadata['frame_pos'] = frame_pos
            
        torch.save(metadata, os.path.join(chunk_dir, "metadata.pt"))
        print(f"Saved chunk data for chunk {chunk_idx} to {chunk_dir}")

    def _save_chunk_with_interpolation(self, chunk_idx, chunk_start_idx, chunk_end_idx, all_processed_paths, predictions, pred_paths, all_normals, all_plane_ds):
        """
        Handles pose interpolation if a stride is used and saves the complete data for a chunk.
        """
        # --- Handle Pose Interpolation ---
        if self.config.stride > 1 and len(pred_paths) > 1:
            full_chunk_paths = all_processed_paths[chunk_start_idx:chunk_end_idx]
            key_frame_poses = [p['camera_poses'][0].cpu().numpy() for p in predictions]
            
            try:
                key_frame_indices = [full_chunk_paths.index(p) for p in pred_paths]
            except ValueError:
                # Fallback if a path isn't found, though this shouldn't happen
                poses_to_save = [p['camera_poses'] for p in predictions]
                paths_to_save = pred_paths
                is_reconstructed = [True] * len(predictions)
            else:
                slerp = Slerp(key_frame_indices, Rotation.from_matrix([p[:3, :3] for p in key_frame_poses]))
                key_translations = np.array([p[:3, 3] for p in key_frame_poses])
                
                min_idx, max_idx = min(key_frame_indices), max(key_frame_indices)
                interp_indices = np.arange(min_idx, max_idx + 1)
                
                interp_rotations = slerp(interp_indices).as_matrix()
                interp_translations = np.array([np.interp(interp_indices, key_frame_indices, key_translations[:, i]) for i in range(3)]).T
                
                interpolated_poses_map = {}
                for i, frame_idx in enumerate(interp_indices):
                    pose = np.eye(4)
                    pose[:3, :3] = interp_rotations[i]
                    pose[:3, 3] = interp_translations[i]
                    interpolated_poses_map[frame_idx] = torch.from_numpy(pose).float().unsqueeze(0)

                poses_to_save, paths_to_save, is_reconstructed = [], [], []
                for i, path in enumerate(full_chunk_paths):
                    if min_idx <= i <= max_idx:
                        poses_to_save.append(interpolated_poses_map[i])
                        paths_to_save.append(path)
                        is_reconstructed.append(i in key_frame_indices)
        else:
            poses_to_save = [p['camera_poses'] for p in predictions]
            paths_to_save = pred_paths
            is_reconstructed = [True] * len(predictions)
            
        frametimes_ns_to_save = []
        frame_pos_to_save = []
        if self.config.video_path:
            # When processing a video, paths are identifiers like "frame_123"
            # We extract the index from the identifier to get the real metadata.
            for path in paths_to_save:
                try:
                    base = os.path.basename(path)
                    name, _sep, _ext = base.partition('.')
                    digits = ''.join([c for c in name if c.isdigit()])
                    frame_idx = int(digits)
                    original_list_idx = all_processed_paths.index(path)
                    frametimes_ns_to_save.append(self.frametimes_ns[original_list_idx])
                    frame_pos_to_save.append(self.frame_pos_buffer[original_list_idx])
                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not parse frame index from path '{path}'. Skipping metadata. Error: {e}")

        # Note: Normals are not interpolated, they are saved per-chunk and associated with the keyframes.
        # This logic assumes normals are constant for the reconstructed part of the chunk.
        self._save_chunk(chunk_idx, predictions, poses_to_save, paths_to_save, is_reconstructed, all_normals, all_plane_ds, frametimes_ns_to_save, frame_pos_to_save)

    def _save_debug_chunk_visualization(self, chunk_idx, predictions):
        """Saves the point cloud and trajectory for a single chunk for debugging."""
        if not predictions:
            return

        debug_dir = os.path.join(self.config.output_dir, "debug_chunks")
        os.makedirs(debug_dir, exist_ok=True)

        # --- Create Point Cloud ---
        all_pts3d = torch.cat([p['pts3d'].reshape(-1, 3) for p in predictions], dim=0)
        all_pts3d_cam = torch.cat([p['pts3d_cam'].reshape(-1, 3) for p in predictions], dim=0)
        all_colors = torch.cat([p['img_no_norm'].reshape(-1, 3) for p in predictions], dim=0)
        all_masks = torch.cat([p['mask'].reshape(-1) for p in predictions], dim=0)

        # Apply confidence filter
        all_pts3d = all_pts3d[all_masks]
        all_pts3d_cam = all_pts3d_cam[all_masks]
        all_colors = all_colors[all_masks]

        # Apply depth filter (using world Z coordinate)
        depth_mask = all_pts3d_cam[:, 2] < self.config.max_depth_for_point_cloud
        all_pts3d = all_pts3d[depth_mask]
        all_colors = all_colors[depth_mask]

        if all_pts3d.shape[0] == 0:
            print(f"Debug chunk {chunk_idx}: No points left after filtering.")
            return

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_pts3d.cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(all_colors.cpu().numpy())
        
        pcd_path = os.path.join(debug_dir, f"debug_chunk_{chunk_idx:04d}_pcd.ply")
        o3d.io.write_point_cloud(pcd_path, pcd)

        # --- Estimate and Visualize Ground Normal ---
        try:
            plane_model, inliers = pcd.segment_plane(distance_threshold=0.1, ransac_n=3, num_iterations=1000)
            [a, b, c, d] = plane_model
            ground_normal = np.array([a, b, c])
            
            # Ensure the normal points upwards (positive Y)
            if ground_normal[1] < 0:
                ground_normal = -ground_normal

            print(f"Debug chunk {chunk_idx}: Estimated ground normal: {ground_normal}")

            # Get a point on the plane to draw the normal from (using the pcd center)
            center_point = pcd.get_center()
            
            # Create a line representing the normal
            normal_line = o3d.geometry.LineSet()
            normal_line.points = o3d.utility.Vector3dVector([center_point, center_point + ground_normal * 2.0]) # 2-meter long line
            normal_line.lines = o3d.utility.Vector2iVector([[0, 1]])
            normal_line.colors = o3d.utility.Vector3dVector([[1, 0, 1]]) # Magenta

            normal_path = os.path.join(debug_dir, f"debug_chunk_{chunk_idx:04d}_normal.ply")
            o3d.io.write_line_set(normal_path, normal_line)

        except Exception as e:
            print(f"Debug chunk {chunk_idx}: Could not estimate ground normal. {e}")

        # --- Create Trajectory ---
        camera_poses = [p['camera_poses'][0].cpu().numpy() for p in predictions]
        camera_positions = [pose[:3, 3] for pose in camera_poses]
        lines = [[i, i + 1] for i in range(len(camera_positions) - 1)]
        
        if not lines:
            print(f"Debug chunk {chunk_idx}: Not enough poses for a trajectory.")
            return

        # Save trajectory as a dense point cloud PLY for easy visualization
        points_on_lines = []
        for i in range(len(camera_positions) - 1):
            start = camera_positions[i]
            end = camera_positions[i+1]
            # Add 100 points per line segment
            points_on_lines.extend(np.linspace(start, end, 100))
        
        if not points_on_lines:
            return
            
        traj_pcd = o3d.geometry.PointCloud()
        traj_pcd.points = o3d.utility.Vector3dVector(points_on_lines)
        traj_pcd.paint_uniform_color([1, 0, 0]) # Red

        traj_path = os.path.join(debug_dir, f"debug_chunk_{chunk_idx:04d}_traj.ply")
        o3d.io.write_point_cloud(traj_path, traj_pcd)

        print(f"Saved debug visualization for chunk {chunk_idx} to {debug_dir}")

    def _save_query_trajectory_and_normals(self, database: List[Dict]):
        """
        Save a trajectory-only PLY for the current run and an auxiliary normals PLY.
        Trajectory is saved as a dense point cloud along the camera path for viewer compatibility.
        Normals are visualized as short colored line segments sampled into points and saved as PLY.
        """
        if not database:
            print("No database entries to save trajectory.")
            return

        try:
            # Extract camera positions and optional normals
            camera_positions = []
            per_frame_normals = []
            for entry in database:
                pose = entry['camera_poses']
                if isinstance(pose, torch.Tensor):
                    pose_np = pose[0].detach().cpu().numpy()
                else:
                    pose_np = pose[0]
                camera_positions.append(pose_np[:3, 3])
                nrm = entry.get('ground_normal', None)
                if isinstance(nrm, torch.Tensor):
                    nrm = nrm.detach().cpu().numpy()
                per_frame_normals.append(nrm)

            if len(camera_positions) < 2:
                print("Not enough poses to form a trajectory.")
                return

            # Determine normal line length from median step size
            steps = [np.linalg.norm(camera_positions[i+1] - camera_positions[i]) for i in range(len(camera_positions)-1)]
            median_step = np.median(steps) if steps else 1.0
            normal_length = float(max(0.25 * median_step, 0.1))

            # Build dense point cloud along the trajectory lines
            num_points_per_segment = 100
            traj_pts = []
            traj_cols = []
            traj_color = np.array([0.0, 0.0, 1.0])
            for i in range(len(camera_positions) - 1):
                start = camera_positions[i]
                end = camera_positions[i + 1]
                seg_points = np.linspace(start, end, num_points_per_segment)
                seg_colors = np.tile(traj_color, (num_points_per_segment, 1))
                traj_pts.append(seg_points)
                traj_cols.append(seg_colors)

            if not traj_pts:
                print("No segments to save for trajectory.")
                return

            traj_pts = np.concatenate(traj_pts, axis=0)
            traj_cols = np.concatenate(traj_cols, axis=0)

            # Build dense points for normals (optional)
            normal_pts = []
            normal_cols = []
            normal_color = np.array([0.0, 1.0, 0.0])
            for i, pos in enumerate(camera_positions):
                nrm = per_frame_normals[i] if i < len(per_frame_normals) else None
                if nrm is None:
                    continue
                end = pos + nrm * normal_length
                seg_points = np.linspace(pos, end, 20)
                seg_colors = np.tile(normal_color, (seg_points.shape[0], 1))
                normal_pts.append(seg_points)
                normal_cols.append(seg_colors)

            # Save PLYs next to the database
            out_dir = os.path.dirname(self.config.database_path) if self.config.database_path else self.config.output_dir
            os.makedirs(out_dir, exist_ok=True)

            # Trajectory
            traj_pcd = o3d.geometry.PointCloud()
            traj_pcd.points = o3d.utility.Vector3dVector(traj_pts)
            traj_pcd.colors = o3d.utility.Vector3dVector(traj_cols)
            traj_path = os.path.join(out_dir, "query_trajectory_only.ply")
            o3d.io.write_point_cloud(traj_path, traj_pcd)
            print(f"Saved trajectory-only PLY to {traj_path}")

            # Normals (if any)
            if normal_pts:
                normals_pcd = o3d.geometry.PointCloud()
                normals_pcd.points = o3d.utility.Vector3dVector(np.concatenate(normal_pts, axis=0))
                normals_pcd.colors = o3d.utility.Vector3dVector(np.concatenate(normal_cols, axis=0))
                normals_path = os.path.join(out_dir, "query_normals.ply")
                o3d.io.write_point_cloud(normals_path, normals_pcd)
                print(f"Saved normals PLY to {normals_path}")
        except Exception as e:
            print(f"Warning: Failed to save trajectory/normals: {e}")
