import cv2
import numpy as np
import torch
from typing import Tuple, Optional, Dict, Any, List, Union
from utils.camera import Camera

# Try to import torchcodec for video processing
try:
    from torchcodec.decoders import VideoDecoder
    TORCHCODEC_AVAILABLE = True
except ImportError:
    TORCHCODEC_AVAILABLE = False
    print("Warning: torchcodec not available. Install with: pip install torchcodec")


class UndistortionMaps:
    """
    A class to handle camera undistortion maps for efficient image undistortion.
    
    This class pre-computes undistortion maps that can be reused for multiple images
    from the same camera, avoiding the need to recalculate the mapping for each image.
    """
    
    def __init__(self, cam_dist: Camera, cam_undist: Camera = None):
        """
        Initialize undistortion maps using distorted and undistorted camera models.
        
        Args:
            cam_dist: Camera object with distortion parameters
            cam_undist: Camera object without distortion parameters. If None, will be created
                       automatically by copying cam_dist and setting distortion to zero.
        """
        self.cam_dist = cam_dist
        self.cam_dist_ = cam_dist.get_camera()
        
        # Create undistorted camera if not provided
        if cam_undist is None:
            self.cam_undist = self._create_undistorted_camera()
        else:
            self.cam_undist = cam_undist
            
        self.cam_undist_ = self.cam_undist.get_camera()
        
        # Initialize maps as None - will be computed when needed
        self.map_x = None
        self.map_y = None
        self.maps_computed = False
        self.current_target_size = None  # Track the target size for which maps were computed
        
    def _create_undistorted_camera(self) -> Camera:
        """
        Create an undistorted camera by copying the distorted camera and setting
        all distortion parameters to zero and aspect ratio to 1.
        
        Returns:
            Camera object with zero distortion parameters and unit aspect ratio
        """
        # Create a new camera object
        cam_undist = Camera()
        
        # Copy the camera calibration data
        cam_undist.cam_intr_json = self.cam_dist.cam_intr_json.copy()
        
        # Set all distortion parameters to zero based on the camera model type
        if self.cam_dist.cam_intr_json["intrinsic_type"] == "DIVISION_UNDISTORTION":
            cam_undist.cam_intr_json["intrinsics"]["div_undist_distortion"] = 0.0
        elif self.cam_dist.cam_intr_json["intrinsic_type"] == "FISHEYE":
            cam_undist.cam_intr_json["intrinsics"]["radial_distortion_1"] = 0.0
            cam_undist.cam_intr_json["intrinsics"]["radial_distortion_2"] = 0.0
            cam_undist.cam_intr_json["intrinsics"]["radial_distortion_3"] = 0.0
            cam_undist.cam_intr_json["intrinsics"]["radial_distortion_4"] = 0.0
        elif self.cam_dist.cam_intr_json["intrinsic_type"] == "PINHOLE":
            cam_undist.cam_intr_json["intrinsics"]["radial_distortion_1"] = 0.0
            cam_undist.cam_intr_json["intrinsics"]["radial_distortion_2"] = 0.0
        elif self.cam_dist.cam_intr_json["intrinsic_type"] == "PINHOLE_RADIAL_TANGENTIAL":
            cam_undist.cam_intr_json["intrinsics"]["radial_distortion_1"] = 0.0
            cam_undist.cam_intr_json["intrinsics"]["radial_distortion_2"] = 0.0
            cam_undist.cam_intr_json["intrinsics"]["radial_distortion_3"] = 0.0
            cam_undist.cam_intr_json["intrinsics"]["tangential_distortion_1"] = 0.0
            cam_undist.cam_intr_json["intrinsics"]["tangential_distortion_2"] = 0.0

        # set principal point to center of image
        cam_undist.cam_intr_json["intrinsics"]["principal_pt_x"] = self.cam_dist.cam_intr_json["image_width"] / 2
        cam_undist.cam_intr_json["intrinsics"]["principal_pt_y"] = self.cam_dist.cam_intr_json["image_height"] / 2
        
        # Set aspect ratio to 1 for square pixels
        cam_undist.cam_intr_json["intrinsics"]["aspect_ratio"] = 1.0
        
        # Load the undistorted camera calibration
        cam_undist._load_camera_calibration()
        
        return cam_undist
        
    def compute_maps(self, target_size: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute undistortion maps for the given target size.
        
        Args:
            target_size: Target size (height, width) for the undistorted image.
                        If None, uses the undistorted camera's image dimensions.
        
        Returns:
            Tuple of (map_x, map_y) arrays for cv2.remap
        """
        if target_size is None:
            target_height = self.cam_undist_.ImageHeight()
            target_width = self.cam_undist_.ImageWidth()
        else:
            target_height, target_width = target_size
            
        # Initialize maps
        self.map_x = np.zeros((target_height, target_width), dtype=np.float32)
        self.map_y = np.zeros((target_height, target_width), dtype=np.float32)
        
        print(f"Computing undistortion maps for size ({target_width}, {target_height})...")
        
        # Compute mapping for each pixel
        for c in range(target_width):
            for r in range(target_height):
                # Point in undistorted image coordinates
                image_pt_undist = np.array([c, r])
                
                # Convert to camera coordinates in undistorted camera
                pt_in_undist_camera = self.cam_dist_.CameraIntrinsics().ImageToCameraCoordinates(image_pt_undist)
                
                # Project to distorted camera coordinates
                distorted_pt = self.cam_undist_.CameraIntrinsics().CameraToImageCoordinates(pt_in_undist_camera)
                
                # Store the mapping
                self.map_x[r, c] = distorted_pt[0]
                self.map_y[r, c] = distorted_pt[1]
        
        self.maps_computed = True
        self.current_target_size = target_size
        print("✓ Undistortion maps computed successfully")
        
        return self.map_x, self.map_y
    
    def get_maps(self, target_size: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the undistortion maps, computing them if necessary.
        
        Args:
            target_size: Target size (height, width) for the undistorted image.
                        If None, uses the undistorted camera's image dimensions.
        
        Returns:
            Tuple of (map_x, map_y) arrays for cv2.remap
        """
        # Only recompute if maps haven't been computed yet, or if target_size has changed
        if not self.maps_computed or target_size != self.current_target_size:
            return self.compute_maps(target_size)
        return self.map_x, self.map_y
    
    def undistort_image(self, img: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Undistort an image using the pre-computed maps.
        
        Args:
            img: Input image (numpy array)
            target_size: Target size (height, width) for the undistorted image.
                        If None, uses the undistorted camera's image dimensions.
        
        Returns:
            Undistorted image
        """
        map_x, map_y = self.get_maps(target_size)
        
        # Resize input image to match the distorted camera dimensions if needed
        if img.shape[:2] != (self.cam_dist_.ImageHeight(), self.cam_dist_.ImageWidth()):
            img = cv2.resize(img, (self.cam_dist_.ImageWidth(), self.cam_dist_.ImageHeight()))
        
        # Apply undistortion
        undistorted_img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        
        return undistorted_img
    
    def undistort_tensor(self, img_tensor: torch.Tensor, target_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Undistort a PyTorch tensor image using the pre-computed maps.
        
        Args:
            img_tensor: Input image tensor (C, H, W) or (H, W, C)
            target_size: Target size (height, width) for the undistorted image.
                        If None, uses the undistorted camera's image dimensions.
        
        Returns:
            Undistorted image tensor
        """
        # Convert tensor to numpy
        if img_tensor.dim() == 3:
            if img_tensor.shape[0] == 3:  # CHW format
                img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
            else:  # HWC format
                img_np = img_tensor.cpu().numpy()
        else:
            raise ValueError(f"Expected 3D tensor, got shape {img_tensor.shape}")
        
        # Convert from [0, 1] to [0, 255] if needed
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        
        # Undistort
        undistorted_np = self.undistort_image(img_np, target_size)
        
        # Convert back to tensor
        if img_tensor.shape[0] == 3:  # Original was CHW
            undistorted_tensor = torch.from_numpy(undistorted_np).permute(2, 0, 1).float() / 255.0
        else:  # Original was HWC
            undistorted_tensor = torch.from_numpy(undistorted_np).float() / 255.0
        
        return undistorted_tensor


def create_undistortion_maps_from_file(
    cam_dist_path: str, 
    scale: float = 1.0
) -> UndistortionMaps:
    """
    Create undistortion maps from a single camera calibration file.
    The undistorted camera will be created automatically by setting distortion to zero.
    
    Args:
        cam_dist_path: Path to distorted camera calibration JSON file
        scale: Scaling factor for camera parameters
    
    Returns:
        UndistortionMaps object
    """
    # Create camera object
    cam_dist = Camera()
    cam_dist.load_camera_calibration_file(cam_dist_path, scale)
    
    return UndistortionMaps(cam_dist)


def create_undistortion_maps_from_files(
    cam_dist_path: str, 
    cam_undist_path: str, 
    scale: float = 1.0
) -> UndistortionMaps:
    """
    Create undistortion maps from camera calibration files.
    
    Args:
        cam_dist_path: Path to distorted camera calibration JSON file
        cam_undist_path: Path to undistorted camera calibration JSON file
        scale: Scaling factor for camera parameters
    
    Returns:
        UndistortionMaps object
    """
    # Create camera objects
    cam_dist = Camera()
    cam_dist.load_camera_calibration_file(cam_dist_path, scale)
    
    cam_undist = Camera()
    cam_undist.load_camera_calibration_file(cam_undist_path, scale)
    
    return UndistortionMaps(cam_dist, cam_undist)


def create_undistortion_maps_from_json(
    cam_dist_json: Dict[str, Any], 
    cam_undist_json: Dict[str, Any] = None, 
    scale: float = 1.0
) -> UndistortionMaps:
    """
    Create undistortion maps from camera calibration JSON objects.
    
    Args:
        cam_dist_json: Distorted camera calibration JSON object
        cam_undist_json: Undistorted camera calibration JSON object. If None, will be created
                        automatically by copying cam_dist_json and setting distortion to zero.
        scale: Scaling factor for camera parameters
    
    Returns:
        UndistortionMaps object
    """
    # Create camera objects
    cam_dist = Camera()
    cam_dist.load_camera_calibration_json(cam_dist_json, scale)
    
    if cam_undist_json is not None:
        cam_undist = Camera()
        cam_undist.load_camera_calibration_json(cam_undist_json, scale)
        return UndistortionMaps(cam_dist, cam_undist)
    else:
        return UndistortionMaps(cam_dist)


class UndistortionImageLoader:
    """
    A wrapper class that adds undistortion functionality to existing image loading.
    """
    
    def __init__(self, undistortion_maps: UndistortionMaps):
        """
        Initialize the undistortion image loader.
        
        Args:
            undistortion_maps: Pre-computed undistortion maps
        """
        self.undistortion_maps = undistortion_maps
    
    def load_and_undistort_image(self, image_path: str, target_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Load an image and apply undistortion.
        
        Args:
            image_path: Path to the image file
            target_size: Target size (height, width) for the undistorted image
        
        Returns:
            Undistorted image tensor (C, H, W) in [0, 1] range
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply undistortion
        undistorted_img = self.undistortion_maps.undistort_image(img_rgb, target_size)
        
        # Convert to tensor
        img_tensor = torch.from_numpy(undistorted_img).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1)  # HWC to CHW
        
        return img_tensor
    
    def load_and_undistort_video_frame(self, video_path: str, frame_idx: int, target_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Load a video frame and apply undistortion using torchcodec if available.
        
        Args:
            video_path: Path to the video file
            frame_idx: Frame index to load
            target_size: Target size (height, width) for the undistorted image
        
        Returns:
            Undistorted image tensor (C, H, W) in [0, 1] range
        """
        # Try to use torchcodec first
        if TORCHCODEC_AVAILABLE:
            try:
                decoder = VideoDecoder(video_path, device="cpu")
                frame_tensor = decoder[frame_idx]  # Shape: [C, H, W], uint8
                del decoder
                
                # Convert to float [0, 1] range
                frame_tensor = frame_tensor.float() / 255.0
                
                # Apply undistortion
                undistorted_tensor = self.undistortion_maps.undistort_tensor(frame_tensor, target_size)
                
                return undistorted_tensor
            except Exception as e:
                print(f"Warning: torchcodec failed, falling back to OpenCV: {e}")
        
        # Fall back to OpenCV
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Could not read frame {frame_idx} from {video_path}")
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Apply undistortion
        undistorted_frame = self.undistortion_maps.undistort_image(frame_rgb, target_size)
        
        # Convert to tensor
        frame_tensor = torch.from_numpy(undistorted_frame).float() / 255.0
        frame_tensor = frame_tensor.permute(2, 0, 1)  # HWC to CHW
        
        return frame_tensor


class VideoUndistortionLoader:
    """
    A class for efficient video loading and undistortion using torchcodec.
    """
    
    def __init__(self, undistortion_maps: UndistortionMaps, device: str = "cpu"):
        """
        Initialize the video undistortion loader.
        
        Args:
            undistortion_maps: Pre-computed undistortion maps
            device: Device to load videos on ("cpu" or "cuda")
        """
        if not TORCHCODEC_AVAILABLE:
            raise ImportError("torchcodec not available. Install with: pip install torchcodec")
        
        self.undistortion_maps = undistortion_maps
        self.device = device
        self.decoders = {}  # Cache for video decoders
    
    def load_video_decoder(self, video_path: str) -> VideoDecoder:
        """
        Load or get cached video decoder.
        
        Args:
            video_path: Path to the video file
        
        Returns:
            VideoDecoder object
        """
        if video_path not in self.decoders:
            self.decoders[video_path] = VideoDecoder(video_path, device=self.device)
        return self.decoders[video_path]
    
    def load_and_undistort_frame(self, video_path: str, frame_idx: int, target_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Load a single video frame and apply undistortion using torchcodec.
        
        Args:
            video_path: Path to the video file
            frame_idx: Frame index to load
            target_size: Target size (height, width) for the undistorted image
        
        Returns:
            Undistorted image tensor (C, H, W) in [0, 1] range
        """
        decoder = self.load_video_decoder(video_path)
        
        # Load frame using torchcodec
        frame_tensor = decoder[frame_idx]  # Shape: [C, H, W], uint8
        
        # Convert to float [0, 1] range
        frame_tensor = frame_tensor.float() / 255.0
        
        # Apply undistortion
        undistorted_tensor = self.undistortion_maps.undistort_tensor(frame_tensor, target_size)
        
        return undistorted_tensor.to(self.device)
    
    def load_and_undistort_frames(self, video_path: str, frame_indices: List[int], target_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Load multiple video frames and apply undistortion using torchcodec.
        
        Args:
            video_path: Path to the video file
            frame_indices: List of frame indices to load
            target_size: Target size (height, width) for the undistorted images
        
        Returns:
            Undistorted image tensor (N, C, H, W) in [0, 1] range
        """
        decoder = self.load_video_decoder(video_path)
        
        # Load frames using torchcodec
        frames_tensor = decoder.get_frames_at(indices=frame_indices)  # Shape: [N, C, H, W], uint8
        
        # Convert to float [0, 1] range
        frames_tensor = frames_tensor.float() / 255.0
        
        # Apply undistortion to each frame
        undistorted_frames = []
        for i in range(frames_tensor.shape[0]):
            undistorted_frame = self.undistortion_maps.undistort_tensor(frames_tensor[i], target_size)
            undistorted_frames.append(undistorted_frame)
        
        # Stack frames
        result = torch.stack(undistorted_frames, dim=0)
        
        return result.to(self.device)
    
    def load_and_undistort_video_slice(self, video_path: str, start_idx: int, end_idx: int, step: int = 1, target_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Load a slice of video frames and apply undistortion using torchcodec.
        
        Args:
            video_path: Path to the video file
            start_idx: Start frame index
            end_idx: End frame index (exclusive)
            step: Step size between frames
            target_size: Target size (height, width) for the undistorted images
        
        Returns:
            Undistorted image tensor (N, C, H, W) in [0, 1] range
        """
        decoder = self.load_video_decoder(video_path)
        
        # Load frames using torchcodec slice indexing
        frames_tensor = decoder[start_idx:end_idx:step]  # Shape: [N, C, H, W], uint8
        
        # Convert to float [0, 1] range
        frames_tensor = frames_tensor.float() / 255.0
        
        # Apply undistortion to each frame
        undistorted_frames = []
        for i in range(frames_tensor.shape[0]):
            undistorted_frame = self.undistortion_maps.undistort_tensor(frames_tensor[i], target_size)
            undistorted_frames.append(undistorted_frame)
        
        # Stack frames
        result = torch.stack(undistorted_frames, dim=0)
        
        return result.to(self.device)
    
    def get_video_metadata(self, video_path: str) -> Any:
        """
        Get video metadata using torchcodec.
        
        Args:
            video_path: Path to the video file
        
        Returns:
            Video metadata
        """
        decoder = self.load_video_decoder(video_path)
        return decoder.metadata
    
    def close_decoder(self, video_path: str):
        """
        Close a specific video decoder to free memory.
        
        Args:
            video_path: Path to the video file
        """
        if video_path in self.decoders:
            del self.decoders[video_path]
    
    def close_all_decoders(self):
        """Close all video decoders to free memory."""
        self.decoders.clear()
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close_all_decoders() 