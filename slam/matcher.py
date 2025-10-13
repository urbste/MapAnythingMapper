"""Keypoint matching and visualization utilities using EfficientLoFTR."""

import os
import cv2
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from typing import Optional, Tuple
from transformers import AutoImageProcessor, AutoModelForKeypointMatching
from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT


class Matcher:
    """Handles keypoint matching and visualization using EfficientLoFTR."""
    
    def __init__(self, device: str = "cuda", keypoint_matching_threshold: float = 0.2, max_resolution: int = 640):
        """
        Initialize the Matcher with EfficientLoFTR model.
        
        Args:
            device: Device to run the model on ('cuda' or 'cpu')
            keypoint_matching_threshold: Threshold for keypoint matching confidence
            max_resolution: Maximum resolution (longest edge) for matching. Default 640px.
                           Smaller values = faster matching but potentially fewer matches.
                           EfficientLoFTR default is 640x480.
        """
        self.device = device
        self.keypoint_matching_threshold = keypoint_matching_threshold
        self.max_resolution = max_resolution
        
        print(f"Initializing EfficientLoFTR model for Matcher (max_resolution={max_resolution})...")
        self.loftr_processor = AutoImageProcessor.from_pretrained("zju-community/efficientloftr")
        self.loftr_model = AutoModelForKeypointMatching.from_pretrained("zju-community/efficientloftr")
        self.loftr_model = self.loftr_model.to(self.device).eval()
        print("EfficientLoFTR model initialized in Matcher.")
    
    def _resize_image_pil(self, pil_image: Image.Image) -> Image.Image:
        """
        Resize PIL image to max_resolution while maintaining aspect ratio.
        
        Args:
            pil_image: PIL Image to resize
            
        Returns:
            Resized PIL Image
        """
        width, height = pil_image.size
        max_dim = max(width, height)
        
        if max_dim > self.max_resolution:
            scale = self.max_resolution / max_dim
            new_width = int(width * scale)
            new_height = int(height * scale)
            return pil_image.resize((new_width, new_height), Image.Resampling.BILINEAR)
        
        return pil_image
    
    def count_keypoint_matches_batch(
        self,
        query_img_tensor: torch.Tensor,
        ref_img_paths: list,
    ) -> list:
        """
        Count keypoint matches between one query image and multiple reference images in batch.
        This is much faster than calling count_keypoint_matches multiple times.
        
        Args:
            query_img_tensor: Query image tensor (normalized, C x H x W)
            ref_img_paths: List of paths to reference images
            
        Returns:
            List of match counts for each reference image
        """
        try:
            # Convert query tensor back to PIL Image
            img_norm = IMAGE_NORMALIZATION_DICT["dinov2"]
            mean = torch.as_tensor(img_norm.mean, dtype=torch.float32).view(3, 1, 1).to(query_img_tensor.device)
            std = torch.as_tensor(img_norm.std, dtype=torch.float32).view(3, 1, 1).to(query_img_tensor.device)
            
            query_denorm = query_img_tensor * std + mean
            query_denorm = torch.clamp(query_denorm * 255, 0, 255).byte()
            query_img_np = query_denorm.permute(1, 2, 0).cpu().numpy()
            query_pil = Image.fromarray(query_img_np)
            query_pil = self._resize_image_pil(query_pil)
            
            match_counts = []
            
            # Process each reference image
            for ref_img_path in ref_img_paths:
                ref_img_bgr = cv2.imread(ref_img_path)
                if ref_img_bgr is None:
                    print(f"Warning: Could not load reference image {ref_img_path}")
                    match_counts.append(0)
                    continue
                
                ref_img_rgb = cv2.cvtColor(ref_img_bgr, cv2.COLOR_BGR2RGB)
                ref_pil = Image.fromarray(ref_img_rgb)
                ref_pil = self._resize_image_pil(ref_pil)
                
                # Prepare images for EfficientLoFTR
                images = [query_pil, ref_pil]
                inputs = self.loftr_processor(images, return_tensors="pt")
                
                # Move inputs to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Run matching
                with torch.inference_mode():
                    outputs = self.loftr_model(**inputs)
                
                # Process outputs with matching threshold
                image_sizes = [[(image.height, image.width) for image in images]]
                threshold = self.keypoint_matching_threshold
                processed_outputs = self.loftr_processor.post_process_keypoint_matching(
                    outputs, image_sizes, threshold=threshold
                )
                
                # Extract match count
                if len(processed_outputs) > 0:
                    output = processed_outputs[0]
                    num_matches = len(output["keypoints0"])
                else:
                    num_matches = 0
                
                match_counts.append(num_matches)
            
            return match_counts
            
        except Exception as e:
            print(f"Warning: Batch keypoint matching failed: {e}")
            return [0] * len(ref_img_paths)
    
    def count_keypoint_matches(
        self, 
        query_img_tensor: torch.Tensor, 
        ref_img_path: str, 
        return_details: bool = False
    ) -> Tuple:
        """
        Count the number of keypoint matches between a query image and a reference image using EfficientLoFTR.
        
        Args:
            query_img_tensor: Query image tensor (normalized, C x H x W)
            ref_img_path: Path to the reference image
            return_details: If True, return (num_matches, query_img_np, ref_img_rgb, keypoints0, keypoints1, matching_scores)
            
        Returns:
            Number of valid keypoint matches, or tuple with details if return_details=True
        """
        try:
            # Convert query tensor back to PIL Image
            # Denormalize using dinov2 normalization
            img_norm = IMAGE_NORMALIZATION_DICT["dinov2"]
            mean = torch.as_tensor(img_norm.mean, dtype=torch.float32).view(3, 1, 1).to(query_img_tensor.device)
            std = torch.as_tensor(img_norm.std, dtype=torch.float32).view(3, 1, 1).to(query_img_tensor.device)
            
            query_denorm = query_img_tensor * std + mean
            query_denorm = torch.clamp(query_denorm * 255, 0, 255).byte()
            query_img_np = query_denorm.permute(1, 2, 0).cpu().numpy()
            query_pil = Image.fromarray(query_img_np)
            query_pil = self._resize_image_pil(query_pil)
            
            # Load reference image
            ref_img_bgr = cv2.imread(ref_img_path)
            if ref_img_bgr is None:
                print(f"Warning: Could not load reference image {ref_img_path}")
                if return_details:
                    return 0, None, None, None, None, None
                return 0
            ref_img_rgb = cv2.cvtColor(ref_img_bgr, cv2.COLOR_BGR2RGB)
            ref_pil = Image.fromarray(ref_img_rgb)
            ref_pil = self._resize_image_pil(ref_pil)
            
            # Prepare images for EfficientLoFTR
            images = [query_pil, ref_pil]
            inputs = self.loftr_processor(images, return_tensors="pt")
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run matching
            with torch.inference_mode():
                outputs = self.loftr_model(**inputs)
            
            # Process outputs with matching threshold
            image_sizes = [[(image.height, image.width) for image in images]]
            threshold = self.keypoint_matching_threshold
            processed_outputs = self.loftr_processor.post_process_keypoint_matching(
                outputs, image_sizes, threshold=threshold
            )
            
            # Extract matches from processed outputs
            if len(processed_outputs) > 0:
                output = processed_outputs[0]
                keypoints0 = output["keypoints0"]
                keypoints1 = output["keypoints1"]
                matching_scores = output["matching_scores"]
                num_matches = len(keypoints0)
            else:
                keypoints0 = None
                keypoints1 = None
                matching_scores = None
                num_matches = 0
            
            if return_details:
                return num_matches, query_img_np, ref_img_rgb, keypoints0, keypoints1, matching_scores
            return num_matches
            
        except Exception as e:
            print(f"Warning: Keypoint matching failed: {e}")
            if return_details:
                return 0, None, None, None, None, None
            return 0
    
    def visualize_keypoint_matches(
        self, 
        query_img: np.ndarray, 
        ref_img: np.ndarray, 
        keypoints0: Optional[torch.Tensor], 
        keypoints1: Optional[torch.Tensor], 
        matching_scores: Optional[torch.Tensor], 
        num_matches: int, 
        save_path: str, 
        title: str = "Keypoint Matches", 
        print_samples: bool = False
    ):
        """
        Visualize and save keypoint matches between two images.
        
        Args:
            query_img: Query image as numpy array (H, W, 3)
            ref_img: Reference image as numpy array (H, W, 3)
            keypoints0: Keypoints in query image (tensor or None)
            keypoints1: Keypoints in reference image (tensor or None)
            matching_scores: Confidence scores for matches (tensor or None)
            num_matches: Number of matches
            save_path: Path to save the visualization
            title: Title for the plot
            print_samples: If True, print sample matches to console
        """
        try:
            matplotlib.use('Agg')  # Use non-interactive backend
            
            # Create figure with two subplots side by side
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Display images
            ax1.imshow(query_img)
            ax1.set_title(f'Query Image')
            ax1.axis('off')
            
            ax2.imshow(ref_img)
            ax2.set_title(f'Reference Image')
            ax2.axis('off')
            
            # Draw matches if available
            if keypoints0 is not None and keypoints1 is not None and num_matches > 0:
                # Convert to numpy if needed
                kp0_np = keypoints0.cpu().numpy() if torch.is_tensor(keypoints0) else keypoints0
                kp1_np = keypoints1.cpu().numpy() if torch.is_tensor(keypoints1) else keypoints1
                scores_np = matching_scores.cpu().numpy() if torch.is_tensor(matching_scores) else matching_scores
                
                # Limit to 200 matches for clarity
                n_draw = min(num_matches, 200)
                
                # Generate random colors for each match (for better visualization)
                np.random.seed(42)
                colors = np.random.rand(n_draw, 3)
                
                # Draw matches
                for i in range(n_draw):
                    kp0 = kp0_np[i]
                    kp1 = kp1_np[i]
                    score = scores_np[i] if matching_scores is not None else 1.0
                    color = colors[i]
                    
                    # Draw keypoints with color based on confidence
                    ax1.plot(kp0[0], kp0[1], 'o', color=color, markersize=4, alpha=0.8)
                    ax2.plot(kp1[0], kp1[1], 'o', color=color, markersize=4, alpha=0.8)
                
                # Print some sample matches for debugging if requested
                if print_samples:
                    print(f"  Sample matches (first 5):")
                    for i in range(min(5, num_matches)):
                        kp0 = kp0_np[i]
                        kp1 = kp1_np[i]
                        score = scores_np[i] if matching_scores is not None else 1.0
                        print(f"    Match {i}: query {kp0} -> ref {kp1}, score={score:.3f}")
            
            fig.suptitle(f'{title}: {num_matches} matches (threshold={self.keypoint_matching_threshold})', fontsize=16)
            plt.tight_layout()
            
            # Save figure
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            print(f"Warning: Failed to visualize keypoint matches: {e}")

