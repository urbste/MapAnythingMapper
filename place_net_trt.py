# Try to import torch_tensorrt
import torch
import os
import json
import torchvision.transforms as tvf

try:
    import torch_tensorrt
    HAS_TENSORRT = True
except ImportError:
    HAS_TENSORRT = False
    print("Warning: torch_tensorrt not found. TensorRT will not be available.")
    print("To install, use: pip install torch-tensorrt --extra-index-url https://pypi.nvidia.com")

COSPLACE_DIM = 128
COSPLACE_NET = 'ResNet18'
EIGENPLACES_DIM = 128
EIGENPLACES_NET = 'ResNet50'

class PlaceNetTRT(torch.nn.Module):
    """
    TensorRT optimized version of PlaceNet for CosPlace inference with FP16 precision
    and batch size 1, providing global descriptors for image-based localization.
    """
    
    def __init__(self, input_size=(224, 224), device=None):
        """
        Initialize the TensorRT PlaceNet inference model
        
        Args:
            input_size: Tuple (height, width) for the model input size (default: 224x224)
            device: Optional torch device. If None, will use CUDA if available
        """
        super(PlaceNetTRT, self).__init__()
        
        # Check for TensorRT support
        if not HAS_TENSORRT:
            raise ImportError("torch_tensorrt is required but not installed")
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        # CUDA is required for TensorRT
        if self.device.type != 'cuda':
            raise RuntimeError("CUDA is required for TensorRT inference")
            
        # Find available model (try both .ep and .ts formats)
        cosplace_dir = os.path.join(os.getenv('MODEL_WEIGHTS'), "cosplace")
        ep_path = os.path.join(cosplace_dir, f"{COSPLACE_NET.lower()}_{COSPLACE_DIM}_fp16.ep")
        ts_path = os.path.join(cosplace_dir, f"{COSPLACE_NET.lower()}_{COSPLACE_DIM}_fp16.ts")

        # Determine which model file to use
        if os.path.exists(ep_path):
            self.model_path = ep_path
            print(f"Using ExportedProgram format model: {self.model_path}")
        elif os.path.exists(ts_path):
            self.model_path = ts_path
            print(f"Using TorchScript format model: {self.model_path}")
        else:
            raise FileNotFoundError(f"No TensorRT model found in {cosplace_dir}. "
                                   f"Please run tests/compile_models_trt.py to generate the model.")
            
        self.input_size = input_size
        
        # Load model
        self.model, self.precision = self._load_model()
        
        # Create transforms
        self.transform = self._create_transform()
        
        print(f"PlaceNetTRT initialized: input_size={input_size}, precision={self.precision}")
            
    def _load_model(self):
        """Load the TensorRT model based on file extension"""
        print(f"Loading TensorRT model from {self.model_path}...")
        
        # Detect precision from model name or metadata
        precision = self._detect_precision()
        print(f"Using {precision.upper()} precision")
        
        try:
            # Detect format based on file extension
            if self.model_path.endswith('.ts'):
                model = torch.jit.load(self.model_path).to(self.device)
                model_format = "TorchScript"
            else:  # .ep format
                # Don't call .module() directly, use the loaded model
                model = torch.export.load(self.model_path).module()
                # For safety, explicitly move to the device if supported
                if hasattr(model, 'to'):
                    model = model.to(self.device)
                model_format = "Exported Program"
            
            print(f"TensorRT model loaded successfully (format: {model_format})")
            return model, precision
            
        except Exception as e:
            raise RuntimeError(f"Failed to load TensorRT model: {e}")
    
    def _detect_precision(self):
        """Detect precision from model filename or metadata"""
        # First, try to load metadata file
        meta_path = self.model_path + ".meta"
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
                if "precision" in metadata:
                    return metadata["precision"]
            except Exception as e:
                print(f"Warning: Failed to load metadata file: {e}")
        
        # Second, try to detect from filename
        if "_fp16" in self.model_path.lower():
            return "fp16"
        elif "_fp32" in self.model_path.lower():
            return "fp32"
        
        # Default to FP16
        return "fp16"
    
    def _create_transform(self):
        """Create preprocessing transform for the input images"""
        return tvf.Compose([
            tvf.ToPILImage(),
            tvf.Resize(self.input_size),
            tvf.ToTensor(),
            tvf.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def preprocess(self, image):
        """
        Preprocess an image for inference
        
        Args:
            image: RGB numpy array of shape (H, W, 3)
            
        Returns:
            tensor: Preprocessed tensor ready for the model
        """
        # Apply transforms
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Convert to half precision if using FP16
        if self.precision == "fp16":
            tensor = tensor.half()
            
        return tensor
    
    def forward(self, image):
        """
        Run inference on a single image
        
        Args:
            image: RGB numpy array of shape (H, W, 3)
            
        Returns:
            descriptor: Global descriptor as numpy array
        """
        with torch.no_grad():
            # Preprocess image
            tensor = self.preprocess(image)
            
            # Run inference
            descriptor = self.model(tensor)
                
            return descriptor.cpu().float().numpy()
    
    def __call__(self, image):
        """
        Allow the model to be called directly on an image
        
        Args:
            image: RGB numpy array of shape (H, W, 3)
            
        Returns:
            descriptor: Global descriptor as numpy array
        """
        return self.forward(image)
    
    def to_device(self, device):
        """
        Move the model to specified device (for API compatibility with PlaceNet)
        
        Args:
            device: Target device
        """
        if device.type != "cuda":
            raise RuntimeError("CUDA is required for TensorRT inference")
            
        self.device = device
        if hasattr(self.model, 'module'):
            self.model = self.model.module().to(device)
        else:
            self.model = self.model.to(device)