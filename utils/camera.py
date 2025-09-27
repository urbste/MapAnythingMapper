import pytheia as pt
import json


class Camera:
    ''' Camera

    A class that represents a camera model and its intrinsic parameters. 
    It provides methods to load camera calibration data from JSON files, 
    manage camera intrinsics, and handle distortion models. 
    The class utilizes the pytheia library for camera-related operations.

    Attributes:
    intr (pytheia.sfm.Camera): The camera object representing the intrinsic parameters.
    cam_intr_json (dict): The camera intrinsic parameters loaded from a JSON file.
    prior (pytheia.sfm.CameraIntrinsicsPrior): The prior information for camera intrinsics.
    '''

    def __init__(self) -> None:
        pass
    
    def get_camera(self):
        ''' Retrieve the camera object.

        Returns:
        pytheia.sfm.Camera: The camera object with current intrinsic parameters.
        '''
        return self.intr

    def load_camera_calibration_json(self, json, scale):
        ''' Load camera calibration data from a JSON object.

        Parameters:
        json (dict): The JSON object containing camera intrinsic parameters.
        scale (float): A scaling factor to apply to the intrinsic parameters.
        '''
        self.cam_intr_json = json
        self._load_camera_calibration(scale)

    def load_camera_calibration_file(self, path_to_json, scale):
        ''' Load camera calibration data from a JSON file.

        Parameters:
        path_to_json (str): The path to the JSON file containing camera intrinsic parameters.
        scale (float): A scaling factor to apply to the intrinsic parameters.
        '''
        with open(path_to_json, "r") as f:
            self.cam_intr_json = json.load(f)
        self._load_camera_calibration(scale)

    def _load_camera_calibration(self, scale=1.0):
        ''' Internal method to load camera calibration data and set intrinsic parameters.

        Parameters:
        scale (float): A scaling factor to apply to the intrinsic parameters.
        '''
        
        self.intr = pt.sfm.Camera()

        self.prior = pt.sfm.CameraIntrinsicsPrior()
        self.prior.aspect_ratio.value = [self.cam_intr_json["intrinsics"]["aspect_ratio"]]
        self.prior.image_width = int(self.cam_intr_json["image_width"] * scale)
        self.prior.image_height = int(self.cam_intr_json["image_height"] * scale)
        self.prior.principal_point.value = [self.cam_intr_json["intrinsics"]["principal_pt_x"] * scale, 
                                    self.cam_intr_json["intrinsics"]["principal_pt_y"] * scale]
        self.prior.focal_length.value = [self.cam_intr_json["intrinsics"]["focal_length"] * scale]
        self.prior.skew.value = [self.cam_intr_json["intrinsics"]["skew"]]
        self.prior.camera_intrinsics_model_type = self.cam_intr_json["intrinsic_type"] 
        self._set_distortion(scale)
       
        self.intr.SetFromCameraIntrinsicsPriors(self.prior)

    def _set_distortion(self, scale):
        ''' Set the distortion parameters based on the camera model type.

        Parameters:
        scale (float): A scaling factor to apply to the distortion parameters.
        '''
        if self.prior.camera_intrinsics_model_type == "DIVISION_UNDISTORTION":
            self.prior.radial_distortion.value = [self.cam_intr_json["intrinsics"]["div_undist_distortion"], 0, 0, 0]
        elif self.prior.camera_intrinsics_model_type == "FISHEYE":
            self.prior.radial_distortion.value = [
                self.cam_intr_json["intrinsics"]["radial_distortion_1"],
                self.cam_intr_json["intrinsics"]["radial_distortion_2"],
                self.cam_intr_json["intrinsics"]["radial_distortion_3"],
                self.cam_intr_json["intrinsics"]["radial_distortion_4"]
            ]
        elif self.prior.camera_intrinsics_model_type == "PINHOLE":
            self.prior.radial_distortion.value = [
                self.cam_intr_json["intrinsics"]["radial_distortion_1"],
                self.cam_intr_json["intrinsics"]["radial_distortion_2"],
                0.0, 0.0
            ]

    def flip_intrinsics(self):
        ''' Flip the camera intrinsics, adjusting the principal point and distortion coefficients. '''
        # Flip the y-coordinate of the principal point
        self.prior.principal_point.value[0] = self.prior.image_width - self.prior.principal_point.value[0]
        self.prior.principal_point.value[1] = self.prior.image_height - self.prior.principal_point.value[1]
        
        # Depending on the distortion model, you might need to adjust distortion coefficients
        if self.prior.camera_intrinsics_model_type in ["FISHEYE", "DIVISION_UNDISTORTION"]:
            # This example assumes no changes, but adjust if your model specifications require
            pass
        
        # Re-set the camera intrinsics
        self.intr.SetFromCameraIntrinsicsPriors(self.prior)
