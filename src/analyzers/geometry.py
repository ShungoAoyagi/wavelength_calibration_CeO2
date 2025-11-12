"""
Module providing geometry calculation functions
Performs detector tilt correction and 2θ angle calculations
"""
import numpy as np


class GeometryCalculator:
    """
    Class responsible for detector geometry and 2θ angle calculations
    """
    
    def __init__(self, pixel_size_h, pixel_size_v, beam_center_x, beam_center_y,
                 tilt_rotation=0.0, tilt_angle=0.0):
        """
        Parameters:
        -----------
        pixel_size_h : float
            Horizontal pixel size [m]
        pixel_size_v : float
            Vertical pixel size [m]
        beam_center_x : float
            X-coordinate of beam center (column)
        beam_center_y : float
            Y-coordinate of beam center (row)
        tilt_rotation : float
            Rotation angle of tilting plane [degrees]
        tilt_angle : float
            Detector tilt angle [degrees]
        """
        self.pixel_size_h = pixel_size_h
        self.pixel_size_v = pixel_size_v
        self.beam_center_x = beam_center_x
        self.beam_center_y = beam_center_y
        self.tilt_rotation = tilt_rotation
        self.tilt_angle = tilt_angle
    
    def pixel_to_distance(self, pixel_radius):
        """
        Convert pixel radius to real-space distance (m)
        
        Parameters:
        -----------
        pixel_radius : float or ndarray
            Radius in pixels
        
        Returns:
        --------
        distance : float or ndarray
            Distance in real space (m)
        """
        # Simple conversion (ignoring detector tilt)
        return pixel_radius * self.pixel_size_h
    
    def calculate_two_theta(self, pixel_radius, camera_length, correct_tilt=True):
        """
        Calculate 2θ angle from pixel radius (with detector tilt correction)
        
        Parameters:
        -----------
        pixel_radius : float or ndarray
            Radius in pixels
        camera_length : float
            Camera length (detector-sample distance) [m]
        correct_tilt : bool
            Whether to perform detector tilt correction
        
        Returns:
        --------
        two_theta : float or ndarray
            2θ angle (radians)
        """
        if not correct_tilt or (self.tilt_angle == 0 and self.tilt_rotation == 0):
            # Simple calculation without tilt correction
            distance = self.pixel_to_distance(pixel_radius)
            two_theta = np.arctan(distance / camera_length)
            return two_theta
        
        # Calculation with tilt correction
        # Handle both scalar and vector cases for pixel_radius
        is_scalar = np.isscalar(pixel_radius)
        if is_scalar:
            pixel_radius = np.array([pixel_radius])
        
        two_theta_corrected = []
        for r in pixel_radius:
            # Calculate average 2θ in radial direction (average over multiple points on circumference)
            angles = np.linspace(0, 2*np.pi, 360, endpoint=False)
            two_thetas = []
            
            for angle in angles:
                # Convert from polar to pixel coordinates
                x_px = self.beam_center_x + r * np.cos(angle)
                y_px = self.beam_center_y + r * np.sin(angle)
                
                # Calculate 2θ at that pixel position
                two_theta_point = self.calculate_two_theta_for_pixel(
                    x_px, y_px, camera_length
                )
                two_thetas.append(two_theta_point)
            
            # Get average value
            two_theta_corrected.append(np.mean(two_thetas))
        
        result = np.array(two_theta_corrected)
        return result[0] if is_scalar else result
    
    def calculate_two_theta_for_pixel(self, x_pixel, y_pixel, camera_length):
        """
        Calculate 2θ angle at specific pixel position (with detector tilt correction)
        
        Parameters:
        -----------
        x_pixel : float
            X-coordinate of pixel
        y_pixel : float
            Y-coordinate of pixel
        camera_length : float
            Camera length (detector-sample distance) [m]
        
        Returns:
        --------
        two_theta : float
            2θ angle (radians)
        """
        # Relative coordinates from beam center (in pixels)
        dx_px = x_pixel - self.beam_center_x
        dy_px = y_pixel - self.beam_center_y
        
        # Convert pixel coordinates to real-space coordinates (m)
        dx = dx_px * self.pixel_size_h
        dy = dy_px * self.pixel_size_v
        
        # Convert tilt angles from degrees to radians
        tilt_angle_rad = np.radians(self.tilt_angle)
        rotation_angle_rad = np.radians(self.tilt_rotation)
        
        # 3D coordinates in detector coordinate system (before tilt)
        # z direction is camera length
        x0, y0, z0 = dx, dy, camera_length
        
        # Coordinate transformation according to tilt plane rotation angle
        # First rotate around z-axis (rotation_angle)
        x1 = x0 * np.cos(rotation_angle_rad) - y0 * np.sin(rotation_angle_rad)
        y1 = x0 * np.sin(rotation_angle_rad) + y0 * np.cos(rotation_angle_rad)
        z1 = z0
        
        # Then apply tilt angle (rotation around y-axis after rotation)
        x2 = x1 * np.cos(tilt_angle_rad) + z1 * np.sin(tilt_angle_rad)
        y2 = y1
        z2 = -x1 * np.sin(tilt_angle_rad) + z1 * np.cos(tilt_angle_rad)
        
        # Distance from sample to pixel on detector
        distance_3d = np.sqrt(x2**2 + y2**2 + z2**2)
        
        # Calculate 2θ angle
        # z2 is the component in the beam axis direction
        two_theta = np.arccos(z2 / distance_3d)
        
        return two_theta
    
    @staticmethod
    def bragg_law(d_spacing, wavelength):
        """
        Calculate 2θ from Bragg's law
        
        Parameters:
        -----------
        d_spacing : float
            Interplanar spacing (A)
        wavelength : float
            Wavelength (A)
        
        Returns:
        --------
        two_theta : float
            2θ angle (radians)
        """
        return 2 * np.arcsin(wavelength / (2 * d_spacing))

