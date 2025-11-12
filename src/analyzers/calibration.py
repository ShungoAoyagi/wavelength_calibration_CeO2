"""
Module providing calibration functions
Performs camera length and wavelength calibration
"""
import numpy as np
from scipy.optimize import curve_fit


class CalibrationHelper:
    """
    Class responsible for camera length and wavelength calibration
    """
    
    def __init__(self, geometry_calculator):
        """
        Parameters:
        -----------
        geometry_calculator : GeometryCalculator
            Object for performing geometry calculations
        """
        self.geometry_calc = geometry_calculator
    
    def calibrate_camera_length(self, peak_positions_pixel, d_spacings, wavelength):
        """
        Fit camera length from known d-spacings and measured peak positions
        
        Parameters:
        -----------
        peak_positions_pixel : array-like
            Measured peak positions (in pixels)
        d_spacings : array-like
            Corresponding d-spacings (A)
        wavelength : float
            X-ray wavelength (A)
        
        Returns:
        --------
        camera_length : float
            Fitted camera length (m)
        camera_length_err : float
            Camera length error (m)
        """
        def model(r_pixel, L):
            """Model with camera length L as parameter"""
            two_theta_calc = np.array([
                self.geometry_calc.calculate_two_theta(r, L) for r in r_pixel
            ])
            return two_theta_calc
        
        # Calculate theoretical 2θ angles
        two_theta_theory = np.array([
            self.geometry_calc.bragg_law(d, wavelength) for d in d_spacings
        ])
        
        # Initial guess (1m)
        initial_guess = 1.0
        
        # Execute fitting
        popt, pcov = curve_fit(
            model, 
            peak_positions_pixel, 
            two_theta_theory,
            p0=[initial_guess]
        )
        
        camera_length = popt[0]
        camera_length_err = np.sqrt(np.diag(pcov))[0]
        
        return camera_length, camera_length_err
    
    def calibrate_wavelength(self, peak_positions_pixel, d_spacings, camera_length):
        """
        Fit wavelength from known d-spacings and measured peak positions
        
        Parameters:
        -----------
        peak_positions_pixel : array-like
            Measured peak positions (in pixels)
        d_spacings : array-like
            Corresponding d-spacings (A)
        camera_length : float
            Camera length (m)
        
        Returns:
        --------
        wavelength : float
            Fitted wavelength (A)
        wavelength_err : float
            Wavelength error (A)
        """
        # Calculate measured 2θ angles
        two_theta_measured = np.array([
            self.geometry_calc.calculate_two_theta(r, camera_length) 
            for r in peak_positions_pixel
        ])
        
        # Calculate wavelength from Bragg's law
        wavelengths = 2 * np.array(d_spacings) * np.sin(two_theta_measured / 2)
        
        # Calculate mean and standard deviation
        wavelength = np.mean(wavelengths)
        wavelength_err = np.std(wavelengths)
        
        return wavelength, wavelength_err
    
    def generate_calibration_report(self, peak_positions, peak_assignments, 
                                   camera_length, wavelength, correct_tilt=True,
                                   show_comparison=True):
        """
        Generate calibration results report
        
        Parameters:
        -----------
        peak_positions : ndarray
            Detected peak positions (in pixels)
        peak_assignments : dict
            Format: {peak_index: (hkl, d_spacing)}
        camera_length : float
            Camera length (m)
        wavelength : float
            Wavelength (A)
        correct_tilt : bool
            Whether to use detector tilt correction
        show_comparison : bool
            Whether to display comparison with/without tilt correction
        """
        geom = self.geometry_calc
        
        print("=" * 70)
        print("Calibration results")
        print("=" * 70)
        print(f"\nCamera length: {camera_length*1000:.6f} mm")
        print(f"Wavelength: {wavelength:.8f} A")
        print(f"Energy: {12.398/wavelength:.6f} keV")
        print(f"\nDetector parameters:")
        print(f"  Pixel size: {geom.pixel_size_h*1e6:.1f} μm × {geom.pixel_size_v*1e6:.1f} μm")
        print(f"  Beam center: ({geom.beam_center_x:.2f}, {geom.beam_center_y:.2f}) pixels")
        print(f"  Tilt angle: {geom.tilt_angle:.4f}°")
        print(f"  Rotation angle of tilting plane: {geom.tilt_rotation:.4f}°")
        print(f"  Tilt correction: {'Enabled' if correct_tilt else 'Disabled'}")
        
        print("\n" + "-" * 70)
        print("Details of detected peaks:")
        print("-" * 70)
        
        if show_comparison and geom.tilt_angle != 0:
            print(f"{'Peak':<6} {'hkl':<8} {'r(px)':<10} {'r(mm)':<10} {'2θ(corr)':<12} {'2θ(uncorr)':<14} {'Δ2θ':<10} {'d(A)':<12}")
            print("-" * 70)
            
            for peak_idx, (hkl, d_theo) in peak_assignments.items():
                r_px = peak_positions[peak_idx]
                r_mm = geom.pixel_to_distance(r_px) * 1000  # m -> mm
                
                # With tilt correction
                two_theta_corr = geom.calculate_two_theta(r_px, camera_length, correct_tilt=True)
                two_theta_corr_deg = np.degrees(two_theta_corr)
                
                # Without tilt correction
                two_theta_uncorr = geom.calculate_two_theta(r_px, camera_length, correct_tilt=False)
                two_theta_uncorr_deg = np.degrees(two_theta_uncorr)
                
                # Difference
                delta_two_theta = two_theta_corr_deg - two_theta_uncorr_deg
                
                # 2θ to use
                two_theta_used = two_theta_corr if correct_tilt else two_theta_uncorr
                d_measured = wavelength / (2 * np.sin(two_theta_used / 2))
                
                print(f"{peak_idx+1:<6} {hkl:<8} {r_px:<10.3f} {r_mm:<10.4f} "
                      f"{two_theta_corr_deg:<12.6f} {two_theta_uncorr_deg:<14.6f} "
                      f"{delta_two_theta:<10.6f} {d_measured:<12.8f}")
        else:
            print(f"{'Peak':<6} {'hkl':<8} {'r (px)':<10} {'r (mm)':<10} {'2θ (deg)':<12} {'d (A)':<12}")
            print("-" * 70)
            
            for peak_idx, (hkl, d_theo) in peak_assignments.items():
                r_px = peak_positions[peak_idx]
                r_mm = geom.pixel_to_distance(r_px) * 1000  # m -> mm
                two_theta = geom.calculate_two_theta(r_px, camera_length, correct_tilt=correct_tilt)
                two_theta_deg = np.degrees(two_theta)
                d_measured = wavelength / (2 * np.sin(two_theta / 2))
                
                print(f"{peak_idx+1:<6} {hkl:<8} {r_px:<10.3f} {r_mm:<10.4f} {two_theta_deg:<12.6f} {d_measured:<12.8f}")
        
        print("=" * 70)

