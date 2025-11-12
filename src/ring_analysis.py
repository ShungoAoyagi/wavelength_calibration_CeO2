import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import tifffile as tif
import yaml
from src.CeO2_d_spacings import CeO2_d_spacings
from src.analyzers import GeometryCalculator, CalibrationHelper, VisualizationHelper, BeamCenterFinder

def load_tiff(file_path):
    return tif.imread(file_path)

def lorentzian(x, amplitude, center, gamma, background):
    """
    Lorentzian function (Cauchy distribution)
    
    Parameters:
    -----------
    x : array-like
        Independent variable
    amplitude : float
        Peak amplitude
    center : float
        Peak center position
    gamma : float
        Half of the full width at half maximum (FWHM)
    background : float
        Background level
    
    Returns:
    --------
    y : array-like
        Lorentzian function values
    """
    return amplitude / (1 + ((x - center) / gamma)**2) + background

class PowderRingAnalyzer:
    """
    Analysis class for powder diffraction data acquired with 2D detectors
    Measures diffraction ring radii and performs camera length and wavelength calibration
    """
    
    def __init__(self, config_file='machine_config.yaml'):
        """Load detector parameters from config file"""
        with open(config_file, 'r') as f:
            content = f.read()
            config = yaml.safe_load(content)
        
        self.nx = config['first_dimensions_of_array']
        self.ny = config['second_dimensions_of_array']
        self.pixel_size_h = config['size_of_horizontal_pixels'] * 1e-6  # μm -> m
        self.pixel_size_v = config['size_of_vertical_pixels'] * 1e-6  # μm -> m
        # Initial value
        self._beam_center_x = config['x_pixel_coordinate_of_direct_beam']
        self._beam_center_y = config['y_pixel_coordinate_of_direct_beam']
        self.tilt_rotation = config['rotation_angle_of_tilting_plane']
        self.tilt_angle = config['angle_of_detector_tilt_in_plane']
        
        # Known d-spacings for CeO2 (A) - cubic, space group Fm-3m, a=5.411 A
        self.ceo2_d_spacings = CeO2_d_spacings
        
        # Initialize helper classes
        self.geometry_calc = GeometryCalculator(
            self.pixel_size_h, 
            self.pixel_size_v,
            self._beam_center_x,
            self._beam_center_y,
            self.tilt_rotation,
            self.tilt_angle
        )
        
        self.calibration_helper = CalibrationHelper(self.geometry_calc)
        self.visualization_helper = VisualizationHelper(
            self.geometry_calc,
            self._beam_center_x,
            self._beam_center_y
        )
        self.beam_center_finder = BeamCenterFinder()
    
    @property
    def beam_center_x(self):
        return self._beam_center_x
    
    @beam_center_x.setter
    def beam_center_x(self, value):
        self._beam_center_x = value
        # Update relative parameters
        if hasattr(self, 'geometry_calc'):
            self.geometry_calc.beam_center_x = value
        if hasattr(self, 'visualization_helper'):
            self.visualization_helper.beam_center_x = value
    
    @property
    def beam_center_y(self):
        return self._beam_center_y
    
    @beam_center_y.setter
    def beam_center_y(self, value):
        self._beam_center_y = value
        # Update relative parameters
        if hasattr(self, 'geometry_calc'):
            self.geometry_calc.beam_center_y = value
        if hasattr(self, 'visualization_helper'):
            self.visualization_helper.beam_center_y = value
        
    def load_image(self, file_path):
        """Load TIFF file"""
        self.image = load_tiff(file_path)
        return self.image
    
    def find_beam_center_from_ring(self, image=None, initial_center=None, 
                                   n_angles=36, min_ring_radius=None, max_ring_radius=None,
                                   outlier_threshold=2.5, debug=False):
        """
        Refine beam center using the first strong diffraction ring
        
        To avoid the effects of detector gaps and beam stoppers,
        the centroid is calculated from the positions of the first strong peaks
        appearing in each direction
        
        Parameters:
        -----------
        image : ndarray, optional
            Image to analyze (uses self.image if None)
        initial_center : tuple, optional
            Initial beam center (x, y). Uses self.beam_center_x/y if None
        n_angles : int
            Number of directions to sample
        min_ring_radius : int, optional
            Minimum search radius (pixels). Auto-determined from image size if None
        max_ring_radius : int, optional
            Maximum search radius (pixels). Auto-determined from image size if None
        outlier_threshold : float
            Outlier removal threshold (multiples of standard deviation)
        debug : bool
            Whether to display debug information
        
        Returns:
        --------
        beam_center_x : float
            Refined beam center x-coordinate (column)
        beam_center_y : float
            Refined beam center y-coordinate (row)
        """
        if image is None:
            image = self.image
        
        if initial_center is None:
            initial_center = (self.beam_center_x, self.beam_center_y)
        
        # Delegate to BeamCenterFinder
        return self.beam_center_finder.find_beam_center_from_ring(
            image, initial_center, n_angles, min_ring_radius, max_ring_radius,
            outlier_threshold, debug
        )
    
    def calculate_radial_profile(self, image=None, num_bins=None):
        """
        Calculate radial integration profile as a function of distance from beam center
        
        Parameters:
        -----------
        image : ndarray, optional
            Image to analyze (uses self.image if None)
        num_bins : int, optional
            Number of bins (uses maximum radius of image if None)
        
        Returns:
        --------
        radii : ndarray
            Distance from beam center (in pixels)
        intensities : ndarray
            Average intensity at each distance
        """
        if image is None:
            image = self.image
        
        # Calculate distance from beam center for each pixel
        # Explicitly create coordinates using meshgrid
        ny, nx = image.shape
        x = np.arange(nx)
        y = np.arange(ny)
        x_grid, y_grid = np.meshgrid(x, y)
        
        # Distance from beam center
        # x_grid[i,j] = j (column index), y_grid[i,j] = i (row index)
        # beam_center_x = column coordinate, beam_center_y = row coordinate
        r = np.sqrt((x_grid - self.beam_center_x)**2 + (y_grid - self.beam_center_y)**2)
        
        # Radial bin width (0.2 pixel steps for high precision)
        radial_bin_width = 0.2
        
        # Bin distances into fine bins
        r_binned = (r / radial_bin_width).astype(int)
        
        # Determine maximum radius
        max_radius_bins = int(r.max() / radial_bin_width) + 1
        if num_bins is None:
            num_bins = max_radius_bins
        
        # Perform radial integration (manual binning)
        radial_sum = np.bincount(r_binned.ravel(), weights=image.ravel(), minlength=max_radius_bins)
        radial_count = np.bincount(r_binned.ravel(), minlength=max_radius_bins)
        
        # Calculate average avoiding locations with zero counts
        radial_profile = np.zeros(max_radius_bins)
        nonzero = radial_count > 0
        radial_profile[nonzero] = radial_sum[nonzero] / radial_count[nonzero]
        
        # Actual radius values (in pixels)
        radii = np.arange(max_radius_bins) * radial_bin_width
        
        self.radii = radii
        self.intensities = radial_profile
        
        return radii, radial_profile
    
    def find_ring_positions(self, min_distance=5, prominence=None, height=None, 
                           refine_with_lorentzian=True, fit_width=10):
        """
        Detect peaks (diffraction rings) from radial profile
        
        Parameters:
        -----------
        min_distance : int or float
            Minimum distance between peaks (in pixels)
        prominence : float, optional
            Peak prominence
        height : float, optional
            Minimum peak height
        refine_with_lorentzian : bool, optional
            Whether to refine peak positions with Lorentzian function (default: False)
        fit_width : float, optional
            Width of the range used for Lorentzian fitting (in pixels)
            Only effective when refine_with_lorentzian=True
        
        Returns:
        --------
        peak_positions : ndarray
            Detected peak positions (in pixels)
            Refined positions if refine_with_lorentzian=True
        peak_properties : dict
            Peak property information
        """
        if prominence is None:
            # Automatically set prominence (5 times the median)
            prominence = np.median(self.intensities) * 5
        
        # Convert min_distance to index units considering radial bin width
        # Automatically calculate bin width from the spacing of self.radii
        radial_bin_width = self.radii[1] - self.radii[0] if len(self.radii) > 1 else 1.0
        min_distance_bins = int(min_distance / radial_bin_width)
        
        peaks, properties = find_peaks(
            self.intensities,
            distance=min_distance_bins,
            prominence=prominence,
            height=height
        )
        
        self.peak_indices = peaks  # Also save indices (needed when radial bins are fine)
        self.peak_positions = self.radii[peaks]
        self.peak_properties = properties
        
        # Refine with Lorentzian function if requested
        if refine_with_lorentzian and len(peaks) > 0:
            try:
                refined_positions, fit_params, fit_quality = self.refine_peaks_with_lorentzian(
                    fit_width=fit_width
                )
                # Return refined positions
                return refined_positions, properties
            except Exception as e:
                print(f"Warning: Lorentzian fitting failed: {e}")
                print("Returning original peak positions.")
                return self.peak_positions, properties
        
        return self.peak_positions, properties
    
    def refine_peaks_with_lorentzian(self, fit_width=None, min_peak_height=None):
        """
        Refine peak positions by fitting with Lorentzian function
        
        Parameters:
        -----------
        fit_width : float, optional
            Width around each peak used for fitting (in pixels)
            Auto-determined (half of peak spacing) if None
        min_peak_height : float, optional
            Minimum peak height for fitting
            Fits all peaks if None
        
        Returns:
        --------
        refined_positions : ndarray
            Refined peak positions (in pixels)
        fit_parameters : list of dict
            Fitting parameters for each peak
        fit_quality : list of float
            Fitting quality for each peak (R-squared values)
        """
        if not hasattr(self, 'peak_positions') or len(self.peak_positions) == 0:
            raise ValueError("No peaks detected. Please run find_ring_positions() first.")
        
        refined_positions = []
        fit_parameters = []
        fit_quality = []
        
        # Determine default fit width
        if fit_width is None:
            # Use half the average peak spacing
            if len(self.peak_positions) > 1:
                peak_spacings = np.diff(self.peak_positions)
                fit_width = np.mean(peak_spacings) / 2
            else:
                fit_width = 10  # Default value
        
        for i, (peak_idx, peak_pos) in enumerate(zip(self.peak_indices, self.peak_positions)):
            # Check peak height
            peak_height = self.intensities[peak_idx]
            if min_peak_height is not None and peak_height < min_peak_height:
                # Skip fitting and use original position
                refined_positions.append(peak_pos)
                fit_parameters.append(None)
                fit_quality.append(0.0)
                continue
            
            # Determine fitting range
            fit_range_start = peak_pos - fit_width
            fit_range_end = peak_pos + fit_width
            
            # Extract data within fitting range
            mask = (self.radii >= fit_range_start) & (self.radii <= fit_range_end)
            x_data = self.radii[mask]
            y_data = self.intensities[mask]
            
            if len(x_data) < 5:  # If insufficient data points for fitting
                refined_positions.append(peak_pos)
                fit_parameters.append(None)
                fit_quality.append(0.0)
                continue
            
            # Estimate initial parameters
            background_estimate = np.min(y_data)
            amplitude_estimate = peak_height - background_estimate
            gamma_estimate = fit_width / 4  # 1/4 of range as initial value
            
            initial_params = [amplitude_estimate, peak_pos, gamma_estimate, background_estimate]
            
            # Set parameter bounds
            bounds = (
                [0, peak_pos - fit_width, 0.1, 0],  # Lower bounds
                [np.inf, peak_pos + fit_width, fit_width, np.max(y_data)]  # Upper bounds
            )
            
            try:
                # Fit with Lorentzian function
                popt, pcov = curve_fit(
                    lorentzian, x_data, y_data,
                    p0=initial_params,
                    bounds=bounds,
                    maxfev=5000
                )
                
                amplitude, center, gamma, background = popt
                
                # Calculate fitting quality (R-squared value)
                y_fit = lorentzian(x_data, *popt)
                ss_res = np.sum((y_data - y_fit)**2)
                ss_tot = np.sum((y_data - np.mean(y_data))**2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                refined_positions.append(center)
                fit_parameters.append({
                    'amplitude': amplitude,
                    'center': center,
                    'gamma': gamma,
                    'background': background,
                    'fwhm': 2 * gamma,  # Full width at half maximum
                    'covariance': pcov
                })
                fit_quality.append(r_squared)
                
            except (RuntimeError, ValueError) as e:
                # Use original position if fitting fails
                print(f"Warning: Fitting failed for peak {i+1} (position {peak_pos:.2f}): {e}")
                refined_positions.append(peak_pos)
                fit_parameters.append(None)
                fit_quality.append(0.0)
        
        refined_positions = np.array(refined_positions)
        
        # Save refined peak positions
        self.refined_peak_positions = refined_positions
        self.peak_fit_parameters = fit_parameters
        self.peak_fit_quality = fit_quality        
        return refined_positions, fit_parameters, fit_quality
    
    def plot_peak_fits(self, peak_indices=None, fit_width=None):
        """
        Visualize Lorentzian fitting results
        
        Parameters:
        -----------
        peak_indices : list of int, optional
            Indices of peaks to display (displays all if None)
        fit_width : float, optional
            Width of display range (in pixels)
        
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Generated figure object
        """
        if not hasattr(self, 'refined_peak_positions'):
            raise ValueError("No refined peaks available. Please run refine_peaks_with_lorentzian() first.")
        
        # Determine peaks to display
        if peak_indices is None:
            peak_indices = range(len(self.refined_peak_positions))
        
        # Determine default fit width
        if fit_width is None:
            if len(self.peak_positions) > 1:
                peak_spacings = np.diff(self.peak_positions)
                fit_width = np.mean(peak_spacings) / 2
            else:
                fit_width = 10
        
        n_peaks = len(peak_indices)
        n_cols = min(3, n_peaks)
        n_rows = (n_peaks + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_peaks == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, peak_idx in enumerate(peak_indices):
            ax = axes[idx]
            
            original_pos = self.peak_positions[peak_idx]
            refined_pos = self.refined_peak_positions[peak_idx]
            fit_params = self.peak_fit_parameters[peak_idx]
            quality = self.peak_fit_quality[peak_idx]
            
            # Data range
            fit_range_start = original_pos - fit_width
            fit_range_end = original_pos + fit_width
            mask = (self.radii >= fit_range_start) & (self.radii <= fit_range_end)
            x_data = self.radii[mask]
            y_data = self.intensities[mask]
            
            # Plot measured data
            ax.plot(x_data, y_data, 'o', markersize=4, alpha=0.6, label='Measured data')
            
            # Plot fitting results
            if fit_params is not None:
                x_fit = np.linspace(fit_range_start, fit_range_end, 200)
                y_fit = lorentzian(x_fit, 
                                  fit_params['amplitude'],
                                  fit_params['center'],
                                  fit_params['gamma'],
                                  fit_params['background'])
                ax.plot(x_fit, y_fit, 'r-', linewidth=2, label='Lorentzian fit')
                
                # Display original and refined peak positions
                ax.axvline(original_pos, color='blue', linestyle='--', alpha=0.5, 
                          label=f'Original: {original_pos:.2f}')
                ax.axvline(refined_pos, color='red', linestyle='--', alpha=0.5,
                          label=f'Refined: {refined_pos:.2f}')
                
                # Display fitting parameters
                param_text = (f'R² = {quality:.4f}\n'
                            f'FWHM = {fit_params["fwhm"]:.2f} px\n'
                            f'Shift = {refined_pos - original_pos:.3f} px')
                ax.text(0.05, 0.95, param_text, transform=ax.transAxes,
                       verticalalignment='top', bbox=dict(boxstyle='round', 
                       facecolor='wheat', alpha=0.5), fontsize=9)
            else:
                ax.axvline(original_pos, color='blue', linestyle='--', alpha=0.5,
                          label=f'Position: {original_pos:.2f}')
                ax.text(0.05, 0.95, 'Fit failed', transform=ax.transAxes,
                       verticalalignment='top', bbox=dict(boxstyle='round',
                       facecolor='lightcoral', alpha=0.5))
            
            ax.set_xlabel('Radius (pixels)')
            ax.set_ylabel('Intensity')
            ax.set_title(f'Peak {peak_idx + 1}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_peaks, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        return fig
    
    # Geometry calculation methods (delegated)
    def pixel_to_distance(self, pixel_radius):
        """Convert pixel radius to real-space distance (m)"""
        return self.geometry_calc.pixel_to_distance(pixel_radius)
    
    def calculate_two_theta(self, pixel_radius, camera_length, correct_tilt=True):
        """Calculate 2θ angle from pixel radius (with detector tilt correction)"""
        return self.geometry_calc.calculate_two_theta(pixel_radius, camera_length, correct_tilt)
    
    def calculate_two_theta_for_pixel(self, x_pixel, y_pixel, camera_length):
        """Calculate 2θ angle at specific pixel position (with detector tilt correction)"""
        return self.geometry_calc.calculate_two_theta_for_pixel(x_pixel, y_pixel, camera_length)
    
    def bragg_law(self, d_spacing, wavelength):
        """Calculate 2θ from Bragg's law"""
        return GeometryCalculator.bragg_law(d_spacing, wavelength)
    
    # Calibration methods (delegated)
    def calibrate_camera_length(self, peak_positions_pixel, d_spacings, wavelength):
        """Fit camera length from known d-spacings and measured peak positions"""
        return self.calibration_helper.calibrate_camera_length(
            peak_positions_pixel, d_spacings, wavelength
        )
    
    def calibrate_wavelength(self, peak_positions_pixel, d_spacings, camera_length):
        """Fit wavelength from known d-spacings and measured peak positions"""
        return self.calibration_helper.calibrate_wavelength(
            peak_positions_pixel, d_spacings, camera_length
        )
    
    def generate_calibration_report(self, peak_assignments, camera_length, wavelength, 
                                   correct_tilt=True, show_comparison=True):
        """Generate calibration results report"""
        return self.calibration_helper.generate_calibration_report(
            self.peak_positions, peak_assignments, camera_length, wavelength,
            correct_tilt, show_comparison
        )
    
    # Visualization methods (delegated)
    def plot_image_with_rings(self, vmin=None, vmax=None):
        """Display image with overlaid diffraction rings"""
        peak_positions = self.peak_positions if hasattr(self, 'peak_positions') else None
        return self.visualization_helper.plot_image_with_rings(
            self.image, peak_positions, vmin, vmax
        )
    
    def plot_radial_profile(self, show_peaks=True):
        """Plot radial profile"""
        peak_positions = self.peak_positions if hasattr(self, 'peak_positions') else None
        peak_indices = self.peak_indices if hasattr(self, 'peak_indices') else None
        return self.visualization_helper.plot_radial_profile(
            self.radii, self.intensities, peak_positions, peak_indices, show_peaks
        )
    
    def plot_tilt_correction_map(self, camera_length, subsample=5):
        """Visualize detector tilt correction effects as 2D map"""
        return self.visualization_helper.plot_tilt_correction_map(
            self.nx, self.ny, camera_length, subsample
        )
