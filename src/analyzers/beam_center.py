"""
Module providing beam center detection functions
Refines beam center using diffraction rings
"""
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


class BeamCenterFinder:
    """
    Class for detecting and refining beam center using diffraction rings
    """
    
    def __init__(self):
        """Initialization"""
        pass
    
    def find_beam_center_from_ring(self, image, initial_center, 
                                   n_angles=36, min_ring_radius=None, max_ring_radius=None,
                                   outlier_threshold=2.5, debug=False):
        """
        Refine beam center using the first strong diffraction ring
        
        To avoid the effects of detector gaps and beam stoppers,
        the centroid is calculated from the positions of the first strong peaks
        appearing in each direction
        
        Parameters:
        -----------
        image : ndarray
            Image to analyze
        initial_center : tuple
            Initial beam center (x, y)
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
        initial_cx, initial_cy = initial_center
        
        # Auto-determine search range (20% to 60% of maximum image size)
        ny, nx = image.shape
        max_possible_radius = min(nx, ny) // 2
        
        if min_ring_radius is None:
            min_ring_radius = int(max_possible_radius * 0.05)  # 5%
        if max_ring_radius is None:
            max_ring_radius = int(max_possible_radius * 0.7)   # 70%
                
        # Search for peaks in each direction
        angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
        peak_points = []
        
        for angle in angles:
            # Get intensity profile along line in this direction (0.5 pixel steps for high precision)
            radii = np.arange(min_ring_radius, max_ring_radius, 0.2)
            intensities = []
            
            for r in radii:
                x = initial_cx + r * np.cos(angle)
                y = initial_cy + r * np.sin(angle)
                
                # Check if within image bounds
                if 0 <= int(x) < image.shape[1] and 0 <= int(y) < image.shape[0]:
                    # Bilinear interpolation
                    intensity = self._bilinear_interpolation(image, x, y)
                    intensities.append(intensity)
                else:
                    intensities.append(0)
            
            intensities = np.array(intensities)
            
            # Detect first strong peak
            if len(intensities) > 10:
                peak_info = self._find_strongest_peak(intensities, radii)
                if peak_info is not None:
                    peak_radius, peak_intensity = peak_info
                    # Coordinates of this peak position
                    peak_x = initial_cx + peak_radius * np.cos(angle)
                    peak_y = initial_cy + peak_radius * np.sin(angle)
                    peak_points.append((peak_x, peak_y, peak_radius, peak_intensity))
        
        if len(peak_points) < 10:
            print(f"  Warning: Too few peaks detected ({len(peak_points)} points)")
            print(f"          Using initial beam center as is")
            return initial_cx, initial_cy
        
        peak_points = np.array(peak_points)
        peak_x = peak_points[:, 0]
        peak_y = peak_points[:, 1]
        peak_r = peak_points[:, 2]
        peak_intensity = peak_points[:, 3]
        
        if debug:
            self._print_debug_info(peak_points, peak_r, initial_cx, initial_cy)
        
        # Filter peaks
        peak_x_filtered, peak_y_filtered, peak_r_filtered = self._filter_outliers(
            peak_x, peak_y, peak_r, outlier_threshold, debug
        )
        
        # Check number of data points after filtering
        if len(peak_x_filtered) < 10:
            print(f"  Warning: Too few peaks after outlier removal ({len(peak_x_filtered)} points)")
            print(f"          Using initial beam center as is")
            return initial_cx, initial_cy
        
        r_median_filtered = np.median(peak_r_filtered)
        r_std_filtered = np.std(peak_r_filtered)
        
        # Determine center by circle fitting
        refined_cx, refined_cy = self._fit_circle(peak_x_filtered, peak_y_filtered)
        
        # Calculate change amount
        delta_x = refined_cx - initial_cx
        delta_y = refined_cy - initial_cy
        delta_total = np.sqrt(delta_x**2 + delta_y**2)
        
        # Check fitting quality (suspicious if correction is too large)
        max_shift = r_median_filtered * 0.3  # Within 30% of ring radius
        if delta_total > max_shift:
            print(f"  Warning: Beam center correction is too large ({delta_total:.1f} px > {max_shift:.1f} px)")
            print(f"          Using initial beam center as is")
            return initial_cx, initial_cy
        
        # Check fitting residuals
        fitted_r = np.sqrt((peak_x_filtered - refined_cx)**2 + (peak_y_filtered - refined_cy)**2)
        fit_residual_std = np.std(fitted_r)
        
        return refined_cx, refined_cy
    
    def _bilinear_interpolation(self, image, x, y):
        """
        Get value at arbitrary position in image using bilinear interpolation
        
        Parameters:
        -----------
        image : ndarray
            Image data
        x : float
            X-coordinate
        y : float
            Y-coordinate
        
        Returns:
        --------
        intensity : float
            Interpolated intensity value
        """
        x0, y0 = int(x), int(y)
        x1, y1 = min(x0 + 1, image.shape[1] - 1), min(y0 + 1, image.shape[0] - 1)
        
        dx, dy = x - x0, y - y0
        intensity = (1 - dx) * (1 - dy) * image[y0, x0] + \
                   dx * (1 - dy) * image[y0, x1] + \
                   (1 - dx) * dy * image[y1, x0] + \
                   dx * dy * image[y1, x1]
        return intensity
    
    def _find_strongest_peak(self, intensities, radii):
        """
        Detect strongest peak from intensity profile
        
        Parameters:
        -----------
        intensities : ndarray
            Intensity profile
        radii : ndarray
            Corresponding radius values
        
        Returns:
        --------
        peak_info : tuple or None
            (peak_radius, peak_intensity) or None
        """
        # Smoothing
        smoothed = gaussian_filter1d(intensities, sigma=2)
        
        # Remove baseline
        baseline = np.percentile(smoothed, 10)
        smoothed_corrected = smoothed - baseline
        
        # Peak detection (with more lenient threshold)
        peak_threshold = np.percentile(smoothed_corrected, 75)  # Top 25%
        peaks, props = find_peaks(smoothed_corrected, 
                                 prominence=peak_threshold * 0.2,
                                 height=peak_threshold * 0.5)
        
        if len(peaks) > 0:
            # Select strongest peak
            peak_heights = smoothed_corrected[peaks]
            strongest_peak_idx = peaks[np.argmax(peak_heights)]
            peak_radius = radii[strongest_peak_idx]
            peak_intensity = smoothed_corrected[strongest_peak_idx]
            return (peak_radius, peak_intensity)
        
        return None
    
    def _filter_outliers(self, peak_x, peak_y, peak_r, outlier_threshold, debug):
        """
        Filter peaks by removing outliers
        
        Parameters:
        -----------
        peak_x : ndarray
            X-coordinates of peaks
        peak_y : ndarray
            Y-coordinates of peaks
        peak_r : ndarray
            Radii of peaks
        outlier_threshold : float
            Multiples of standard deviation
        debug : bool
            Whether to display debug information
        
        Returns:
        --------
        peak_x_filtered : ndarray
            Filtered x-coordinates
        peak_y_filtered : ndarray
            Filtered y-coordinates
        peak_r_filtered : ndarray
            Filtered radii
        """
        # Radius clustering: identify the most frequently detected radius
        r_hist, r_bins = np.histogram(peak_r, bins=max(10, len(peak_r)//3))
        most_common_bin_idx = np.argmax(r_hist)
        most_common_r_center = (r_bins[most_common_bin_idx] + r_bins[most_common_bin_idx + 1]) / 2
        bin_width = r_bins[1] - r_bins[0]
        
        if debug:
            print(f"  [DEBUG] Most popular radius: {most_common_r_center:.1f} px (frequency: {r_hist[most_common_bin_idx]} points)")
            print(f"  [DEBUG] Histogram bin width: {bin_width:.1f} px")
        
        # Step 1: Coarse clustering (select around most popular bin)
        tolerance_coarse = max(bin_width * 2, 15.0)  # 2x bin width or minimum 15 pixels
        mask_coarse = np.abs(peak_r - most_common_r_center) < tolerance_coarse
        peak_r_coarse = peak_r[mask_coarse]
        
        # Step 2: Strict filtering based on standard deviation
        r_median = np.median(peak_r_coarse)
        r_std = np.std(peak_r_coarse)
        
        # Use only peaks within 2.5σ (more strict)
        tolerance_fine = outlier_threshold * r_std
        mask_fine = np.abs(peak_r - r_median) < tolerance_fine
        
        peak_x_filtered = peak_x[mask_fine]
        peak_y_filtered = peak_y[mask_fine]
        peak_r_filtered = peak_r[mask_fine]
        
        if debug:
            print(f"  [DEBUG] Step1 coarse clustering: {len(peak_r_coarse)} points (tolerance: ±{tolerance_coarse:.1f} px)")
            print(f"  [DEBUG] Step2 std-based: median={r_median:.1f}, std={r_std:.1f}, tolerance={tolerance_fine:.1f} px")
            print(f"  [DEBUG] Final number of peaks used: {len(peak_x_filtered)} points")
            print(f"  [DEBUG] Percentage removed: {(1 - len(peak_x_filtered)/len(peak_x))*100:.1f}%")
            print(f"  [DEBUG] Radius range used: {r_median-tolerance_fine:.1f} - {r_median+tolerance_fine:.1f} px")
        
        return peak_x_filtered, peak_y_filtered, peak_r_filtered
    
    def _fit_circle(self, peak_x, peak_y):
        """
        Fit circle center from peak positions using least squares method
        
        Parameters:
        -----------
        peak_x : ndarray
            X-coordinates of peaks
        peak_y : ndarray
            Y-coordinates of peaks
        
        Returns:
        --------
        center_x : float
            Circle center x-coordinate
        center_y : float
            Circle center y-coordinate
        """
        # Circle fitting
        # (x - cx)^2 + (y - cy)^2 = r^2
        # x^2 - 2*cx*x + cx^2 + y^2 - 2*cy*y + cy^2 = r^2
        # 2*cx*x + 2*cy*y = x^2 + y^2 - r^2 + cx^2 + cy^2
        
        A = np.column_stack([peak_x, peak_y, np.ones(len(peak_x))])
        b = peak_x**2 + peak_y**2
        
        # Least squares method
        params, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        
        center_x = params[0] / 2
        center_y = params[1] / 2
        
        return center_x, center_y
    
    def _print_debug_info(self, peak_points, peak_r, initial_cx, initial_cy):
        """
        Display debug information
        
        Parameters:
        -----------
        peak_points : ndarray
            Detected peak points
        peak_r : ndarray
            Peak radii
        initial_cx : float
            Initial beam center x-coordinate
        initial_cy : float
            Initial beam center y-coordinate
        """
        peak_x = peak_points[:, 0]
        peak_y = peak_points[:, 1]
        
        print(f"  [DEBUG] Total number of detected peaks: {len(peak_points)}")
        print(f"  [DEBUG] Radius range: {peak_r.min():.1f} - {peak_r.max():.1f} px")
        print(f"  [DEBUG] Radius mean±std: {peak_r.mean():.1f} ± {peak_r.std():.1f} px")
        
        # Check angular distribution of peaks
        angles_detected = np.arctan2(peak_y - initial_cy, peak_x - initial_cx)
        print(f"  [DEBUG] Angular distribution of peaks (degrees):")
        print(f"    Overall: {np.degrees(angles_detected).min():.1f} - {np.degrees(angles_detected).max():.1f}")
        
        # Radius histogram information (for checking excluded peaks)
        r_bins_detail = np.arange(peak_r.min(), peak_r.max()+5, 5)
        r_hist_detail, _ = np.histogram(peak_r, bins=r_bins_detail)
        print(f"  [DEBUG] Radius histogram of all peaks (5px bins):")
        for i, (bin_start, count) in enumerate(zip(r_bins_detail[:-1], r_hist_detail)):
            if count > 0:
                print(f"    {bin_start:.0f}-{bin_start+5:.0f} px: {count} points")

