"""
Iterative calibration method

Simultaneously refines wavelength and camera length using data acquired at multiple camera lengths.

Procedure:
1. Determine camera length from each image with fixed initial wavelength
2. Apply linear fitting correction to (initial camera length, measured camera length)
3. Recalculate wavelength using corrected camera lengths
4. Repeat steps 1-3 until convergence
"""

import numpy as np
import matplotlib.pyplot as plt
from src.ring_analysis import PowderRingAnalyzer
import json
from src.CeO2_d_spacings import CeO2_d_spacings


class IterativeCalibrator:
    """
    Iterative calibration class
    """
    
    def __init__(self, config_file: str, initial_wavelength: float, set_num: int, camera_length_list: np.array):
        """Initialization"""
        # Initialize analyzer
        self.analyzer = PowderRingAnalyzer(config_file)
        
        self.set_num = set_num
        self.wavelength_init = initial_wavelength
        self.camera_length_init = camera_length_list / 1000.0  # mm -> m
        
        print(f"Initial values:")
        print(f"  Set number: {self.set_num}")
        print(f"  Initial wavelength: {self.wavelength_init:.4f} A")
        print(f"  Initial camera length: {self.camera_length_init*1000} mm")
        
        # CeO2 d-spacings (reflection planes to use)
        self.ceo2_peaks = CeO2_d_spacings
        
        # Record convergence history
        self.history = {
            'iteration': [],
            'wavelength': [],
            'camera_lengths': [],
            'linear_fit_slope': [],
            'linear_fit_intercept': [],
            'residuals': []
        }
    
    def load_images(self, image_paths: np.array):
        """
        Load images
        
        Parameters:
        -----------
        image_paths : list of str
            Image file paths
        """
        if len(image_paths) != self.set_num:
            raise ValueError(f"The number of images must be {self.set_num}")
        
        self.image_paths = image_paths
        self.images = []
        
        for path in image_paths:
            img = self.analyzer.load_image(path)
            self.images.append(img)

    def detect_peaks_all_images(self, min_distance=5, prominence_factor=5,
                                auto_detect_center=True, debug_beam_center=False,
                                refine_with_lorentzian=True, fit_width=10):
        """
        Detect peaks in all images (auto-detect beam center for each image)
        
        Parameters:
        -----------
        min_distance : int
            Minimum distance between peaks (pixels)
        prominence_factor : float
            Prominence factor of peaks
        auto_detect_center : bool
            Auto-detect beam center from the first ring for each image
        debug_beam_center : bool
            Show debug information of beam center detection
        refine_with_lorentzian : bool
            Refine peak positions with Lorentzian function
        fit_width : float
            Width of the Lorentzian function (pixels)
        Returns:
        --------
        all_peaks : list of ndarray
            List of detected peak positions in each image
        """
        print("\nDetecting peaks in all images...")
        print(f"Parameters: min_distance={min_distance}, prominence_factor={prominence_factor}")
        print(f"Beam center: {'Auto-detect from the first ring' if auto_detect_center else 'Use machine_config.yaml'}")
        
        self.all_peaks = []
        self.all_radii = []
        self.all_intensities = []
        self.detected_beam_centers = []  # Save detected beam centers
        
        # Save initial beam center
        original_beam_center_x = self.analyzer.beam_center_x
        original_beam_center_y = self.analyzer.beam_center_y
        
        for i, img in enumerate(self.images):
            # Set image
            self.analyzer.image = img
            
            # Refine beam center (use the first ring)
            if auto_detect_center:
                print(f"Image {i+1}: Refining beam center...")
                center_x, center_y = self.analyzer.find_beam_center_from_ring(
                    img, 
                    initial_center=(original_beam_center_x, original_beam_center_y),
                    debug=debug_beam_center
                )
                self.analyzer.beam_center_x = center_x
                self.analyzer.beam_center_y = center_y
                self.detected_beam_centers.append((center_x, center_y))
            else:
                self.detected_beam_centers.append((original_beam_center_x, original_beam_center_y))
            
            # Calculate radial profile
            radii, intensities = self.analyzer.calculate_radial_profile()
            self.all_radii.append(radii)
            self.all_intensities.append(intensities)
            
            # Detect peaks
            prominence = np.percentile(intensities, 50) * prominence_factor
            peaks, props = self.analyzer.find_ring_positions(
                min_distance=min_distance,
                prominence=prominence,
                refine_with_lorentzian=refine_with_lorentzian,
                fit_width=fit_width
            )
            
            self.all_peaks.append(peaks)
        
        return self.all_peaks
    
    def assign_peaks_auto(self, num_peaks=4):
        """
        Assign peaks to CeO2 reflections automatically
        
        Parameters:
        -----------
        num_peaks : int
            Number of peaks to use
        
        Returns:
        --------
        peak_assignments : dict
            {peak_index: (hkl, d_spacing)}
        """
        ceo2_list = list(self.ceo2_peaks.items())[:num_peaks]
        peak_assignments = {i: ceo2_list[i] for i in range(num_peaks)}
        
        self.peak_assignments = peak_assignments
        return peak_assignments
    
    def calibrate_camera_lengths_fixed_wavelength(self, wavelength):
        """
        Determine camera length from each image with fixed wavelength
        
        Parameters:
        -----------
        wavelength : float
            Fixed wavelength (A)
        
        Returns:
        --------
        camera_lengths : ndarray
            Camera lengths obtained from each image (m)
        camera_lengths_err : ndarray
            Camera length errors (m)
        """
        camera_lengths = []
        camera_lengths_err = []
        
        assigned_d_spacings = [self.peak_assignments[i][1] for i in range(len(self.peak_assignments))]
        
        print(f"\nDetermining camera length with fixed wavelength {wavelength:.6f} A...")
        
        for i, peaks in enumerate(self.all_peaks):
            assigned_positions = [peaks[j] for j in range(len(self.peak_assignments)) if j < len(peaks)]
            
            # Calibrate camera length
            L, L_err = self.analyzer.calibrate_camera_length(
                assigned_positions,
                assigned_d_spacings[:len(assigned_positions)],
                wavelength
            )
            
            camera_lengths.append(L)
            camera_lengths_err.append(L_err)
            
            print(f"  Image {i+1}: L = {L*1000:.6f} ± {L_err*1000:.6f} mm")
        
        return np.array(camera_lengths), np.array(camera_lengths_err)
    
    def linear_fit_camera_lengths(self, camera_lengths_measured, camera_lengths_reference):
        """
        Linear fitting of (reference camera length, measured camera length)
        
        Parameters:
        -----------
        camera_lengths_measured : ndarray
            Measured camera lengths (m)
        camera_lengths_reference : ndarray
            Reference camera lengths (m)
        
        Returns:
        --------
        slope : float
            Fitting slope
        intercept : float
            Fitting intercept
        camera_lengths_corrected : ndarray
            Corrected camera lengths (m)
        residuals : float
            Residuals
        """
        # Linear fitting: y = a*x + b
        # x: reference camera length, y: measured camera length
        x = camera_lengths_reference
        y = camera_lengths_measured
        
        # Least squares method
        coeffs = np.polyfit(x, y, 1)
        slope, intercept = coeffs
        
        # Corrected camera lengths (values on the fitting line)
        camera_lengths_corrected = slope * x + intercept
        
        # Residuals
        residuals = np.sqrt(np.mean((y - camera_lengths_corrected)**2))
        
        return slope, intercept, camera_lengths_corrected, residuals
    
    def calibrate_wavelength_first_image(self, camera_length):
        """
        Determine wavelength using the first image
        
        Parameters:
        -----------
        camera_length : float
            Camera length to use (m)
        
        Returns:
        --------
        wavelength : float
            Determined wavelength (A)
        wavelength_err : float
            Wavelength error (A)
        """
        assigned_d_spacings = [self.peak_assignments[i][1] for i in range(len(self.peak_assignments))]
        assigned_positions = [self.all_peaks[0][j] for j in range(len(self.peak_assignments)) 
                             if j < len(self.all_peaks[0])]
        
        wavelength, wavelength_err = self.analyzer.calibrate_wavelength(
            assigned_positions,
            assigned_d_spacings[:len(assigned_positions)],
            camera_length
        )
        
        return wavelength, wavelength_err
    
    def iterate_calibration(self, max_iterations=20, convergence_threshold=1e-4):
        """
        Execute iterative calibration
        
        Parameters:
        -----------
        max_iterations : int
            Maximum number of iterations
        convergence_threshold : float
            Convergence threshold (difference between slope and 1, absolute value of intercept)
        
        Returns:
        --------
        results : dict
            Calibration results
        """
        print("\n" + "="*70)
        print("Iterative calibration started")
        print("="*70)
        
        # Initial values
        wavelength = self.wavelength_init
        camera_lengths_reference = self.camera_length_init.copy()
        
        for iteration in range(max_iterations):
            print(f"\n{'='*70}")
            print(f"Iteration {iteration + 1}/{max_iterations}")
            print(f"{'='*70}")
            print(f"Current wavelength: {wavelength:.8f} A")
            print(f"Reference camera length: {camera_lengths_reference*1000} mm")
            
            # Step 1: Determine camera length with fixed wavelength
            camera_lengths_measured, camera_lengths_err = self.calibrate_camera_lengths_fixed_wavelength(
                wavelength
            )
            
            # Step 2: Linear fitting (use the values in initial_guess.yaml as reference)
            slope, intercept, camera_lengths_corrected, residuals = self.linear_fit_camera_lengths(
                camera_lengths_measured, camera_lengths_reference
            )
            
            # Step 3: Correct wavelength from slope/intercept
            # Small angle approximation: L ∝ 1/λ
            wavelength_new = wavelength * slope
            
            # Estimate Error from first image
            _, wavelength_err = self.calibrate_wavelength_first_image(
                camera_lengths_measured[0]
            )
            
            self.history['iteration'].append(iteration + 1)
            self.history['wavelength'].append(wavelength_new)
            self.history['camera_lengths'].append(camera_lengths_measured.copy())
            self.history['linear_fit_slope'].append(slope)
            self.history['linear_fit_intercept'].append(intercept)
            self.history['residuals'].append(residuals)
            
            # Convergence check
            slope_deviation = abs(slope - 1.0)
            intercept_abs = abs(intercept)
            wavelength_change = abs(wavelength_new - wavelength)
            
            # Detect abnormal values
            if wavelength_new < 0.1 or wavelength_new > 5.0:
                print(f"\n[WARNING] Wavelength is abnormal ({wavelength_new:.8f} A)")
                print("Possible problems with peak detection.")
                print("Check radial_profiles_all_images.png.")
                break
            
            if slope > 2.0 or slope < 0.5:
                print(f"\n[WARNING] Slope is abnormal ({slope:.4f})")
                print("Initial camera length is too far off, or there is a problem with peak detection.")
                break
            
            if camera_lengths_corrected.max() > 10.0:  # 10 m = 10000 mm
                print(f"\n[WARNING] Camera length is too large ({camera_lengths_corrected.max()*1000:.0f} mm)")
                print("Possible problems with peak detection.")
                break
            
            # Detect divergence
            if iteration > 3:
                recent_slopes = self.history['linear_fit_slope'][-3:]
                if all(s > 1.5 for s in recent_slopes) or all(s < 0.7 for s in recent_slopes):
                    print(f"\n[WARNING] Divergence detected (slope is not converging)")
                    print("Check peak assignments.")
                    break
            
            # Convergence check: slope is within 1±0.001
            # (intercept is systematic deviation, so not included in the check)
            if slope_deviation < convergence_threshold:
                print(f"\n[OK] Converged! (Iteration: {iteration + 1})")
                print(f"  Final slope: {slope:.8f}, intercept: {intercept*1000:.6f} mm")
                break
            
            # Update wavelength for next iteration
            wavelength = wavelength_new
        
        else:
            print(f"\nMaximum number of iterations ({max_iterations}) reached")
        
        # Final results (use measured camera lengths as final values)
        results = {
            'converged': iteration < max_iterations - 1,
            'iterations': iteration + 1,
            'final_wavelength': wavelength_new,
            'final_wavelength_err': wavelength_err,
            'final_energy_keV': 12.398 / wavelength_new,
            'final_camera_lengths_mm': camera_lengths_measured * 1000,
            'final_slope': slope,
            'final_intercept_mm': intercept * 1000,
            'final_residuals_mm': residuals * 1000,
            'detected_beam_centers': self.detected_beam_centers,
            'history': self.history
        }
        
        return results
    
    def plot_radial_profiles_all_images(self):
        """
        Plot radial profiles and peak positions for all images
        
        Returns:
        --------
        fig, axes : matplotlib figure and axes
        """
        n_images = len(self.images)
        fig, axes = plt.subplots(n_images, 1, figsize=(14, 6*n_images))
        
        # For single image case
        if n_images == 1:
            axes = [axes]
        
        print("\nCreating radial profiles for all images...")
        
        for i, peaks in enumerate(self.all_peaks):
            ax = axes[i]
            
            # Use saved radial profiles
            radii = self.all_radii[i]
            intensities = self.all_intensities[i]
            
            # Calculate radial bin width (0.2 pixel steps)
            radial_bin_width = radii[1] - radii[0] if len(radii) > 1 else 1.0
            
            ax.plot(radii, intensities, 'b-', linewidth=1.5, alpha=0.8, label='Radial profile')
            
            # Draw vertical lines at peak positions
            assigned_count = 0
            unassigned_count = 0
            for j, peak_pos in enumerate(peaks):
                if j < len(self.peak_assignments):
                    hkl, d = self.peak_assignments[j]
                    label = 'CeO2 peaks (used)' if assigned_count == 0 else None
                    ax.axvline(peak_pos, color='red', linestyle='--', linewidth=2, alpha=0.7, label=label)
                    assigned_count += 1
                    # Label for peak position   
                    peak_idx = int(peak_pos / radial_bin_width)
                    y_pos = intensities[peak_idx] if peak_idx < len(intensities) else intensities.max()
                    # ax.text(peak_pos, y_pos * 1.05, f'{hkl}\n{peak_pos:.1f}px', 
                    #        ha='center', va='bottom', fontsize=9, color='red', weight='bold',
                    #        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
                else:
                    label = 'Unused peaks' if unassigned_count == 0 else None
                    ax.axvline(peak_pos, color='gray', linestyle=':', linewidth=1, alpha=0.5, label=label)
                    unassigned_count += 1
                    ax.text(peak_pos, intensities.max() * 0.05, f'{peak_pos:.0f}', 
                           ha='center', va='bottom', fontsize=7, color='gray', alpha=0.7)
            
            # Set labels
            ax.set_xlabel('Distance from beam center (pixels)', fontsize=11)
            ax.set_ylabel('Intensity (a.u.)', fontsize=11)
            
            # Display camera length information
            camera_length_mm = self.camera_length_init[i] * 1000
            ax.set_title(f'Image {i+1}: {self.image_paths[i].split("/")[-1]} '
                        f'(Init. camera length: {camera_length_mm:.0f} mm, Detected peaks: {len(peaks)}, '
                        f'Used: {min(len(peaks), len(self.peak_assignments))})',
                        fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=9)
            
            # Adjust y-axis range (for better visibility of peaks)
            y_max = np.percentile(intensities, 99.5)
            ax.set_ylim(0, y_max * 3)
                    
        plt.tight_layout()
        
        return fig, axes
    
    def plot_results(self, results):
        """
        Plot results
        
        Parameters:
        -----------
        results : dict
            Calibration results
        """
        # Get number of iterations
        n_iterations = len(self.history['iteration'])
        
        # Calculate number of rows and columns for subplots
        n_cols = 3
        n_rows = (n_iterations + n_cols) // n_cols  # +1 for final, rounded up
        
        fig = plt.figure(figsize=(5*n_cols, 4*n_rows))
        
        x_ref = self.camera_length_init * 1000  # mm (reference camera lengths)
        
        # Plot each iteration
        for idx, iter_num in enumerate(self.history['iteration']):
            ax = plt.subplot(n_rows, n_cols, idx + 1)
            
            # Get measured camera lengths at this iteration
            y_measured = self.history['camera_lengths'][idx] * 1000  # mm
            
            # Get fitting line parameters
            slope = self.history['linear_fit_slope'][idx]
            intercept = self.history['linear_fit_intercept'][idx] * 1000  # mm
            y_fit = slope * x_ref + intercept
            
            # Plot ideal line (y=x)
            ax.plot([x_ref.min(), x_ref.max()], [x_ref.min(), x_ref.max()], 'k--', 
                    alpha=0.5, linewidth=1, label='y=x')
            
            # Plot measured data points
            ax.plot(x_ref, y_measured, 'o', markersize=6, label='Measured', color='C0')
            
            # Plot fitting line
            ax.plot(x_ref, y_fit, '-', linewidth=2, label=f'Fit: y={slope:.4f}x+{intercept:.2f}', 
                   color='C1')
            
            # Get wavelength at this iteration
            wavelength = self.history['wavelength'][idx]
            residual = self.history['residuals'][idx] * 1000  # mm
            
            ax.set_xlabel('Reference camera length (mm)', fontsize=10)
            ax.set_ylabel('Measured camera length (mm)', fontsize=10)
            ax.set_title(f'Iteration {iter_num}\nλ={wavelength:.6f} Å, RMSE={residual:.3f} mm', 
                        fontsize=11, fontweight='bold')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Plot final result in the last panel
        ax_final = plt.subplot(n_rows, n_cols, n_iterations + 1)
        
        # Use final wavelength to calculate final camera lengths
        wavelength_final = results['final_wavelength']
        camera_lengths_measured, _ = self.calibrate_camera_lengths_fixed_wavelength(wavelength_final)
        y_measured_final = camera_lengths_measured * 1000  # mm
        
        # Final fitting parameters
        slope_final = results['final_slope']
        intercept_final = results['final_intercept_mm']
        y_fit_final = slope_final * x_ref + intercept_final
        
        # Plot ideal line (y=x)
        ax_final.plot([x_ref.min(), x_ref.max()], [x_ref.min(), x_ref.max()], 'k--', 
                     alpha=0.5, linewidth=1, label='y=x')
        
        # Plot measured data points
        ax_final.plot(x_ref, y_measured_final, 'o', markersize=8, label='Measured', color='C2')
        
        # Plot fitting line
        ax_final.plot(x_ref, y_fit_final, '-', linewidth=2.5, 
                     label=f'Fit: y={slope_final:.4f}x+{intercept_final:.2f}', color='C3')
        
        ax_final.set_xlabel('Reference camera length (mm)', fontsize=10)
        ax_final.set_ylabel('Measured camera length (mm)', fontsize=10)
        ax_final.set_title(f'Final Result\nλ={wavelength_final:.6f} Å, E={results["final_energy_keV"]:.3f} keV', 
                          fontsize=11, fontweight='bold', color='red')
        ax_final.legend(loc='best', fontsize=8)
        ax_final.grid(True, alpha=0.3)
        
        # Hide unused subplots if any
        for idx in range(n_iterations + 1, n_rows * n_cols):
            ax = plt.subplot(n_rows, n_cols, idx + 1)
            ax.axis('off')
        
        plt.tight_layout()
        
        return fig
    
    def save_results(self, results, output_file: str):
        """
        Save results to a JSON file
        
        Parameters:
        -----------
        results : dict
            Calibration results
        output_file : str
            Name of the output file
        """
        # Convert numpy arrays to lists
        results_serializable = {
            'converged': results['converged'],
            'iterations': results['iterations'],
            'final_wavelength': float(results['final_wavelength']),
            'final_wavelength_err': float(results['final_wavelength_err']),
            'final_energy_keV': float(results['final_energy_keV']),
            'final_camera_lengths_mm': results['final_camera_lengths_mm'].tolist(),
            'final_slope': float(results['final_slope']),
            'final_intercept_mm': float(results['final_intercept_mm']),
            'final_residuals_mm': float(results['final_residuals_mm']),
            'detected_beam_centers': results['detected_beam_centers'],
            'history': {
                'iteration': results['history']['iteration'],
                'wavelength': results['history']['wavelength'],
                'camera_lengths': [cl.tolist() for cl in results['history']['camera_lengths']],
                'linear_fit_slope': results['history']['linear_fit_slope'],
                'linear_fit_intercept': results['history']['linear_fit_intercept'],
                'residuals': results['history']['residuals']
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_serializable, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to {output_file}")
    
    def print_final_report(self, results):
        """
        Display final report
        
        Parameters:
        -----------
        results : dict
            Calibration results
        """
        print("\n" + "="*70)
        print("Final calibration results")
        print("="*70)
        print(f"\nConvergence state: {'Success' if results['converged'] else 'Not converged'}")
        print(f"Number of iterations: {results['iterations']}")
        print(f"\nFinal wavelength: {results['final_wavelength']:.8f} ± {results['final_wavelength_err']:.8f} A")
        print(f"\nFinal camera lengths:")
        for i, L in enumerate(results['final_camera_lengths_mm']):
            print(f"  L{i+1} = {L:.6f} mm")
        print("="*70)

