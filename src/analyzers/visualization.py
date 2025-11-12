"""
Module providing visualization functions
Plots diffraction patterns and profiles
"""
import numpy as np
import matplotlib.pyplot as plt


class VisualizationHelper:
    """
    Class responsible for plotting functions
    """
    
    def __init__(self, geometry_calculator, beam_center_x, beam_center_y):
        """
        Parameters:
        -----------
        geometry_calculator : GeometryCalculator
            Object for performing geometry calculations
        beam_center_x : float
            X-coordinate of beam center
        beam_center_y : float
            Y-coordinate of beam center
        """
        self.geometry_calc = geometry_calculator
        self.beam_center_x = beam_center_x
        self.beam_center_y = beam_center_y
    
    def plot_image_with_rings(self, image, peak_positions=None, vmin=None, vmax=None):
        """
        Display image with overlaid diffraction rings
        
        Parameters:
        -----------
        image : ndarray
            Diffraction pattern image
        peak_positions : ndarray, optional
            Detected peak positions (in pixels)
        vmin : float, optional
            Minimum value for colormap
        vmax : float, optional
            Maximum value for colormap
        
        Returns:
        --------
        fig, ax : matplotlib figure and axes
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        if vmax is None:
            vmax = np.percentile(image, 99.5)
        if vmin is None:
            vmin = np.percentile(image, 0.5)
        
        ax.imshow(image, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
        
        # Plot beam center
        ax.plot(self.beam_center_x, self.beam_center_y, 'r+', 
               markersize=20, markeredgewidth=2)
        
        # Display detected rings as circles
        if peak_positions is not None:
            for i, r in enumerate(peak_positions):
                circle = plt.Circle(
                    (self.beam_center_x, self.beam_center_y), 
                    r, 
                    color='red', 
                    fill=False, 
                    linewidth=2,
                    alpha=0.7
                )
                ax.add_patch(circle)
                ax.text(
                    self.beam_center_x + r, 
                    self.beam_center_y, 
                    f'  R{i+1}', 
                    color='red', 
                    fontsize=10,
                    weight='bold'
                )
        
        ax.set_xlabel('X (pixels)', fontsize=12)
        ax.set_ylabel('Y (pixels)', fontsize=12)
        ax.set_title('Powder Diffraction Pattern with Detected Rings', fontsize=14)
        plt.tight_layout()
        
        return fig, ax
    
    def plot_radial_profile(self, radii, intensities, peak_positions=None, 
                           peak_indices=None, show_peaks=True):
        """
        Plot radial profile
        
        Parameters:
        -----------
        radii : ndarray
            Distance from beam center (in pixels)
        intensities : ndarray
            Intensity at each distance
        peak_positions : ndarray, optional
            Detected peak positions (in pixels)
        peak_indices : ndarray, optional
            Peak indices
        show_peaks : bool
            Whether to display peaks
        
        Returns:
        --------
        fig, ax : matplotlib figure and axes
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(radii, intensities, 'b-', linewidth=1.5, label='Radial profile')
        
        if show_peaks and peak_positions is not None:
            # Get accurate intensity values using peak indices
            if peak_indices is not None:
                peak_intensities = intensities[peak_indices]
            else:
                # Estimate from positions if indices are not available
                radial_bin_width = radii[1] - radii[0] if len(radii) > 1 else 1.0
                peak_indices = (peak_positions / radial_bin_width).astype(int)
                peak_indices = np.clip(peak_indices, 0, len(intensities) - 1)
                peak_intensities = intensities[peak_indices]
            
            ax.plot(
                peak_positions, 
                peak_intensities, 
                'ro', 
                markersize=10,
                label='Detected peaks'
            )
            
            # Add annotations to peak positions
            for i, (pos, intensity) in enumerate(zip(peak_positions, peak_intensities)):
                ax.annotate(
                    f'Peak {i+1}\nr={pos:.1f} px',
                    xy=(pos, intensity),
                    xytext=(10, 20),
                    textcoords='offset points',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
                )
        
        ax.set_xlabel('Radius from beam center (pixels)', fontsize=12)
        ax.set_ylabel('Intensity (a.u.)', fontsize=12)
        ax.set_title('Radial Integration Profile', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig, ax
    
    def plot_tilt_correction_map(self, nx, ny, camera_length, subsample=5):
        """
        Visualize detector tilt correction effects as 2D map
        
        Parameters:
        -----------
        nx : int
            Number of horizontal pixels on detector
        ny : int
            Number of vertical pixels on detector
        camera_length : float
            Camera length (m)
        subsample : int
            Subsampling rate (for improving calculation speed)
        
        Returns:
        --------
        fig, axes : matplotlib figure and axes
        """
        print("Calculating tilt correction map...")
        
        geom = self.geometry_calc
        
        # Create subsampled grid
        y_coords = np.arange(0, ny, subsample)
        x_coords = np.arange(0, nx, subsample)
        X, Y = np.meshgrid(x_coords, y_coords)
        
        # Calculate 2θ correction amount at each pixel
        delta_two_theta = np.zeros_like(X, dtype=float)
        
        total_pixels = X.size
        for idx, (x, y) in enumerate(zip(X.flat, Y.flat)):
            if idx % 1000 == 0:
                print(f"  Progress: {idx}/{total_pixels} ({100*idx/total_pixels:.1f}%)")
            
            # With tilt correction
            two_theta_corr = geom.calculate_two_theta_for_pixel(x, y, camera_length)
            
            # Without tilt correction
            dx = (x - self.beam_center_x) * geom.pixel_size_h
            dy = (y - self.beam_center_y) * geom.pixel_size_v
            r = np.sqrt(dx**2 + dy**2)
            two_theta_uncorr = np.arctan(r / camera_length)
            
            # Difference (degrees)
            delta_two_theta.flat[idx] = np.degrees(two_theta_corr - two_theta_uncorr)
        
        print("  Complete!")
        
        # Create plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Map of 2θ correction amount
        im1 = axes[0].imshow(
            delta_two_theta, 
            extent=[0, nx, 0, ny],
            cmap='RdBu_r', 
            origin='lower',
            vmin=-np.abs(delta_two_theta).max(),
            vmax=np.abs(delta_two_theta).max()
        )
        axes[0].plot(self.beam_center_x, self.beam_center_y, 'k+', 
                    markersize=20, markeredgewidth=2)
        axes[0].set_xlabel('X (pixels)', fontsize=12)
        axes[0].set_ylabel('Y (pixels)', fontsize=12)
        axes[0].set_title('Tilt Correction: Δ2θ (degrees)', fontsize=14)
        cbar1 = plt.colorbar(im1, ax=axes[0])
        cbar1.set_label('Δ2θ (deg)', fontsize=11)
        
        # Distance from beam center vs correction amount
        r_map = np.sqrt((X - self.beam_center_x)**2 + (Y - self.beam_center_y)**2)
        
        # Format data for plotting
        r_flat = r_map.flatten()
        delta_flat = delta_two_theta.flatten()
        
        # Calculate average per bin
        r_bins = np.linspace(0, r_flat.max(), 50)
        r_centers = (r_bins[:-1] + r_bins[1:]) / 2
        delta_binned = []
        delta_std = []
        
        for i in range(len(r_bins) - 1):
            mask = (r_flat >= r_bins[i]) & (r_flat < r_bins[i+1])
            if np.any(mask):
                delta_binned.append(np.mean(delta_flat[mask]))
                delta_std.append(np.std(delta_flat[mask]))
            else:
                delta_binned.append(0)
                delta_std.append(0)
        
        axes[1].plot(r_centers, delta_binned, 'b-', linewidth=2, label='Mean correction')
        axes[1].fill_between(
            r_centers, 
            np.array(delta_binned) - np.array(delta_std),
            np.array(delta_binned) + np.array(delta_std),
            alpha=0.3,
            label='Std. dev.'
        )
        axes[1].axhline(0, color='k', linestyle='--', alpha=0.5)
        axes[1].set_xlabel('Radius from beam center (pixels)', fontsize=12)
        axes[1].set_ylabel('Δ2θ (degrees)', fontsize=12)
        axes[1].set_title('Tilt Correction vs Radius', fontsize=14)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig, axes

