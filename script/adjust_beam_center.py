# adjust_beam_center.py
"""
Interactive beam center adjustment tool with zoom, pan, and fine-tuning controls
Helps users find approximate beam center coordinates visually
"""

import numpy as np
import matplotlib.pyplot as plt
import tifffile as tif
import yaml
from matplotlib.widgets import Button, TextBox
from matplotlib.patches import Circle
import os


class BeamCenterAdjuster:
    """
    Interactive tool for adjusting beam center coordinates
    Features:
    - Mouse wheel zoom
    - Click and drag to pan
    - Arrow keys for fine adjustment
    - Real-time radial profile update
    """
    
    def __init__(self, image_path, config_file='machine_config.yaml'):
        """
        Parameters:
        -----------
        image_path : str
            Path to a diffraction image
        config_file : str
            Path to machine config file
        """
        self.image_path = image_path
        self.config_file = config_file
        
        # Load image
        self.image = tif.imread(image_path)
        print(f"Loaded image: {image_path}")
        print(f"Image shape: {self.image.shape}")
        
        # Load current config
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.beam_center_x = self.config['x_pixel_coordinate_of_direct_beam']
        self.beam_center_y = self.config['y_pixel_coordinate_of_direct_beam']
        
        # Store original values for reset
        self.original_x = self.beam_center_x
        self.original_y = self.beam_center_y
        
        print(f"Current beam center: ({self.beam_center_x:.2f}, {self.beam_center_y:.2f})")
        
        # For panning
        self.pan_active = False
        self.pan_start_x = None
        self.pan_start_y = None
        self.pan_xlim = None
        self.pan_ylim = None
        
        # For fine adjustment
        self.step_size = 1.0  # pixels per arrow key press
        
        # Circle radii
        self.circle_radii = [50, 100, 200, 300, 400]
        
    def calculate_radial_profile(self, center_x, center_y):
        """
        Calculate radial profile from given center
        
        Parameters:
        -----------
        center_x : float
            X coordinate of center
        center_y : float
            Y coordinate of center
        
        Returns:
        --------
        radii : ndarray
            Distance from center (pixels)
        intensities : ndarray
            Average intensity at each distance
        """
        ny, nx = self.image.shape
        y, x = np.ogrid[:ny, :nx]
        
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        radial_bin_width = 0.5
        r_binned = (r / radial_bin_width).astype(int)
        
        max_radius_bins = int(r.max() / radial_bin_width) + 1
        
        radial_sum = np.bincount(r_binned.ravel(), weights=self.image.ravel(), 
                                minlength=max_radius_bins)
        radial_count = np.bincount(r_binned.ravel(), minlength=max_radius_bins)
        
        radial_profile = np.zeros(max_radius_bins)
        nonzero = radial_count > 0
        radial_profile[nonzero] = radial_sum[nonzero] / radial_count[nonzero]
        
        radii = np.arange(max_radius_bins) * radial_bin_width
        
        return radii, radial_profile
    
    def create_interactive_plot(self):
        """
        Create interactive plot for beam center adjustment
        """
        # Create figure with two subplots
        self.fig = plt.figure(figsize=(18, 8))
        
        # Left: Image with crosshair (larger)
        self.ax_img = plt.subplot(1, 2, 1)
        
        # Right: Radial profile
        self.ax_profile = plt.subplot(1, 2, 2)
        
        # Display image
        vmax = np.percentile(self.image, 99.5)
        vmin = np.percentile(self.image, 0.5)
        
        self.im = self.ax_img.imshow(self.image, cmap='viridis', vmin=vmin, vmax=vmax, 
                                     origin='lower', interpolation='nearest')
        
        # Draw initial crosshair
        self.vline = self.ax_img.axvline(self.beam_center_x, color='red', linewidth=2, 
                                        alpha=0.8, label='Beam center', zorder=10)
        self.hline = self.ax_img.axhline(self.beam_center_y, color='red', linewidth=2, 
                                        alpha=0.8, zorder=10)
        
        # Draw concentric circles
        self.circles = []
        for radius in self.circle_radii:
            circle = Circle((self.beam_center_x, self.beam_center_y), radius,
                          color='yellow', fill=False, linewidth=1.5, 
                          alpha=0.6, linestyle='--', zorder=5)
            self.ax_img.add_patch(circle)
            self.circles.append(circle)
        
        self.ax_img.set_xlabel('X (pixels)', fontsize=12)
        self.ax_img.set_ylabel('Y (pixels)', fontsize=12)
        self.ax_img.set_title('Diffraction Pattern - Left click: move center | Right click: pan | Scroll: zoom', 
                             fontsize=13, fontweight='bold')
        self.ax_img.legend(loc='upper right')
        
        # Calculate and plot initial radial profile
        radii, intensities = self.calculate_radial_profile(
            self.beam_center_x, self.beam_center_y
        )
        self.radial_line, = self.ax_profile.plot(radii, intensities, 'b-', linewidth=1.5)
        
        self.ax_profile.set_xlabel('Distance from beam center (pixels)', fontsize=12)
        self.ax_profile.set_ylabel('Intensity (a.u.)', fontsize=12)
        self.ax_profile.set_title('Radial Profile', fontsize=13, fontweight='bold')
        self.ax_profile.grid(True, alpha=0.3)
        
        # Add text box for coordinates and instructions
        self.coord_text = self.ax_img.text(
            0.02, 0.98, 
            f'Center: ({self.beam_center_x:.2f}, {self.beam_center_y:.2f})\n'
            f'Step size: {self.step_size:.1f} px',
            transform=self.ax_img.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
            fontsize=10,
            fontweight='bold',
            family='monospace'
        )
        
        # ==================== UI Layout Configuration ====================
        # Refined color scheme
        color_increment = '#90EE90'      # Light green (softer)
        color_decrement = '#FFB6C1'      # Light pink (softer)
        color_save = '#4CAF50'           # Material green
        color_reset = '#FF6B6B'          # Coral red
        color_zoom = '#64B5F6'           # Blue
        color_textbox_bg = 'white'       # Clean white background
        
        # Layout parameters
        left_margin = 0.08
        box_height = 0.045
        box_spacing = 0.008
        group_spacing = 0.035
        
        # Fine adjustment row (bottom)
        fine_adjust_y = 0.02
        btn_width_small = 0.03
        textbox_width = 0.09
        
        # Action buttons row (top)
        action_btn_y = 0.08
        btn_width_large = 0.10
        
        # ==================== X Coordinate Controls ====================
        x_group_start = left_margin
        ax_x_minus = plt.axes([x_group_start, fine_adjust_y, btn_width_small, box_height])
        ax_x_box = plt.axes([x_group_start + btn_width_small + box_spacing, fine_adjust_y, 
                            textbox_width, box_height])
        ax_x_plus = plt.axes([x_group_start + btn_width_small + box_spacing + textbox_width + box_spacing, 
                             fine_adjust_y, btn_width_small, box_height])
        
        self.btn_x_minus = Button(ax_x_minus, '◀', color=color_decrement, hovercolor='#FFD1DC')
        self.x_box = TextBox(ax_x_box, 'X: ', initial=f'{self.beam_center_x:.2f}', 
                            color=color_textbox_bg, label_pad=0.01)
        self.btn_x_plus = Button(ax_x_plus, '▶', color=color_increment, hovercolor='#C1FFC1')
        
        # Styling for X coordinate controls
        self.x_box.label.set_fontsize(10)
        self.x_box.label.set_fontweight('bold')
        
        # ==================== Y Coordinate Controls ====================
        y_group_start = x_group_start + btn_width_small + box_spacing + textbox_width + box_spacing + btn_width_small + group_spacing
        ax_y_minus = plt.axes([y_group_start, fine_adjust_y, btn_width_small, box_height])
        ax_y_box = plt.axes([y_group_start + btn_width_small + box_spacing, fine_adjust_y, 
                            textbox_width, box_height])
        ax_y_plus = plt.axes([y_group_start + btn_width_small + box_spacing + textbox_width + box_spacing, 
                             fine_adjust_y, btn_width_small, box_height])
        
        self.btn_y_minus = Button(ax_y_minus, '◀', color=color_decrement, hovercolor='#FFD1DC')
        self.y_box = TextBox(ax_y_box, 'Y: ', initial=f'{self.beam_center_y:.2f}',
                            color=color_textbox_bg, label_pad=0.01)
        self.btn_y_plus = Button(ax_y_plus, '▶', color=color_increment, hovercolor='#C1FFC1')
        
        # Styling for Y coordinate controls
        self.y_box.label.set_fontsize(10)
        self.y_box.label.set_fontweight('bold')
        
        # ==================== Step Size Controls ====================
        step_group_start = y_group_start + btn_width_small + box_spacing + textbox_width + box_spacing + btn_width_small + group_spacing
        ax_step_minus = plt.axes([step_group_start, fine_adjust_y, btn_width_small, box_height])
        ax_step_box = plt.axes([step_group_start + btn_width_small + box_spacing, fine_adjust_y, 
                               textbox_width * 0.8, box_height])
        ax_step_plus = plt.axes([step_group_start + btn_width_small + box_spacing + textbox_width * 0.8 + box_spacing, 
                                fine_adjust_y, btn_width_small, box_height])
        
        self.btn_step_minus = Button(ax_step_minus, '−', color=color_decrement, hovercolor='#FFD1DC')
        self.step_box = TextBox(ax_step_box, 'Step: ', initial=f'{self.step_size:.1f}',
                               color=color_textbox_bg, label_pad=0.01)
        self.btn_step_plus = Button(ax_step_plus, '+', color=color_increment, hovercolor='#C1FFC1')
        
        # Styling for step size controls
        self.step_box.label.set_fontsize(10)
        self.step_box.label.set_fontweight('bold')
        
        # ==================== Connect Fine Adjustment Events ====================
        self.x_box.on_submit(self.on_text_submit)
        self.y_box.on_submit(self.on_text_submit)
        self.step_box.on_submit(self.on_step_submit)
        
        self.btn_x_minus.on_clicked(lambda event: self.adjust_x(-self.step_size))
        self.btn_x_plus.on_clicked(lambda event: self.adjust_x(self.step_size))
        self.btn_y_minus.on_clicked(lambda event: self.adjust_y(-self.step_size))
        self.btn_y_plus.on_clicked(lambda event: self.adjust_y(self.step_size))
        self.btn_step_minus.on_clicked(lambda event: self.adjust_step(-0.5))
        self.btn_step_plus.on_clicked(lambda event: self.adjust_step(0.5))
        
        # ==================== Action Buttons ====================
        action_spacing = 0.015
        ax_save = plt.axes([left_margin, action_btn_y, btn_width_large, box_height])
        ax_reset = plt.axes([left_margin + btn_width_large + action_spacing, action_btn_y, 
                            btn_width_large, box_height])
        ax_zoom_reset = plt.axes([left_margin + 2 * (btn_width_large + action_spacing), action_btn_y, 
                                 btn_width_large, box_height])
        
        self.btn_save = Button(ax_save, '💾 Save Config', color=color_save, hovercolor='#66BB6A')
        self.btn_reset = Button(ax_reset, '↺ Reset Center', color=color_reset, hovercolor='#FF8A80')
        self.btn_zoom_reset = Button(ax_zoom_reset, '🔍 Reset Zoom', color=color_zoom, hovercolor='#90CAF9')
        
        # Styling for action buttons
        for btn in [self.btn_save, self.btn_reset, self.btn_zoom_reset]:
            btn.label.set_fontsize(10)
            btn.label.set_fontweight('bold')
        
        self.btn_save.on_clicked(self.save_config)
        self.btn_reset.on_clicked(self.reset_center)
        self.btn_zoom_reset.on_clicked(self.reset_zoom)
        
        # Connect mouse and keyboard events
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_motion)
        self.cid_scroll = self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        
        # Print keyboard shortcuts
        print("\n" + "="*70)
        print("Keyboard Shortcuts:")
        print("="*70)
        print("  Arrow keys       : Move beam center by step size")
        print("  Shift + Arrows   : Move beam center by 10x step size")
        print("  +/- (numpad)     : Increase/decrease step size")
        print("  r                : Reset zoom")
        print("  h                : Reset beam center")
        print("  s                : Save configuration")
        print("="*70 + "\n")
        
        plt.show()
    
    def on_mouse_press(self, event):
        """Handle mouse button press"""
        if event.inaxes != self.ax_img:
            return
        
        if event.button == 1:  # Left click - move beam center
            self.beam_center_x = event.xdata
            self.beam_center_y = event.ydata
            self.update_plot()
        
        elif event.button == 3:  # Right click - start panning
            self.pan_active = True
            self.pan_start_x = event.xdata
            self.pan_start_y = event.ydata
            self.pan_xlim = self.ax_img.get_xlim()
            self.pan_ylim = self.ax_img.get_ylim()
            self.fig.canvas.set_cursor(1)  # Hand cursor
    
    def on_mouse_release(self, event):
        """Handle mouse button release"""
        if event.button == 3:  # Right click released
            self.pan_active = False
            self.fig.canvas.set_cursor(0)  # Default cursor
    
    def on_mouse_motion(self, event):
        """Handle mouse motion for panning"""
        if not self.pan_active or event.inaxes != self.ax_img:
            return
        
        if event.xdata is None or event.ydata is None:
            return
        
        # Calculate pan offset
        dx = self.pan_start_x - event.xdata
        dy = self.pan_start_y - event.ydata
        
        # Update axis limits
        xlim = [self.pan_xlim[0] + dx, self.pan_xlim[1] + dx]
        ylim = [self.pan_ylim[0] + dy, self.pan_ylim[1] + dy]
        
        self.ax_img.set_xlim(xlim)
        self.ax_img.set_ylim(ylim)
        self.fig.canvas.draw_idle()
    
    def on_scroll(self, event):
        """Handle mouse wheel scroll for zooming"""
        if event.inaxes != self.ax_img:
            return
        
        # Get current axis limits
        xlim = self.ax_img.get_xlim()
        ylim = self.ax_img.get_ylim()
        
        # Get cursor position
        xdata = event.xdata
        ydata = event.ydata
        
        # Zoom factor
        zoom_factor = 1.2 if event.button == 'down' else 1/1.2
        
        # Calculate new limits centered on cursor
        x_range = (xlim[1] - xlim[0]) * zoom_factor
        y_range = (ylim[1] - ylim[0]) * zoom_factor
        
        x_ratio = (xdata - xlim[0]) / (xlim[1] - xlim[0])
        y_ratio = (ydata - ylim[0]) / (ylim[1] - ylim[0])
        
        new_xlim = [xdata - x_range * x_ratio, xdata + x_range * (1 - x_ratio)]
        new_ylim = [ydata - y_range * y_ratio, ydata + y_range * (1 - y_ratio)]
        
        self.ax_img.set_xlim(new_xlim)
        self.ax_img.set_ylim(new_ylim)
        self.fig.canvas.draw_idle()
    
    def on_key_press(self, event):
        """Handle keyboard shortcuts"""
        if event.key is None:
            return
        
        # Calculate step (10x with shift modifier)
        step = self.step_size * 10 if event.key.startswith('shift') else self.step_size
        key = event.key.replace('shift+', '')
        
        # Arrow keys for beam center adjustment
        if key == 'left':
            self.adjust_x(-step)
        elif key == 'right':
            self.adjust_x(step)
        elif key == 'up':
            self.adjust_y(step)
        elif key == 'down':
            self.adjust_y(-step)
        
        # +/- for step size adjustment
        elif key in ['+', '=']:
            self.adjust_step(0.5)
        elif key in ['-', '_']:
            self.adjust_step(-0.5)
        
        # r for reset zoom
        elif key == 'r':
            self.reset_zoom(None)
        
        # h for reset beam center (home)
        elif key == 'h':
            self.reset_center(None)
        
        # s for save
        elif key == 's':
            self.save_config(None)
    
    def adjust_x(self, delta):
        """Adjust X coordinate by delta"""
        self.beam_center_x += delta
        self.update_plot()
    
    def adjust_y(self, delta):
        """Adjust Y coordinate by delta"""
        self.beam_center_y += delta
        self.update_plot()
    
    def adjust_step(self, delta):
        """Adjust step size by delta"""
        self.step_size = max(0.1, self.step_size + delta)
        self.step_box.set_val(f'{self.step_size:.1f}')
        self.update_coordinate_text()
        print(f"Step size: {self.step_size:.1f} pixels")
    
    def on_text_submit(self, text):
        """Handle text box input for precise coordinates"""
        try:
            self.beam_center_x = float(self.x_box.text)
            self.beam_center_y = float(self.y_box.text)
            self.update_plot()
        except ValueError:
            print("Invalid input. Please enter numeric values.")
            self.x_box.set_val(f'{self.beam_center_x:.2f}')
            self.y_box.set_val(f'{self.beam_center_y:.2f}')
    
    def on_step_submit(self, text):
        """Handle step size text box input"""
        try:
            self.step_size = max(0.1, float(self.step_box.text))
            self.step_box.set_val(f'{self.step_size:.1f}')
            self.update_coordinate_text()
        except ValueError:
            print("Invalid step size. Please enter a positive number.")
            self.step_box.set_val(f'{self.step_size:.1f}')
    
    def update_coordinate_text(self):
        """Update only the coordinate text without full plot update"""
        self.coord_text.set_text(
            f'Center: ({self.beam_center_x:.2f}, {self.beam_center_y:.2f})\n'
            f'Step size: {self.step_size:.1f} px'
        )
        self.fig.canvas.draw_idle()
    
    def update_plot(self):
        """Update crosshair, circles, and radial profile"""
        # Update crosshair
        self.vline.set_xdata([self.beam_center_x, self.beam_center_x])
        self.hline.set_ydata([self.beam_center_y, self.beam_center_y])
        
        # Update circles
        for circle in self.circles:
            circle.center = (self.beam_center_x, self.beam_center_y)
        
        # Update coordinate text
        self.update_coordinate_text()
        
        # Update text boxes (without triggering events)
        self.x_box.set_val(f'{self.beam_center_x:.2f}')
        self.y_box.set_val(f'{self.beam_center_y:.2f}')
        
        # Update radial profile
        radii, intensities = self.calculate_radial_profile(
            self.beam_center_x, self.beam_center_y
        )
        self.radial_line.set_data(radii, intensities)
        self.ax_profile.relim()
        self.ax_profile.autoscale_view()
        
        self.fig.canvas.draw_idle()
        
        print(f"Center: ({self.beam_center_x:.2f}, {self.beam_center_y:.2f})")
    
    def reset_center(self, event):
        """Reset to original config values"""
        self.beam_center_x = self.original_x
        self.beam_center_y = self.original_y
        self.update_plot()
        print("Reset to original beam center")
    
    def reset_zoom(self, event):
        """Reset zoom to show full image"""
        ny, nx = self.image.shape
        self.ax_img.set_xlim(0, nx)
        self.ax_img.set_ylim(0, ny)
        self.fig.canvas.draw_idle()
        print("Reset zoom")
    
    def save_config(self, event):
        """Save updated beam center to config file"""
        # Update config
        self.config['x_pixel_coordinate_of_direct_beam'] = float(self.beam_center_x)
        self.config['y_pixel_coordinate_of_direct_beam'] = float(self.beam_center_y)
        
        # Backup original file
        backup_file = self.config_file + '.backup'
        if os.path.exists(self.config_file):
            import shutil
            shutil.copy2(self.config_file, backup_file)
            print(f"Backup saved to: {backup_file}")
        
        # Save new config
        with open(self.config_file, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
        
        print(f"\n{'='*70}")
        print(f"✓ Beam center saved to {self.config_file}!")
        print(f"  x_pixel_coordinate_of_direct_beam: {self.beam_center_x:.6f}")
        print(f"  y_pixel_coordinate_of_direct_beam: {self.beam_center_y:.6f}")
        print(f"{'='*70}\n")
        
        # Update original values for reset button
        self.original_x = self.beam_center_x
        self.original_y = self.beam_center_y


def main():
    """
    Main function for interactive beam center adjustment
    """
    print("="*70)
    print("Beam Center Adjustment Tool")
    print("="*70)
    print("\nThis tool helps you find the approximate beam center coordinates.")
    print("The beam center will be further refined automatically during calibration.\n")
    
    # Ask for image file
    while True:
        image_path = input("Enter path to a diffraction image (.tif): ").strip()
        if os.path.exists(image_path):
            break
        else:
            print(f"File not found: {image_path}")
            print("Please try again.\n")
    
    # Ask for config file
    default_config = 'machine_config.yaml'
    config_path = input(f"Enter path to config file (default: {default_config}): ").strip()
    if not config_path:
        config_path = default_config
    
    if not os.path.exists(config_path):
        print(f"\nWarning: Config file '{config_path}' not found.")
        create_new = input("Create new config file? (y/n): ").strip().lower()
        if create_new in ['y', 'yes']:
            # Create default config
            default_config_data = {
                'first_dimensions_of_array': 981,
                'second_dimensions_of_array': 1043,
                'size_of_horizontal_pixels': 172,
                'size_of_vertical_pixels': 172,
                'x_pixel_coordinate_of_direct_beam': 490.0,
                'y_pixel_coordinate_of_direct_beam': 520.0,
                'rotation_angle_of_tilting_plane': 0.0,
                'angle_of_detector_tilt_in_plane': 0.0
            }
            with open(config_path, 'w') as f:
                yaml.dump(default_config_data, f, default_flow_style=False)
            print(f"Created default config: {config_path}")
        else:
            print("Exiting.")
            return
    
    print("\n" + "="*70)
    print("Mouse Controls:")
    print("="*70)
    print("  Left click       : Move beam center to cursor position")
    print("  Right click+drag : Pan the image")
    print("  Scroll wheel     : Zoom in/out (centered on cursor)")
    print("="*70)
    print("\nFine Adjustment:")
    print("="*70)
    print("  +/- buttons      : Adjust values by small increments")
    print("  Text boxes       : Enter precise coordinates manually")
    print("  Step size        : Control increment for arrow keys and buttons")
    print("="*70 + "\n")
    
    # Create adjuster and show interactive plot
    adjuster = BeamCenterAdjuster(image_path, config_path)
    adjuster.create_interactive_plot()


if __name__ == '__main__':
    main()