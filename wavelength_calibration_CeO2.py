import os
from src.iterative_calibration import IterativeCalibrator
import numpy as np
import yaml
from src.create_detail_settings import create_detail_settings

detail_settings_filename = 'detail_settings.yaml'

def main():
    prominent_factor = 5
    min_distance = 5
    convergence_threshold = 1e-5
    max_iterations = 20

    # Ask input directory
    while True:
        image_directory = input("Input the directory of the images: ")
        if not os.path.exists(image_directory):
            print("The directory does not exist. Please try again.")
            continue

        image_paths = [os.path.join(image_directory, f) for f in os.listdir(image_directory) if f.endswith('.tif')]
        if len(image_paths) == 0:
            print("No images found in the directory. Please try again.")
        else:
            break

    # Ask input file names
    while True:
        image_file_names = input("Input the file names of the images (example: filename_00001.tif -> filename): ")
        image_paths = np.array([os.path.join(image_directory, f) for f in os.listdir(image_directory) if f.endswith('.tif') and f.startswith(image_file_names)])
        image_paths = np.sort(image_paths)
        if len(image_paths) == 0:
            print("No images found in the directory. Please try again.")
        else:
            break

    # Search for the config file in the directory
    config_file = os.path.join(image_directory, 'machine_config.yaml')
    if not os.path.exists(config_file):
        print("The config file does not exist. Use default config file.")
        config_file = 'machine_config.yaml'
    else:
        print("The config file exists. Use the config file.")

    # Search for the detail settings file in the directory
    detail_settings_file = os.path.join(image_directory, detail_settings_filename)
    if not os.path.exists(detail_settings_file):
        print("The detail settings file does not exist. Use default detail settings.")
    else:
        print("The detail settings file exists. Use the detail settings file.")
        with open(detail_settings_file, 'r') as f:
            content = f.read()
            detail_settings = yaml.safe_load(content)
            if 'prominent_factor' in detail_settings:
                prominent_factor = detail_settings['prominent_factor']
            if 'min_distance' in detail_settings:
                min_distance = detail_settings['min_distance']
            if 'convergence_threshold' in detail_settings:
                convergence_threshold = detail_settings['convergence_threshold']
            if 'max_iterations' in detail_settings:
                max_iterations = detail_settings['max_iterations']

    # Ask for the initial wavelength
    while True:
        initial_wavelength = input("Input the initial wavelength: ")
        try:
            initial_wavelength = float(initial_wavelength)
            break
        except ValueError:
            print("Please input a valid number. Please try again.")

    set_num = 5
    camera_length_list = np.array([130, 195, 260, 325, 390])
    while True:
        camera_length_answer = input("Default set of camera length is 130mm, 195mm, 260mm, 325mm, 390mm. Is it OK? (y/n): ")
        if camera_length_answer == 'y' or camera_length_answer == 'Y' or camera_length_answer == 'yes' or camera_length_answer == 'Yes':
            break
        elif camera_length_answer == 'n' or camera_length_answer == 'N' or camera_length_answer == 'no' or camera_length_answer == 'No':
            set_num = int(input("Input the number of sets: "))
            camera_length_list = np.array([float(input(f"Input the camera length {i+1} (mm): ")) for i in range(set_num)])
            if set_num > image_paths.shape[0]:
                print("The number of sets is greater than the number of images. Please try again.")
            else:
                break
        else:
            print("Please input y or n. Please try again.")
    
    # Initialize the calibrator
    calibrator = IterativeCalibrator(
        config_file=config_file,
        initial_wavelength=initial_wavelength,
        set_num=set_num,
        camera_length_list=camera_length_list
    )
    
    # Set image paths
    image_paths = image_paths[:set_num]
    
    # Load images
    calibrator.load_images(image_paths)
    
    # Detect peaks in all images (auto-detect beam center for each image)
    calibrator.detect_peaks_all_images(
        min_distance=min_distance, 
        prominence_factor=prominent_factor,
        auto_detect_center=True,  # Refine beam center from the first diffraction ring
        debug_beam_center=False  # Set to True to display beam center detection details
    )
    
    # Automatically assign peaks
    calibrator.assign_peaks_auto(num_peaks=8)
    
    # Visualize peak detection results
    print("\nVisualizing peak detection results...")
    fig_profiles, axes_profiles = calibrator.plot_radial_profiles_all_images()
    fig_profiles.savefig(os.path.join(image_directory, 'radial_profiles_all_images.png'), dpi=150, bbox_inches='tight')
    print(f"  -> Saved to {os.path.join(image_directory, 'radial_profiles_all_images.png')}")
    
    # Prompt user to confirm
    print("\n" + "="*70)
    print("[IMPORTANT] Please verify the peak detection results")
    print("="*70)
    print("Open radial_profiles_all_images.png and")
    print("verify that the red vertical lines correctly capture the CeO2 peak positions.")
    print("\nIf they are not correct:")
    print("  - Adjust prominence_factor (default: 5)")
    print("  - Adjust min_distance (default: 5)")
    while True:
        input_answer = input("The results are correct? (y/n): ")
        if input_answer == 'y' or input_answer == 'Y' or input_answer == 'yes' or input_answer == 'Yes':
            break
        elif input_answer == 'n' or input_answer == 'N' or input_answer == 'no' or input_answer == 'No':
            create_detail_settings(os.path.join(image_directory, detail_settings_filename))
            print("The detail settings file has been created. Please adjust the parameters and try again.")
            return
        else:
            print("Please input y or n. Please try again.")
    
    # Execute iterative calibration
    results = calibrator.iterate_calibration(
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold  # Convergence is considered when slope is within 0.001 of 1.0
    )
    
    # Display final report
    calibrator.print_final_report(results)
    
    # Save results
    calibrator.save_results(results, os.path.join(image_directory, 'calibration_results_iterative.json'))
    
    # Plot results
    fig = calibrator.plot_results(results)
    fig.savefig(os.path.join(image_directory, 'iterative_calibration_results.png'), dpi=300, bbox_inches='tight')
    print(f"\nSaved graph to {os.path.join(image_directory, 'iterative_calibration_results.png')}")
        
    return calibrator, results


if __name__ == '__main__':
    calibrator, results = main()

