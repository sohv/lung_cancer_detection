import os
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def load_itk_image(filename):
    """Load a .mhd file and return its image data, origin, and spacing."""
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
    return numpyImage, numpyOrigin, numpySpacing

def load_raw_image(raw_file, metadata):
    """Load a .raw file using metadata from a .mhd file."""
    dtype = np.dtype(metadata['ElementType'])
    shape = (int(metadata['Dim1']), int(metadata['Dim2']), int(metadata['Dim3']))
    with open(raw_file, 'rb') as file:
        image = np.fromfile(file, dtype=dtype)
    image = image.reshape(shape)
    return image

def normalize_and_clip(image):
    """Normalize and clip image data."""
    image = np.clip(image, -1000, 400)
    image = (image - image.min()) / (image.max() - image.min())
    return image

def process_and_save_scan(mhd_path, output_path):
    """Process a .mhd file and save the processed image and metadata."""
    try:
        image, origin, spacing = load_itk_image(str(mhd_path))
        image = normalize_and_clip(image)
        np.save(output_path, image)
        
        # Save metadata
        metadata = {
            'origin': origin.tolist(),
            'spacing': spacing.tolist()
        }
        np.save(output_path.with_suffix('.meta.npy'), metadata)
    except Exception as e:
        print(f"Error processing scan {mhd_path}: {str(e)}")

def process_raw_scan(raw_path, mhd_path, output_path):
    """Process a .raw file using the corresponding .mhd file for metadata."""
    try:
        metadata = read_mhd_metadata(mhd_path)
        image = load_raw_image(raw_path, metadata)
        image = normalize_and_clip(image)
        np.save(output_path, image)
        
        # Save metadata
        origin = np.zeros(3)  # Default value or derived from metadata
        spacing = np.array([float(metadata['ElementSpacingX']), float(metadata['ElementSpacingY']), float(metadata['ElementSpacingZ'])])
        metadata = {
            'origin': origin.tolist(),
            'spacing': spacing.tolist()
        }
        np.save(output_path.with_suffix('.meta.npy'), metadata)
    except Exception as e:
        print(f"Error processing scan {raw_path}: {str(e)}")

def read_mhd_metadata(mhd_file):
    """Read metadata from a .mhd file."""
    metadata = {}
    with open(mhd_file, 'r') as file:
        for line in file:
            if '=' in line:
                key, value = line.strip().split(' = ')
                metadata[key] = value
    return metadata

def process_all_scans(input_folder, output_folder):
    """Process all .mhd and .raw files in the input folder."""
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    mhd_files = list(input_path.glob('*.mhd'))
    raw_files = list(input_path.glob('*.raw'))
    
    for mhd_file in tqdm(mhd_files, desc="Processing MHD scans"):
        output_file = output_path / f"{mhd_file.stem}_processed.npy"
        if not output_file.exists():
            process_and_save_scan(mhd_file, output_file)
    
    for raw_file in tqdm(raw_files, desc="Processing RAW scans"):
        mhd_file = input_path / f"{raw_file.stem}.mhd"  # Corresponding MHD file
        if mhd_file.exists():
            output_file = output_path / f"{raw_file.stem}_processed.npy"
            if not output_file.exists():
                process_raw_scan(raw_file, mhd_file, output_file)

def create_dataset_df(processed_dir, annotations_file):
    """Create a DataFrame from processed images and merge with annotations."""
    processed_dir = Path(processed_dir)
    
    annotations = pd.read_csv(annotations_file)
    
    image_files = list(processed_dir.glob('*_processed.npy'))
    
    annotations.rename(columns={'seriesuid': 'series_uid'}, inplace=True)

    df = pd.DataFrame({
        'image_path': [str(f) for f in image_files],
        'series_uid': [f.stem.split('_')[0] for f in image_files]
    })
    
    df = df.merge(annotations, on='series_uid', how='left')
    
    return df

if __name__ == "__main__":
     # Define the base directory for relative paths
    base_dir = "C:/Users/Ashish Mahendran/Documents/lung_cancer_detection"
    
    # Convert absolute paths to relative paths
    input_folder = os.path.relpath("subset2", base_dir)
    output_folder = os.path.relpath("output_images/processed_scans3", base_dir)
    annotation_file = os.path.relpath("annotations.csv", base_dir)

    print("Input folder:", input_folder)
    print("Output folder:", output_folder)
    print("Annotation file:", annotation_file)
    
    # Check if input folder exists
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input folder not found: {input_folder}")
    
    # Process all scans
    print(f"Processing scans from: {input_folder}")
    print(f"Saving processed scans to: {output_folder}")
    process_all_scans(input_folder, output_folder)
    
    # Create dataset DataFrame
    print("Creating dataset DataFrame...")
    df = create_dataset_df(output_folder, annotation_file)
    
    # Save DataFrame
    dataset_csv_path = os.path.join(output_folder, "dataset2.csv")
    df.to_csv(dataset_csv_path, index=False)
    print(f"Dataset CSV created: {dataset_csv_path}")
    
    print("Processing complete.")
