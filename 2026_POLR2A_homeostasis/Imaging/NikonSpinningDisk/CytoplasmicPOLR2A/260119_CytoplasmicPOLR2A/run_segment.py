from pathlib import Path

from glob import glob

from typing import List, Tuple, Union, Dict

import numpy as np

from aicsimageio import AICSImage, readers

from skimage import io, filters, exposure, feature, transform, util, morphology, segmentation, measure

from cellpose import models

from scipy import ndimage

from blimp.preprocessing.illumination_correction import IlluminationCorrection

def segment_cona(input_file,input_dir,output_dir,illumcorr_file):

    Intensity_Image = AICSImage(input_dir / input_file)

    illumination_correction = IlluminationCorrection(from_file=illumcorr_file)
    Intensity_Image_Corrected = illumination_correction.correct(Intensity_Image)

    Image_Data = Intensity_Image_Corrected.get_image_data('CYX', T=0, Z=0)

    ConA_Channel = 1
    DAPI_Channel = 5

    Image_Data_ConA_DAPI = np.asarray([Image_Data[ConA_Channel], Image_Data[DAPI_Channel]])
    Image_Data_DAPI = Image_Data[DAPI_Channel]

    model = models.Cellpose(model_type='cyto3')
    Cell_Masks, flows, styles, diams = model.eval(Image_Data_ConA_DAPI, diameter=150, channels = (1,2), channel_axis = 0)
    
    cell_curated_rprops = measure.regionprops(Cell_Masks)
    bboxes = [rp.bbox for rp in cell_curated_rprops]
    labels = [rp.label for rp in cell_curated_rprops]
    
    cell_curated_connected = np.zeros_like(Cell_Masks)
    
    for obj in range(1,len(bboxes)+1):
        
        # extract object as subarray and convert to binary
        ymin, xmin, ymax, xmax = bboxes[obj-1]
        extracted = Cell_Masks.copy()[ymin:ymax,xmin:xmax]
        extracted[extracted!=labels[obj-1]] = 0
        extracted = extracted > 0
    
        # measure
        extracted_relabeled = measure.label(extracted.astype(np.uint8))
        region_props = measure.regionprops(extracted_relabeled)
        largest_component = max(region_props, key=lambda prop: prop.area)
        print(f"obj = {obj}, n_connected = {len(region_props)}, largest_area = {largest_component.area}")
    
        # keep largest only
        if largest_component.area > 100:
            extracted = (extracted_relabeled == largest_component.label)
        else:
            extracted = np.zeros_like(extracted_relabeled)
        
        # fill interior holes (using 2D slices)
        extracted_filled = ndimage.binary_fill_holes(extracted)
        
        # find the non-zero indices and increment these with the "min" positions of the bbox
        nonzero_indices_local = extracted_filled.nonzero()
        nonzero_indices_global = tuple(np.array(arr) + offset for arr, offset in zip(nonzero_indices_local, [ymin,xmin]))
    
        # Place object labels only where there are nonzero elements (to avoid bounding box overlaps)
        cell_curated_connected[nonzero_indices_global] = obj

    Cell_Masks = cell_curated_connected.astype('i4')

    model = models.Cellpose(model_type='nuclei')
    Nuclei_Masks, flows, styles, diams = model.eval(Image_Data_DAPI, diameter=90, channels = (0,0))

    nuclei_curated_rprops = measure.regionprops(Nuclei_Masks)
    bboxes = [rp.bbox for rp in nuclei_curated_rprops]
    labels = [rp.label for rp in nuclei_curated_rprops]
    
    nuclei_curated_connected = np.zeros_like(Nuclei_Masks)
    
    for obj in range(1,len(bboxes)+1):
        
        # extract object as subarray and convert to binary
        ymin, xmin, ymax, xmax = bboxes[obj-1]
        extracted = Nuclei_Masks.copy()[ymin:ymax,xmin:xmax]
        extracted[extracted!=labels[obj-1]] = 0
        extracted = extracted > 0
    
        # measure
        extracted_relabeled = measure.label(extracted.astype(np.uint8))
        region_props = measure.regionprops(extracted_relabeled)
        largest_component = max(region_props, key=lambda prop: prop.area)
        print(f"obj = {obj}, n_connected = {len(region_props)}, largest_area = {largest_component.area}")
    
        # keep largest only
        if largest_component.area > 50:
            extracted = (extracted_relabeled == largest_component.label)
        else:
            extracted = np.zeros_like(extracted_relabeled)
        
        # fill interior holes (using 2D slices)
        extracted_filled = ndimage.binary_fill_holes(extracted)
        
        # find the non-zero indices and increment these with the "min" positions of the bbox
        nonzero_indices_local = extracted_filled.nonzero()
        nonzero_indices_global = tuple(np.array(arr) + offset for arr, offset in zip(nonzero_indices_local, [ymin,xmin]))
    
        # Place object labels only where there are nonzero elements (to avoid bounding box overlaps)
        nuclei_curated_connected[nonzero_indices_global] = obj

    
    Nuclei_Masks_Binary_Filtered = np.asarray(nuclei_curated_connected > 0).astype(np.int32)
    
    Nuclei_Masks_Numbered = Nuclei_Masks_Binary_Filtered * Cell_Masks

    Nuclei_Masks_Binary_Dilated = morphology.dilation(morphology.dilation(morphology.dilation(Nuclei_Masks_Binary_Filtered)))

    Cyto_Masks = (1 - Nuclei_Masks_Binary_Dilated) * Cell_Masks

    Masks = np.asarray([Cell_Masks, Nuclei_Masks_Numbered, Cyto_Masks]).astype('i4')
    Masks = np.expand_dims(Masks, axis = 1) # add z axis
    Masks = np.expand_dims(Masks, axis = 0) # add t axis


    AICSImage(Masks,
              channel_names=["Cells","Nuclei","Cytoplasm"],
              physical_pixel_sizes=Intensity_Image.physical_pixel_sizes).save(
        output_dir / Path("labels_" + Path(input_file).name)
    )

    return

def select_input_file(input_dir,index,extension="tiff"):
    input_files = glob(str(input_dir / ("*." + extension)))
    input_files.sort()
    print(input_files[index])
    return(input_files[index])


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(prog="run_spotdetection")

    # path setup

    parser.add_argument(
        "--id",
        type=int,
        help="batch id",
        required=True
    )

    parser.add_argument(
        "-i",
        "--input_dir",
        default="/srv/scratch/berrylab/z3536241/NikonSpinningDisk/250131_mACmChPOLR2A_smFISH/20250204_171458_755/OME-TIFF-SAMPLE"
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        default="/srv/scratch/berrylab/z3536241/NikonSpinningDisk/250131_mACmChPOLR2A_smFISH/20250204_171458_755/SPOTS_633"
    )

    parser.add_argument(
        "--illumination_correction_file",
        default="/srv/scratch/berrylab/z3532965/systems_Ti2/20240814_dTAG-DIS3/ILLUMCORR/202408_20X.pkl"
    )

    args = parser.parse_args()
    input_file = select_input_file(Path(args.input_dir),index=args.id)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    illumcorr_file = Path(args.illumination_correction_file)

    segment_cona(input_file,input_dir,output_dir,illumcorr_file)
