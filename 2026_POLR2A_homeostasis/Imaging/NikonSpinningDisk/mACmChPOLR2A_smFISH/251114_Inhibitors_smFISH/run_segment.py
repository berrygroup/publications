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

    Stack = Intensity_Image_Corrected.get_image_data('ZCYX', T=0)

    s = []

    for z in Stack:
        s2 = []
        for c in z:
            rs = transform.rescale(c, 0.1)
            s2.append(rs)
        s.append(s2)

    Downscaled_Stack = np.asarray(s)
    Downscaled_Stack = util.img_as_int(Downscaled_Stack)

    ConA_Channel = 1
    DAPI_Channel = 3

    DownScaled_Stack_ConA_DAPI = np.asarray([Downscaled_Stack[:,ConA_Channel], Downscaled_Stack[:,DAPI_Channel]])
    DownScaled_Stack_DAPI = Downscaled_Stack[:,DAPI_Channel]

    model = models.Cellpose(model_type='cyto3')
    Downscaled_3D_masks, flows, styles, diams = model.eval(DownScaled_Stack_ConA_DAPI, diameter=50, do_3D = True, channels = (1,2), channel_axis = 0)

    Upscaled_3D_masks = transform.resize(Downscaled_3D_masks, [61, 2304, 2304], preserve_range = True, order = 0)

    Upscaled_3D_masks_eroded = []
    Upscaled_3D_CellFree_Eroded = []

    for z,each in enumerate(Upscaled_3D_masks):
        each_filtered = morphology.remove_small_objects(each, min_size=1000)
        each_boundaries = segmentation.find_boundaries(each_filtered)
        each_boundaries_dilated = ndimage.binary_dilation(each_boundaries, iterations = 50)
        each_boundaries_mask = (1-each_boundaries_dilated) * each
        Upscaled_3D_masks_eroded.append(each_boundaries_mask)
        CellFree = (each == 0)
        CellFreeErode = ndimage.binary_erosion(CellFree, iterations = 20)
        Upscaled_3D_CellFree_Eroded.append(CellFreeErode)

    Upscaled_3D_masks_eroded = np.asarray(Upscaled_3D_masks_eroded)
    Upscaled_3D_CellFree_Eroded = np.asarray(Upscaled_3D_CellFree_Eroded)

    Upscaled_3D_masks_CellFreeMasked = []

    for z,each in enumerate(Upscaled_3D_masks_eroded):
        each_step = each + 1
        Each_Renumbered = each_step * (each > 0)
        Upscaled_3D_masks_CellFreeMasked.append(Each_Renumbered + Upscaled_3D_CellFree_Eroded[z])

    Upscaled_3D_masks_CellFreeMasked = np.asarray(Upscaled_3D_masks_CellFreeMasked)

    ConA_Filtered = []

    for z in Stack[:,ConA_Channel]:
        z_f = filters.sato(filters.gaussian(z, sigma = 3), black_ridges = False)
        ConA_Filtered.append(z_f)
        
    ConA_Filtered = np.asarray(ConA_Filtered)

    Masked_Seeded_Watershed = []

    for z,each in enumerate(ConA_Filtered):
        ws = segmentation.watershed(image = each, 
                                    markers = Upscaled_3D_masks_CellFreeMasked[z])
        Masked_Seeded_Watershed.append(ws)
        
    Masked_Seeded_Watershed = np.asarray(Masked_Seeded_Watershed).astype('i4')

    Cell_Masks = (Masked_Seeded_Watershed - 1)

    
    cell_curated_rprops = measure.regionprops(Cell_Masks)
    bboxes = [rp.bbox for rp in cell_curated_rprops]
    labels = [rp.label for rp in cell_curated_rprops]
    
    cell_curated_connected = np.zeros_like(Cell_Masks)
    
    for obj in range(1,len(bboxes)+1):
        
        # extract object as subarray and convert to binary
        zmin, ymin, xmin, zmax, ymax, xmax = bboxes[obj-1]
        extracted = Cell_Masks.copy()[zmin:zmax,ymin:ymax,xmin:xmax]
        extracted[extracted!=labels[obj-1]] = 0
        extracted = extracted > 0
    
        # measure
        extracted_relabeled = measure.label(extracted.astype(np.uint8))
        region_props = measure.regionprops(extracted_relabeled)
        largest_component = max(region_props, key=lambda prop: prop.area)
        print(f"obj = {obj}, n_connected = {len(region_props)}, largest_area = {largest_component.area}")
    
        # keep largest only
        if largest_component.area > 5000:
            extracted = (extracted_relabeled == largest_component.label)
        else:
            extracted = np.zeros_like(extracted_relabeled)
        
        # fill interior holes (using 2D slices)
        extracted_filled = np.stack(
            [ndimage.binary_fill_holes(extracted[z,:,:]) for z in range(extracted.shape[0])])
        
        # find the non-zero indices and increment these with the "min" positions of the bbox
        nonzero_indices_local = extracted_filled.nonzero()
        nonzero_indices_global = tuple(np.array(arr) + offset for arr, offset in zip(nonzero_indices_local, [zmin,ymin,xmin]))
    
        # Place object labels only where there are nonzero elements (to avoid bounding box overlaps)
        cell_curated_connected[nonzero_indices_global] = obj

    Cell_Masks = cell_curated_connected.astype('i4')



    
    model = models.Cellpose(model_type='nuclei')
    Downscaled_3D_masks, flows, styles, diams = model.eval(DownScaled_Stack_DAPI, diameter=25, do_3D = True, channels = (0,0))
    Upscaled_3D_masks = transform.resize(Downscaled_3D_masks, [61, 2304, 2304], preserve_range = True, order = 0)

    Upscaled_3D_masks_eroded = []
    Upscaled_3D_CellFree_Eroded = []

    for z,each in enumerate(Upscaled_3D_masks):
        
        each_filtered = morphology.remove_small_objects(each, min_size=1000)
        each_boundaries = segmentation.find_boundaries(each_filtered)
        each_boundaries_dilated = ndimage.binary_dilation(each_boundaries, iterations = 20)
        each_boundaries_mask = (1-each_boundaries_dilated) * each
        Upscaled_3D_masks_eroded.append(each_boundaries_mask)
        
        
        CellFree = (each == 0)
        CellFreeErode = ndimage.binary_erosion(CellFree, iterations = 20)
        Upscaled_3D_CellFree_Eroded.append(CellFreeErode)

    Upscaled_3D_masks_eroded = np.asarray(Upscaled_3D_masks_eroded)
    Upscaled_3D_CellFree_Eroded = np.asarray(Upscaled_3D_CellFree_Eroded)

    Upscaled_3D_masks_CellFreeMasked = []

    for z,each in enumerate(Upscaled_3D_masks_eroded):    
        each_step = each + 1
        Each_Renumbered = each_step * (each > 0)
        Upscaled_3D_masks_CellFreeMasked.append(Each_Renumbered + Upscaled_3D_CellFree_Eroded[z])
        
    Upscaled_3D_masks_CellFreeMasked = np.asarray(Upscaled_3D_masks_CellFreeMasked)

    DAPI_Filter = []

    for z in Stack[:,DAPI_Channel]:
        z_f = filters.roberts(filters.gaussian(z, sigma = 3))
        DAPI_Filter.append(z_f)
        
    DAPI_Filter = np.asarray(DAPI_Filter)

    Masked_Seeded_Watershed = []

    for z,each in enumerate(DAPI_Filter):
        ws = segmentation.watershed(image = each, markers = Upscaled_3D_masks_CellFreeMasked[z])
        Masked_Seeded_Watershed.append(ws)
        
    Masked_Seeded_Watershed = np.asarray(Masked_Seeded_Watershed)

    Nuclei_Masks = Masked_Seeded_Watershed - 1
    Nuclei_Masks_Binary = Nuclei_Masks > 0
    nuclei_in = measure.label(Nuclei_Masks_Binary)
    
    nuclei_curated_rprops = measure.regionprops(nuclei_in)
    bboxes = [rp.bbox for rp in nuclei_curated_rprops]
    labels = [rp.label for rp in nuclei_curated_rprops]
    
    nuclei_curated_connected = np.zeros_like(nuclei_in)
    
    for obj in range(1,len(bboxes)+1):
        
        # extract object as subarray and convert to binary
        zmin, ymin, xmin, zmax, ymax, xmax = bboxes[obj-1]
        extracted = nuclei_in.copy()[zmin:zmax,ymin:ymax,xmin:xmax]
        extracted[extracted!=labels[obj-1]] = 0
        extracted = extracted > 0
    
        # measure
        extracted_relabeled = measure.label(extracted.astype(np.uint8))
        region_props = measure.regionprops(extracted_relabeled)
        largest_component = max(region_props, key=lambda prop: prop.area)
        print(f"obj = {obj}, n_connected = {len(region_props)}, largest_area = {largest_component.area}")
    
        # keep largest only
        if largest_component.area > 5000:
            extracted = (extracted_relabeled == largest_component.label)
        else:
            extracted = np.zeros_like(extracted_relabeled)
        
        # fill interior holes (using 2D slices)
        extracted_filled = np.stack(
            [ndimage.binary_fill_holes(extracted[z,:,:]) for z in range(extracted.shape[0])])
        
        # find the non-zero indices and increment these with the "min" positions of the bbox
        nonzero_indices_local = extracted_filled.nonzero()
        nonzero_indices_global = tuple(np.array(arr) + offset for arr, offset in zip(nonzero_indices_local, [zmin,ymin,xmin]))
    
        # Place object labels only where there are nonzero elements (to avoid bounding box overlaps)
        nuclei_curated_connected[nonzero_indices_global] = obj

    
    Nuclei_Masks_Binary_Filtered = np.asarray(nuclei_curated_connected > 0).astype(np.int32)
    
    Nuclei_Masks_Numbered = Nuclei_Masks_Binary_Filtered * Cell_Masks

    Masks = np.asarray([Cell_Masks,Nuclei_Masks_Numbered]).astype('i4')


    AICSImage(Masks[np.newaxis,:,:,:,:],
              channel_names=["Cells","Nuclei"],
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
