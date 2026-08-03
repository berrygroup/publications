from pathlib import Path

from glob import glob

from typing import List, Tuple, Union, Dict

import numpy as np
import pandas as pd

from aicsimageio import AICSImage, readers


import bigfish.detection as detection
import bigfish.stack as stack

def spot_detection(input_file,input_dir,output_dir_565,output_dir_633,threshold_565,threshold_633):

    img = AICSImage(input_dir / input_file)

    spot_radius_px = detection.get_object_radius_pixel(
        voxel_size_nm=(
            img.physical_pixel_sizes.Z*1000,
            img.physical_pixel_sizes.Y*1000,
            img.physical_pixel_sizes.X*1000),
        object_radius_nm=(375, 175,175),ndim=3)

    FISH_633_Filtered = stack.log_filter(img.get_image_data('ZYX', C=0, T=0), spot_radius_px)
    FISH_565_Filtered = stack.log_filter(img.get_image_data('ZYX', C=2, T=0), spot_radius_px)

    FISH_633_Mask_Local_Max = detection.local_maximum_detection(FISH_633_Filtered,  min_distance=spot_radius_px)
    FISH_565_Mask_Local_Max = detection.local_maximum_detection(FISH_565_Filtered, min_distance=spot_radius_px)

    FISH633_spots, _ = detection.spots_thresholding(
        FISH_633_Filtered,FISH_633_Mask_Local_Max,
        threshold=threshold_633,
        remove_duplicate=True)

    FISH565_spots, _ = detection.spots_thresholding(
        FISH_565_Filtered,
        FISH_565_Mask_Local_Max,
        threshold=threshold_565,
        remove_duplicate=True)
    
    np.save(output_dir_633 / Path(Path(input_file).stem + "_FISH633_spots.npy"), FISH633_spots)

    np.save(output_dir_565 / Path(Path(input_file).stem + "_FISH565_spots.npy"), FISH565_spots)

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


    threshold_565 = 15
    threshold_633 = 10

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
        "-o1",
        "--output_dir_565",
        default="/srv/scratch/berrylab/z3536241/NikonSpinningDisk/250131_mACmChPOLR2A_smFISH/20250204_171458_755/SPOTS_565"
    )

    parser.add_argument(
        "-o2",
        "--output_dir_633",
        default="/srv/scratch/berrylab/z3536241/NikonSpinningDisk/250131_mACmChPOLR2A_smFISH/20250204_171458_755/SPOTS_633"
    )

    args = parser.parse_args()
    input_file = select_input_file(Path(args.input_dir),index=args.id)
    input_dir = Path(args.input_dir)
    output_dir_565 = Path(args.output_dir_565)
    output_dir_633 = Path(args.output_dir_633)

    spot_detection(input_file,input_dir,output_dir_565,output_dir_633,threshold_565,threshold_633)
