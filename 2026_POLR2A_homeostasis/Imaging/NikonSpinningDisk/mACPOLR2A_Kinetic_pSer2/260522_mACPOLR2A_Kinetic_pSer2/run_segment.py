from pathlib import Path

from glob import glob

from typing import List, Tuple, Union, Dict

import numpy as np

from aicsimageio import AICSImage, readers

from skimage import io, filters, exposure, feature, transform, util, morphology, segmentation, measure

from scipy import ndimage

from blimp.preprocessing.illumination_correction import IlluminationCorrection
from blimp.processing.segment import segment_nuclei_cellpose

def segment_nucleus(input_file,input_dir,output_dir,illumcorr_file):

    Intensity_Image = AICSImage(input_dir / input_file)

    illumination_correction = IlluminationCorrection(from_file=illumcorr_file)
    Intensity_Image_Corrected = illumination_correction.correct(Intensity_Image)

    nuclei = segment_nuclei_cellpose(Intensity_Image_Corrected, nuclei_channel=3)
    nuclei.save(output_dir / Path(input_file).name)

    return

def select_input_file(input_dir,index,extension="tiff"):
    input_files = glob(str(input_dir / ("*." + extension)))
    input_files.sort()
    print(input_files[index])
    return(input_files[index])


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(prog="run_segment")

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

    segment_nucleus(input_file,input_dir,output_dir,illumcorr_file)
