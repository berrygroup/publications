import os
from glob import glob
import numpy as np
import pandas as pd
from pathlib import Path
from aicsimageio import AICSImage

from blimp.processing.quantify import quantify
from blimp.preprocessing.illumination_correction import IlluminationCorrection

from skimage import measure

def process_single_site(input_file,input_dir,label_image_dir,features_dir,illumcorr_file):

    # load intensity image
    intensity_image = AICSImage(input_dir / input_file)
    illumination_correction = IlluminationCorrection(
        from_file=illumcorr_file
    )
    intensity_image_corrected = illumination_correction.correct(intensity_image)

    # load corresponding label image
    labels = AICSImage(label_image_dir / input_file)

    # quantify
    features = quantify(intensity_image_corrected, labels)

    features[0].to_csv(features_dir / Path(Path(input_file).stem + ".csv"), index=False)

    return


def select_input_file(input_dir,index,extension="tiff"):
    input_files = glob(str(input_dir / ("*." + extension)))
    input_files.sort()
    print(input_files[index])
    return(input_files[index])


if __name__ == "__main__":
    from argparse import ArgumentParser
    
    parser = ArgumentParser(prog="run_quantification_spots")

    parser.add_argument(
        "--id",
        type=int,
        help="batch id",
        required=True
    )
    parser.add_argument(
        "--input_dir"
    )
    parser.add_argument(
        "--label_image_dir"
    )
    parser.add_argument(
        "--features_dir"
    )
    parser.add_argument(
        "--illumination_correction_file",
    )
    
    args = parser.parse_args()
    input_file = select_input_file(Path(args.input_dir),index=args.id)
    input_dir = Path(args.input_dir)
    label_image_dir = Path(args.label_image_dir)
    features_dir = Path(args.features_dir)
    illumcorr_file = Path(args.illumination_correction_file)


    if not features_dir.exists(): 
        features_dir.mkdir()
    if not illumcorr_file.exists():
        print("Illumination correction file does not exist")
        exit(-1)

    process_single_site(input_file,input_dir,label_image_dir,features_dir,illumcorr_file)
    
