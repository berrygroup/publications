import os
from glob import glob
import numpy as np
from pathlib import Path
from aicsimageio import AICSImage
from blimp.processing.segment import segment_nuclei_cellpose
from blimp.processing.quantify import quantify
from blimp.preprocessing.illumination_correction import IlluminationCorrection


def process_single_site(input_file,input_dir,label_dir,features_dir,illumcorr_file):

    # get intensity image and correct
    intensity_image = AICSImage(input_dir / input_file)
    illumination_correction = IlluminationCorrection(
        from_file=illumcorr_file
    )
    intensity_image_corrected = illumination_correction.correct(intensity_image)

    # use exisiting nuclei segmentation

    label_files = glob(str(Path(label_dir) / ("*." + "tiff")))
    input_file_p = Path(input_file)
    label = [x for x in label_files if input_file_p.stem[:7] in Path(x).stem and input_file_p.stem[30:34] in Path(x).stem][0]
    nuclei_label_image = AICSImage(label)

    # quantify relative to these features
    features = quantify(intensity_image_corrected,nuclei_label_image)
    features[0].to_csv(features_dir / Path(Path(input_file).stem + ".csv"), index=False)

    return


def select_input_file(input_dir,index,extension="tiff"):
    input_files = glob(str(input_dir / ("*." + extension)))
    input_files.sort()
    print(input_files[index])
    return(input_files[index])


if __name__ == "__main__":
    from argparse import ArgumentParser
    
    parser = ArgumentParser(prog="run_quantification_202411_dTAG-DIS3")

    parser.add_argument(
        "--id",
        type=int,
        help="batch id",
        required=True
    )
    parser.add_argument(
        "-i",
        "--input_dir"
    )
    parser.add_argument(
        "-l",
        "--label_dir"
    )
    parser.add_argument(
        "-f",
        "--features_dir"
    )
    parser.add_argument(
        "--illumination_correction_file",
        default="/srv/scratch/berrylab/z3532965/systems_Ti2/20241118_dTAG-DIS3/ILLUMCORR/202411_20X.pkl"
    )


    args = parser.parse_args()
    input_file = select_input_file(Path(args.input_dir),index=args.id)
    input_dir = Path(args.input_dir)
    label_dir = Path(args.label_dir)
    features_dir = Path(args.features_dir)
    illumcorr_file = Path(args.illumination_correction_file)


    if not features_dir.exists(): 
        features_dir.mkdir()
    if not illumcorr_file.exists():
        print("Illumination correction file does not exist")
        exit(-1)

    process_single_site(input_file,input_dir,label_dir,features_dir,illumcorr_file)
    
