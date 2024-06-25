import os
from glob import glob
import numpy as np
from pathlib import Path
from aicsimageio import AICSImage
from blimp.processing.segment_and_quantify import segment_nuclei_cellpose, quantify
from blimp.preprocessing.illumination_correction import IlluminationCorrection


def process_single_site(input_file,input_dir,label_image_dir,features_dir,cellpose_model_file):

    # get intensity image and correct
    intensity_image = AICSImage(input_dir / input_file)
    illumination_correction = IlluminationCorrection(
        from_file=input_dir.parent.parent / "ILLUMCORR" / "illumination_correction.pkl"
    )
    intensity_image_corrected = illumination_correction.correct(intensity_image)

    # crop 500 pixels from left
    intensity_image_corrected = AICSImage(intensity_image_corrected.get_image_data('TCZYX')[:,:,:,:,500:],
                                          physical_pixel_sizes=intensity_image.physical_pixel_sizes,
                                          channel_names=intensity_image.channel_names)

    # segment all timepoints
    nuclei_label_image = segment_nuclei_cellpose(
        intensity_image_corrected,
        pretrained_model=cellpose_model_file,
    )

    # save label images
    nuclei_label_image.save(label_image_dir / Path("nuclei_" + Path(input_file).name))

    # quantify relative to these features
    features = quantify(intensity_image_corrected,nuclei_label_image)
    features.to_csv(features_dir / Path(Path(input_file).stem + ".csv"), index=False)

    return


def select_input_file(input_dir,index,extension="tiff"):
    input_files = glob(str(input_dir / ("*." + extension)))
    input_files.sort()
    print(input_files[index])
    return(input_files[index])


if __name__ == "__main__":
    from argparse import ArgumentParser
    
    parser = ArgumentParser(prog="run_quantification_202402_BleachChase")

    parser.add_argument(
        "--id",
        type=int,
        help="batch id",
        required=True
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        default="/srv/scratch/berrylab/z3532965/systems_Ti2/202402_BleachChase_mCherry-POLR2A/20240208_085709_941/OME-TIFF-MIP"
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default="/srv/scratch/berrylab/z3532965/systems_Ti2/202402_BleachChase_mCherry-POLR2A/20240208_085709_941/SEGMENTATION"
    )
    parser.add_argument(
        "-f",
        "--features_dir",
        default="/srv/scratch/berrylab/z3532965/systems_Ti2/202402_BleachChase_mCherry-POLR2A/20240208_085709_941/QUANTIFICATION"
    )
    parser.add_argument(
        "--cellpose_model_file",
        default="/srv/scratch/z3532965/src/blana/Scott/202402_BleachChase_mCherry-POLR2A/CELLPOSE/20X_mCherry-POLR2A_20240209"
    )

    args = parser.parse_args()
    input_file = select_input_file(Path(args.input_dir),index=args.id)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    features_dir = Path(args.features_dir)
    cellpose_model_file = Path(args.cellpose_model_file)

    if not output_dir.exists(): 
        output_dir.mkdir()
    if not features_dir.exists(): 
        features_dir.mkdir()
    if not cellpose_model_file.exists():
        print("Cellpose model file does not exist")
        exit(-1)

    process_single_site(input_file,input_dir,output_dir,features_dir,cellpose_model_file)
    