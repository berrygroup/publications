import os
from glob import glob
import numpy as np
from pathlib import Path
from aicsimageio import AICSImage
from blimp.processing.segment_and_quantify import segment_nuclei_cellpose, quantify, segment_secondary
from blimp.preprocessing.illumination_correction import IlluminationCorrection


def process_single_site(input_file,input_dir,label_image_dir,features_dir,cellpose_model_file,illumcorr_file):

    # get intensity image and correct
    intensity_image = AICSImage(input_dir / input_file)
    illumination_correction = IlluminationCorrection(
        from_file=illumcorr_file
    )
    intensity_image_corrected = illumination_correction.correct(intensity_image)

    # segment DAPI channel (channel 1)
    nuclei_label_image = segment_nuclei_cellpose(
        intensity_image_corrected,
        nuclei_channel=1,
        pretrained_model=cellpose_model_file,
    )

    # save label images
    nuclei_label_image.save(label_image_dir / Path("nuclei_" + Path(input_file).name))

    # quantify relative to these features
    features_nuclei = quantify(intensity_image_corrected,nuclei_label_image)

    # segment cells using poly(A) (channel 0)
    cell_label_array = segment_secondary(
        primary_label_image=nuclei_label_image.get_image_data('YX',C=0,Z=0,T=0),
        intensity_image=intensity_image.get_image_data('YX',C=0,Z=0,T=0),
        contrast_threshold=2,
        min_threshold=160)

    cell_label_image = AICSImage(
        cell_label_array[np.newaxis, np.newaxis, np.newaxis, :, :],
        channel_names=["Cell"],
        physical_pixel_sizes=intensity_image.physical_pixel_sizes,
    )

    # save label images
    cell_label_image.save(label_image_dir / Path("cell_" + Path(input_file).name))

    # quantify relative to these features
    features_cell = quantify(intensity_image_corrected,cell_label_image)

    # merge and dave features
    features = features_nuclei.merge(features_cell,on=['label','TimepointID'])
    features.to_csv(features_dir / Path(Path(input_file).stem + ".csv"), index=False)

    return

    return


def select_input_file(input_dir,index,extension="tiff"):
    input_files = glob(str(input_dir / ("*." + extension)))
    input_files.sort()
    print(input_files[index])
    return(input_files[index])


if __name__ == "__main__":
    from argparse import ArgumentParser
    
    parser = ArgumentParser(prog="run_quantification_202402_ARMC5_Fixed")

    parser.add_argument(
        "--id",
        type=int,
        help="batch id",
        required=True
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        default="/srv/scratch/berrylab/z3532965/NikonSpinningDisk/240222/240219_HEK293_ARMC5_IF/OME-TIFF-MIP"
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default="/srv/scratch/berrylab/z3532965/NikonSpinningDisk/240222/240219_HEK293_ARMC5_IF/SEGMENTATION"
    )
    parser.add_argument(
        "-f",
        "--features_dir",
        default="/srv/scratch/berrylab/z3532965/NikonSpinningDisk/240222/240219_HEK293_ARMC5_IF/QUANTIFICATION"
    )
    parser.add_argument(
        "--cellpose_model_file",
        default="/srv/scratch/z3532965/src/blana/Alex/202402_ARMC5_Fixed/CELLPOSE/DAPI_HEK_20240313_140337"
    )
    parser.add_argument(
        "--illumination_correction_file",
        default="/srv/scratch/berrylab/z3532965/NikonSpinningDisk/240315/240314_ARMC5KO_PolyA/ILLUMCORR/illumination_correction.pkl"
    )

    args = parser.parse_args()
    input_file = select_input_file(Path(args.input_dir),index=args.id)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    features_dir = Path(args.features_dir)
    cellpose_model_file = Path(args.cellpose_model_file)
    illumcorr_file = Path(args.illumination_correction_file)

    if not output_dir.exists(): 
        output_dir.mkdir()
    if not features_dir.exists(): 
        features_dir.mkdir()
    if not cellpose_model_file.exists():
        print("Cellpose model file does not exist")
        exit(-1)
    if not illumcorr_file.exists():
        print("Illumination correction file does not exist")
        exit(-1)

    process_single_site(input_file,input_dir,output_dir,features_dir,cellpose_model_file,illumcorr_file)
    