import os
from glob import glob
import numpy as np
from pathlib import Path
from aicsimageio import AICSImage
from blimp.processing.segment_and_quantify import segment_nuclei_cellpose, quantify
from blimp.preprocessing.illumination_correction import IlluminationCorrection


def process_single_site(input_file,input_dir,label_image_dir,features_dir,cellpose_model_file,illumcorr_file):

    # get intensity image and correct
    intensity_image = AICSImage(input_dir / input_file)
    illumination_correction = IlluminationCorrection(
        from_file=illumcorr_file
    )

    # acquisition 3 has an extra two channels, remove these here to fit the shape of the correction object
    if intensity_image.dims.C > 2:
        arr = np.stack([intensity_image.get_image_data('ZYX',C=i) for i in range(2)])
        im = AICSImage(
            arr[np.newaxis,:,:,:,:],
            physical_pixel_sizes=intensity_image.physical_pixel_sizes,
            channel_names=intensity_image.channel_names[:2])
        intensity_image_corrected = illumination_correction.correct(im)
    else:
        intensity_image_corrected = illumination_correction.correct(intensity_image)

    # segment DAPI channel (1)
    nuclei_label_image = segment_nuclei_cellpose(
        intensity_image_corrected,
        nuclei_channel=1,
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
        default="/srv/scratch/berrylab/z3532965/NikonSpinningDisk/240222/ILLUMCORR/illumination_correction.pkl"
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
    