import sys
import logging

from pathlib import Path
from glob import glob
from aicsimageio import AICSImage

from blimp.preprocessing.illumination_correction import IlluminationCorrection
from blimp.processing.segment import segment_nuclei_cellpose
from blimp.processing.quantify import quantify

def segment_and_quantify_nucleus(input_file,input_dir,output_dir,illumcorr_file,features_dir):

    Intensity_Image = AICSImage(input_dir / input_file)

    illumination_correction = IlluminationCorrection(from_file=illumcorr_file)
    intensity_image_corrected = illumination_correction.correct(Intensity_Image)

    nuclei = segment_nuclei_cellpose(intensity_image_corrected, nuclei_channel=5)
    nuclei.save(output_dir / Path(input_file).name)

    features = quantify(
        intensity_image=intensity_image_corrected,
        label_image=nuclei,
        texture_channels=5,
        texture_objects="Nuclei"
    )

    features[0].to_csv(features_dir / (Path(input_file).stem + ".csv"), index=False)

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
        default="/srv/scratch/berrylab/z3532965/systems_Ti2/20260626_POLR2A_heterogeneity/20260629_134305_894/OME-TIFF-MIP"
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        default="/srv/scratch/berrylab/z3532965/systems_Ti2/20260626_POLR2A_heterogeneity/20260629_134305_894/SEGMENTATION"
    )

    parser.add_argument(
        "-f",
        "--features_dir",
        default="/srv/scratch/berrylab/z3532965/systems_Ti2/20260626_POLR2A_heterogeneity/20260629_134305_894/QUANTIFICATION"
    )

    parser.add_argument(
        "--illumination_correction_file",
        default="/srv/scratch/berrylab/z3532965/systems_Ti2/20260626_POLR2A_heterogeneity/20260629_134305_894/illumination_correction.pkl"
    )
    parser.add_argument("-v", "--verbose", action="count", default=0,
        help="Increase verbosity (-v for INFO, -vv for DEBUG)")

    args = parser.parse_args()
    
    # Configure logging to stdout
    log_level = [logging.WARNING, logging.INFO, logging.DEBUG][min(args.verbose, 2)]
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )

    input_dir = Path(args.input_dir)
    input_file = select_input_file(input_dir,index=args.id)

    output_dir = Path(args.output_dir)
    features_dir = Path(args.features_dir)
    illumcorr_file = Path(args.illumination_correction_file)

    output_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)

    segment_and_quantify_nucleus(input_file,input_dir,output_dir,illumcorr_file,features_dir)
