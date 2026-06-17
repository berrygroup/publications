import os
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from aicsimageio import AICSImage
from blimp.preprocessing.illumination_correction import IlluminationCorrection
from blimp.processing.quantify import quantify
from blimp.utils import concatenate_images
from skimage import measure
from skimage.morphology import binary_erosion, disk


def process_single_site(
    input_file,
    input_dir,
    label_image_input_dir,
    label_image_output_dir,
    probmap_dir,
    features_dir,
    illumcorr_file,
):

    # load intensity image
    intensity_image = AICSImage(input_dir / input_file)
    illumination_correction = IlluminationCorrection(from_file=illumcorr_file)
    intensity_image_corrected = illumination_correction.correct(intensity_image)

    def erode_labels(label_img, selem):
        """Erode each label independently so touching labels separate at their shared boundary."""
        out = np.zeros_like(label_img)
        for lbl in np.unique(label_img):
            if lbl == 0:
                continue
            out[binary_erosion(label_img == lbl, selem)] = lbl
        return out

    # load corresponding label image
    labels = AICSImage(
        label_image_input_dir / str(Path(input_file).name)
    )

    # load nucleoplasm probability image
    nucleoplasm_prob = AICSImage(
        str(probmap_dir) + "/" + Path(Path(input_file).stem).stem + "_Probabilities.tiff"
    )

    # check probability map is 8-bit as expected (threshold of 128 assumes uint8)
    if nucleoplasm_prob.dtype != np.uint8:
        raise ValueError(
            f"Expected uint8 probability map, got {nucleoplasm_prob.dtype}. "
            "Adjust the threshold accordingly."
        )

    # threshold nucleoplasm probability image to create a mask for the nucleoplasm
    nucleoplasm_mask = nucleoplasm_prob.get_image_data("YX") > 180

    # first channel of labels = nucleus label image (YX), preserving label IDs
    nucleus_labels = labels.get_image_data("YX", C=0)

    # erode the nucleus (per-label, so touching nuclei separate at the shared seam)
    eroded_nuclei = erode_labels(nucleus_labels, disk(2))

    # mask the eroded nucleus by the nucleoplasm mask
    nucleoplasm_labels = np.where(nucleoplasm_mask, eroded_nuclei, 0)

    # erode again to ensure no overlap with nucleolus
    nucleoplasm_labels = erode_labels(nucleoplasm_labels, disk(1))

    # convert to AICSImage
    all_labels = AICSImage(
        np.stack([nucleus_labels, eroded_nuclei, nucleoplasm_labels], axis=0).astype(np.int32),
        dim_order="CYX",
        channel_names=["Nucleus", "ErodedNucleus", "Nucleoplasm"],
        pixel_sizes=intensity_image.physical_pixel_sizes,
    )

    all_labels.save(
        str(label_image_output_dir) + "/labels_" + str(Path(input_file).name)
    )

    # quantify
    features = quantify(
        intensity_image=intensity_image_corrected,
        label_image=all_labels,
        parent_object=0,
        aggregate=True,
    )

    features.to_csv(features_dir / Path(Path(input_file).stem + ".csv"), index=False)

    return


def select_input_file(input_dir, index, extension="tiff"):
    input_files = glob(str(input_dir / ("*." + extension)))
    input_files.sort()
    print(input_files[index])
    return input_files[index]


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(prog="run_quantification_spots")

    parser.add_argument("--id", type=int, help="batch id", required=True)
    parser.add_argument("--input_dir")
    parser.add_argument("--label_image_input_dir")
    parser.add_argument("--label_image_output_dir")
    parser.add_argument("--probmap_dir")
    parser.add_argument("--features_dir")
    parser.add_argument(
        "--illumination_correction_file",
    )

    args = parser.parse_args()
    input_file = select_input_file(Path(args.input_dir), index=args.id)
    input_dir = Path(args.input_dir)
    label_image_input_dir = Path(args.label_image_input_dir)
    label_image_output_dir = Path(args.label_image_output_dir)
    probmap_dir = Path(args.probmap_dir)
    features_dir = Path(args.features_dir)
    illumcorr_file = Path(args.illumination_correction_file)

    if not features_dir.exists():
        features_dir.mkdir()
    if not illumcorr_file.exists():
        print("Illumination correction file does not exist")
        exit(-1)

    process_single_site(
        input_file,
        input_dir,
        label_image_input_dir,
        label_image_output_dir,
        probmap_dir,
        features_dir,
        illumcorr_file,
    )
