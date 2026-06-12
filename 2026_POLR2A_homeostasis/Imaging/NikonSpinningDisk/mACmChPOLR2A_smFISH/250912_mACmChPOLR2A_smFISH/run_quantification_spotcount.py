import os
from glob import glob
import numpy as np
import pandas as pd
from pathlib import Path
from aicsimageio import AICSImage

from blimp.processing.segment import segment_nuclei_cellpose
from blimp.processing.quantify import quantify
from blimp.preprocessing.illumination_correction import IlluminationCorrection

from skimage import measure

def process_single_site(input_file,input_dir,label_image_dir,fish_spots_1_dir,fish_spots_2_dir,features_dir,illumcorr_file):

    # load intensity image
    intensity_image = AICSImage(input_dir / input_file)
    illumination_correction = IlluminationCorrection(
        from_file=illumcorr_file
    )
    intensity_image_corrected = illumination_correction.correct(intensity_image)

    # load corresponding label image
    labels = AICSImage(str(label_image_dir) + "/labels_" + str(Path(input_file).name))

    # load corresponding FISH spot localisations and intron segmentation
    spots1 = np.load(str(fish_spots_1_dir) + '/' + str(Path(input_file).stem) + "_FISH565_spots.npy")
    spots2 = np.load(str(fish_spots_2_dir) + '/' + str(Path(input_file).stem) + "_FISH633_spots.npy")

    # quantify
    features = quantify(intensity_image_corrected, labels)

    # count spots per cell and per nucleus
    cell_labels = labels.get_image_data('ZYX', C=0)
    nuc_labels = labels.get_image_data('ZYX', C=1)

    spots1_arr = np.zeros_like(cell_labels)
    spots1_arr[tuple(spots1.T)] = 1
    spots1_bycell = spots1_arr*cell_labels
    spots1_bynucleus = spots1_arr*nuc_labels
    spots_1_cell_counts = np.unique(spots1_bycell[spots1_bycell > 0], return_counts=True)
    spots_1_nuc_counts = np.unique(spots1_bynucleus[spots1_bynucleus > 0], return_counts=True)

    spots2_arr = np.zeros_like(cell_labels)
    spots2_arr[tuple(spots2.T)] = 1
    spots2_bycell = spots2_arr*cell_labels
    spots2_bynucleus = spots2_arr*nuc_labels
    spots_2_cell_counts = np.unique(spots2_bycell[spots2_bycell > 0], return_counts=True)
    spots_2_nuc_counts = np.unique(spots2_bynucleus[spots2_bynucleus > 0], return_counts=True)

    cell_counts = pd.DataFrame({"label" : np.unique(cell_labels[cell_labels>0])})

    spots_1_cell_counts_df = pd.DataFrame({
        "label" : spots_1_cell_counts[0], 
        "FISH_Channel1_SpotsPerCell" : spots_1_cell_counts[1]
    })
    spots_1_nuc_counts_df = pd.DataFrame({
        "label" : spots_1_nuc_counts[0], 
        "FISH_Channel1_SpotsPerNucleus" : spots_1_nuc_counts[1]
    })
    spots_2_cell_counts_df = pd.DataFrame({
        "label" : spots_2_cell_counts[0], 
        "FISH_Channel2_SpotsPerCell" : spots_2_cell_counts[1]
    })
    spots_2_nuc_counts_df = pd.DataFrame({
        "label" : spots_2_nuc_counts[0], 
        "FISH_Channel2_SpotsPerNucleus" : spots_2_nuc_counts[1]
    })

    cell_counts = cell_counts.merge(spots_2_cell_counts_df, how = 'left')
    cell_counts = cell_counts.merge(spots_2_nuc_counts_df, how = 'left')
    cell_counts = cell_counts.merge(spots_1_cell_counts_df, how = 'left')
    cell_counts = cell_counts.merge(spots_1_nuc_counts_df, how = 'left')
    cols_to_fill = ['FISH_Channel1_SpotsPerNucleus','FISH_Channel1_SpotsPerCell','FISH_Channel2_SpotsPerNucleus','FISH_Channel2_SpotsPerCell']
    cell_counts[cols_to_fill] = cell_counts[cols_to_fill].fillna(0).astype(int)

    features_cell = features[0].merge(cell_counts, how = 'left')
    features_nuc = features[1].merge(cell_counts, how = 'left')

    features_cell.to_csv(features_dir / Path("cells_" + Path(input_file).stem + ".csv"), index=False)
    features_nuc.to_csv(features_dir / Path("nuclei_" + Path(input_file).stem + ".csv"), index=False)

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
        "--fish_spots_1_dir"
    )
    parser.add_argument(
        "--fish_spots_2_dir"
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
    fish_spots_1_dir = Path(args.fish_spots_1_dir)
    fish_spots_2_dir = Path(args.fish_spots_2_dir)
    features_dir = Path(args.features_dir)
    illumcorr_file = Path(args.illumination_correction_file)


    if not features_dir.exists(): 
        features_dir.mkdir()
    if not illumcorr_file.exists():
        print("Illumination correction file does not exist")
        exit(-1)

    process_single_site(input_file,input_dir,label_image_dir,fish_spots_1_dir,fish_spots_2_dir,features_dir,illumcorr_file)
    
