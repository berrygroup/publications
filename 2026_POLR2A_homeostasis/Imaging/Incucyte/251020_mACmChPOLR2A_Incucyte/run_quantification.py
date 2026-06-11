import os
from glob import glob
import numpy as np
from pathlib import Path
from aicsimageio import AICSImage

from cellpose import models

from blimp.processing.quantify import quantify


def process_single_site(input_file,input_dir,label_image_dir,features_dir):

    # get intensity image and correct
    intensity_image = AICSImage(input_dir / input_file)
    intensity_image = AICSImage(intensity_image.get_image_data('TCZYX'))

    intensity_image_data = intensity_image.get_image_data('YX')

    # import cellpose model
    model = models.CellposeModel(model_type='livecell_cp3')

    # segment live cell image
    live_cell_label_image_data, _, _, = model.eval(intensity_image_data, channels=[0, 0], do_3D=False, diameter = 40)

    live_cell_label_image_data = live_cell_label_image_data.astype('i4')

    live_cell_label_image = AICSImage(live_cell_label_image_data[np.newaxis,np.newaxis,np.newaxis,:,:])

    # save label images
    live_cell_label_image.save(label_image_dir / Path("cells_" + Path(input_file).name))

    # quantify relative to these features
    features = quantify(intensity_image,live_cell_label_image)
    features[0].to_csv(features_dir / Path(Path(input_file).stem + ".csv"), index=False)

    return


def select_input_file(input_dir,index,extension="tif"):
    input_files = glob(str(input_dir / ("*." + extension)))
    input_files.sort()
    print(input_files[index])
    return(input_files[index])


if __name__ == "__main__":
    from argparse import ArgumentParser
    
    parser = ArgumentParser(prog="run_quantification_202408_dTAG-DIS3")

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
        "-o",
        "--output_dir"
    )
    parser.add_argument(
        "-f",
        "--features_dir"
    )


    args = parser.parse_args()
    input_file = select_input_file(Path(args.input_dir),index=args.id)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    features_dir = Path(args.features_dir)

    if not output_dir.exists(): 
        output_dir.mkdir()
    if not features_dir.exists(): 
        features_dir.mkdir()

    process_single_site(input_file,input_dir,output_dir,features_dir)
    
