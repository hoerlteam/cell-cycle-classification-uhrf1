from nd2 import ND2File
from pathlib import Path
from calmutils.imageio import xarray_to_tiffs


def resave_nd2_flexible(
    in_path,
    file_pattern="*.nd2",
    out_path=None,
    split_dimensions=("C", "T", "P"),
    prefixes={"C": "_ch", "T": "_tp", "P": "_pos", "Z": "_z", "X": "_x", "Y": "_y"},
    use_indices=True,
    min_index_len=1
):
    # by default, save in "tif" subfolder
    if out_path is None:
        out_path = Path(in_path) / 'tif'
    # make out_path a pathlib path in any case
    out_path = Path(out_path)

    # create if it does not yet exist
    if not out_path.exists():
        out_path.mkdir()

    for in_file in Path(in_path).glob(file_pattern):
        resave_nd2_flexible_single(
            in_file,
            out_path,
            split_dimensions,
            prefixes,
            use_indices,
            min_index_len
        )


def resave_nd2_flexible_single(
    in_file,
    out_path,
    split_dimensions=("C", "T", "P"),
    prefixes={"C": "_ch", "T": "_tp", "P": "_pos", "Z": "_z", "X": "_x", "Y": "_y"},
    use_indices=True,
    min_index_len=1
):

    # read file to XArray
    with ND2File(in_file) as reader:
        img = reader.to_xarray()
        pixel_size = list(reader.voxel_size())[::-1]

    xarray_to_tiffs(
        img,
        out_path,
        in_file.stem,
        split_dimensions,
        prefixes,
        use_indices,
        min_index_len,
        pixel_size
    )
