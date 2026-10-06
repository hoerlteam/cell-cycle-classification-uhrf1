from nd2 import ND2File
from pathlib import Path
from tifffile import imwrite
from calmutils.imageio.tiff_imagej import save_tiff_imagej


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


def xarray_to_tiffs(
    img,
    out_directory,
    out_prefix,
    split_dimensions=("T", "P", "C"),
    prefixes={"C": "_ch", "T": "_tp", "P": "_pos", "Z": "_z", "X": "_x", "Y": "_y"},
    use_indices=True,
    min_index_len=1,
    pixel_size = None,
):


    # if no pixel sizes are given, guess from xarray coordinate ticks (difference between adjacent)
    if pixel_size is None:
        psz_x = (img.coords['X'].values[1:] - img.coords['X'].values[:-1]).mean() if 'X' in img.coords else 1
        psz_y = (img.coords['Y'].values[1:] - img.coords['Y'].values[:-1]).mean() if 'Y' in img.coords else 1
        psz_z = (img.coords['Z'].values[1:] - img.coords['Z'].values[:-1]).mean() if 'Z' in img.coords else 1
        pixel_size = [psz_z, psz_y, psz_x]

    # find which of the selected split dimensions are present
    present_split_dimensions = [d for d in split_dimensions if d in img.dims]

    # handle no splitting
    if len(present_split_dimensions) == 0:

        # construct filename, dimension names
        out_filename = out_prefix + '.tif'
        out_filename = Path(out_directory) / out_filename
        axes = "".join([d for d in img.dims])

        save_tiff_imagej(
            out_filename,
            img.values.squeeze(),
            axes=axes,
            distance_unit="micron",
            pixel_size=pixel_size
        )

        return

    # group by split dimensions
    for idx, sub_img in img.groupby(present_split_dimensions):

        # treat even single split dimension as list of one
        if len(present_split_dimensions) == 1:
            idx = [idx]

        # get integer indices if desired
        if use_indices:
            filename_idx = [img.get_index(d).get_loc(i) for d,i in zip(present_split_dimensions, idx)]
            filename_idx = [str(i).rjust(min_index_len, '0') for i in filename_idx]
        else:
            filename_idx = idx

        # construct out filename
        out_filename = out_prefix + "".join(prefixes[d] + i for d, i in zip(present_split_dimensions, filename_idx)) + '.tif'
        out_filename = Path(out_directory) / out_filename

        # save as tiff
        axes = "".join([d for d in img.dims if d not in split_dimensions])
        save_tiff_imagej(out_filename, sub_img.values.squeeze(), axes=axes, distance_unit="micron", pixel_size=pixel_size)
