"""

=======
trailpy
=======

Visualize trail GPX data

"""

import argparse
import gpxpy
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy.ma as ma
import numpy as np
import pandas as pd

import os
import glob
import copy

from bmi_topography import Topography
import rioxarray as rxr
from osgeo import gdal
from matplotlib.collections import LineCollection
from scipy.interpolate import RegularGridInterpolator
import osmnx as ox
from rasterio.errors import RasterioIOError

import subprocess

gdal.UseExceptions()

ARG_PARSER = argparse.ArgumentParser(description="Visualize trail GPX data")

ARG_PARSER.add_argument("gpx_files", type=str, help="File path to GPX file")

ARG_PARSER.add_argument(
    "--trail_scale_fraction",
    "-s",
    type=float,
    help="Fraction of length trail occupies",
    default=0.6,
)

ARG_PARSER.add_argument(
    "--dem_type",
    "-d",
    type=str,
    help="Source os DEM",
    default="USGS10m",
    choices=[
        "SRTMGL3",
        "SRTMGL1",
        "SRTMGL1_E",
        "AW3D30",
        "AW3D30_E",
        "SRTM15Plus",
        "NASADEM",
        "COP30",
        "COP90",
        "EU_DT",
        "GEDI_L3",
        "GEBCOIceTopo",
        "GEBCOSubIceTopo",
        "CA_MRDEM_DSM",
        "CA_MRDEM_DTM",
        "USGS30m",
        "USGS10m",
        "USGS1m",
    ],
)

ARG_PARSER.add_argument(
    "--show_peaks",
    "-p",
    help="Show peaks",
    action="store_true",
)

ARG_PARSER.add_argument(
    "--show_water",
    "-w",
    help="Show water",
    action="store_true",
)

ARG_PARSER.add_argument(
    "--skip_plot",
    help="Ship showing plot",
    action="store_true",
)

ARG_PARSER.add_argument(
    "--cmap",
    "-c",
    type=str,
    help=(
        "Colormap for elevation channel. "
        "See: https://matplotlib.org/stable/users/explain/colors/colormaps.html"
    ),
    default="gray",
)

ARG_PARSER.add_argument(
    "--alpha_sf",
    "-A",
    type=float,
    help="Scale factor for alpha channel.",
    default=1,
)

ARG_PARSER.add_argument(
    "--color_sf",
    "-C",
    type=float,
    help="Scale factor for color channel.",
    default=1,
)


def load_gpxs(gpx_files):
    """
    Load any number of GPX file data

    Parameters
    ----------
    gpx_files : list of str
        File path to GPX files

    Returns
    -------
    tracks : list
        Numby array of shape (ntracks, nsegments, 3)
    extents : dict
        Dictionary with 'north', 'east', 'south', and 'west' keys to define the extents
        of the tracks
    """

    tracks = []
    for gpx_file in gpx_files:
        _, new_tracks, _ = load_gpx(gpx_file)
        tracks.extend(new_tracks)

    combined = np.concatenate(tracks)

    # Get the extreme values for the extents
    extents = {
        "south": combined[..., 1].min(),
        "north": combined[..., 1].max(),
        "west": combined[..., 0].min(),
        "east": combined[..., 0].max(),
    }

    return tracks, extents


def load_gpx(gpx_file):
    """
    Load GPX file data

    Parameters
    ----------
    gpx_file : str
        File path to GPX file

    Returns
    -------
    tracks : np.array
        Numby array of shape (ntracks, nsegments, 3)
    extents : dict
        Dictionary with 'north', 'east', 'south', and 'west' keys to define the extents
        of the tracks
    """

    track_arrays = []
    with open(gpx_file, "r") as gpx_contents:
        gpx = gpxpy.parse(gpx_contents)

    for track in gpx.tracks:
        for segment in track.segments:
            seg_points = np.array(
                [
                    [point.longitude, point.latitude, point.elevation]
                    for point in segment.points
                ]
            )
            if len(seg_points) > 0:
                track_arrays.append(seg_points)

    # Stack all the tracks together
    tracks = track_arrays
    combined = np.concatenate(track_arrays)

    # Get the extreme values for the extents
    extents = {
        "south": combined[..., 1].min(),
        "north": combined[..., 1].max(),
        "west": combined[..., 0].min(),
        "east": combined[..., 0].max(),
    }

    return gpx, tracks, extents


def extend_extents(extents_orig, scale=1, dim="deg"):
    """
    Docstring for extend_extents

    Params
    ------
    extents_orig : dict
        Dictionary with 'north', 'east', 'south', and 'west' keys to define the extents
        of the tracks
    scale : float
        Scale factor for the longest dimension. Defaults to 1
    dim : str
        The dimension to perform the scaling in, either "km" or "deg". Defaults to "deg"

    Returns
    -------
    extents : dict
        Scaled extents with the same keys
    """

    lons = np.array([extents_orig["west"], extents_orig["east"]])
    lats = np.array([extents_orig["south"], extents_orig["north"]])

    center_lon = lons.mean()
    center_lat = lats.mean()

    if dim.lower() == "km":
        # Do the scaling in km
        km_per_deg = 2 * np.pi * 6371 / 360
        km_per_deg_eastwest = km_per_deg * np.cos(np.radians(center_lat))
    elif dim.lower() == "deg":
        # Do the scaling in deg
        km_per_deg = 1
        km_per_deg_eastwest = 1

    # Get the maximum dimension
    height = np.diff(lats)[0] * km_per_deg
    width = np.diff(lons)[0] * km_per_deg_eastwest
    max_dim = max([height, width]) * scale

    # Convert back to original units
    half_height = max_dim / 2 / km_per_deg
    half_width = max_dim / 2 / km_per_deg_eastwest

    extents = {
        "south": center_lat - half_height,
        "north": center_lat + half_height,
        "west": center_lon - half_width,
        "east": center_lon + half_height,
    }

    return extents


def retrieve_topo(extents, output_dir, dem_type="SRTMGL1", catch_dem_err=True):
    """
    Retrieve topodata

    Parameters
    ----------
    extents : dict
        Dictionary with 'north', 'east', 'south', and 'west' keys to define the extents
        of the area to pull
    output_dir : str
        Output directory name
    dem_type : str, optional
        DEM type. Defaults to SRTMGL1
    catch_dem_err : bool, optional
        Flag to catch DEM unavailable error and witch to another DEM source. Defaults to
        True

    Returns
    -------
    topo_data : xarray.core.dataarray.DataArray
        Topographic data
    filename : str
        File where topo data is saved at

    """

    params = Topography.DEFAULT.copy()
    params.update(extents)
    params.update({"cache_dir": output_dir})
    params.update({"dem_type": dem_type})

    location = Topography(**params)
    try:
        location.fetch()
        topo_data = location.load()
    except RasterioIOError as err:
        if catch_dem_err:
            dem_type = "SRTMGL1"
            params.update({"dem_type": dem_type})
            location = Topography(**params)
            location.fetch()
            topo_data = location.load()
        else:
            raise RasterioIOError(err.__str__())

    filename = os.path.join(
        output_dir,
        "_".join(
            [
                dem_type,
                f"{extents['south']}",
                f"{extents['west']}",
                f"{extents['north']}",
                f"{extents['east']}",
            ]
        )
        + ".tif",
    )

    return topo_data, filename


def hillshade(in_file, out_file):
    """
    Perform hillshade using gdal

    Parameters
    ----------
    in_file : str
        Input file
    out_file : str
        Output file

    Returns
    -------
    raster_data : xarray.core.dataarray.DataArray
        Hill shaded data
    """

    in_data = gdal.Open(in_file)

    # Create hillshade
    options = ["-combined"]
    hill_file = out_file.replace(".tif", "_itermediate.tif")
    gdal.DEMProcessing(hill_file, in_data, "hillshade", options=options)

    # Correct for gamma
    command = [
        "gdal_calc",
        "-A",
        hill_file,
        f"--outfile={out_file}",
        '--calc="uint8(((A / 255.)**(1/0.5)) * 255)"',
    ]

    subprocess.run(command, check=True)

    # Load back data
    raster_data = rxr.open_rasterio(out_file, masked=False)
    return raster_data


def colored_line_plot(ax, x, y, c_values, cmap="viridis", norm=None, linewidth=2):
    """
    Plots a line with color varying based on c_values.

    Parameters
    ----------
    x : array-like
        X coordinates.
    y :
        Y coordinates.
    c_values : array-like, optional
        Values used for coloring (same size as x and y)
    cmap : str or Colormap, optional
        Colormap to use. Defaults to 'viridis'
    norm : Normalize, optional
        Color normalization instance. If None, linear scaling is used.
    linewith : float, optional
        Width of line
    """
    # 1. Create a list of (x, y) points
    points = np.array([x, y]).T.reshape(-1, 1, 2)

    # 2. Create segments from points
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # 3. Create a LineCollection object
    # The 'c' argument is used for coloring based on a colormap (cmap)
    lc = LineCollection(segments, cmap=plt.get_cmap(cmap), norm=norm)

    # 4. Set the values that map to the color scale
    lc.set_array(c_values)

    # 5. Set line properties (optional)
    lc.set_linewidth(linewidth)

    # 6. Get the current axis and add the collection to the plot
    line = ax.add_collection(lc)

    return line


def osm_locations(extents):
    """
    Get locations of interest from Open Street Maps

    Parameters
    ----------
    extents : dict
        Dictionary with 'north', 'east', 'south', and 'west' keys to define the extents

    Returns
    -------
    peaks : GeoPandas.DataFrame
        DataFrame containing peak information
    water : GeoPandas.DataFrame
        DataFrame containing water information
    """

    # Define the tags for peaks
    tags = {"natural": "peak"}

    # Retrieve POIs using the bounding box
    try:
        peaks = ox.features.features_from_bbox(
            (extents["west"], extents["south"], extents["east"], extents["north"]), tags
        )
        if "name" not in peaks.columns:
            peaks["name"] = None
        peaks = peaks.reset_index().dropna(subset="name").reset_index()
        peaks["lon"] = [geom.x for geom in peaks["geometry"].tolist()]
        peaks["lat"] = [geom.y for geom in peaks["geometry"].tolist()]
        peaks["marker"] = "^"
        peaks["label"] = "peak"
        peaks["color"] = "C2"
        peaks["markersize"] = 10
        peaks["markeredgecolor"] = "white"
    except ox._errors.InsufficientResponseError:
        peaks = pd.DataFrame()

    # Define the tags for peaks
    tags = {"natural": "water"}

    # Retrieve POIs using the bounding box
    try:
        water = ox.features.features_from_bbox(
            (extents["west"], extents["south"], extents["east"], extents["north"]), tags
        )
        if "name" not in water.columns:
            water["name"] = None
        water = water.reset_index().dropna(subset="name").reset_index()
        # water["centroid"] = water.geometry.centroid
        # water["lon"] = [geom.x for geom in water.geometry.centroid]
        # water["lat"] = [geom.y for geom in water.geometry.centroid]
        water["marker"] = "o"
        water["label"] = "water"
        water["color"] = "C0"
    except ox._errors.InsufficientResponseError:
        water = pd.DataFrame()

    return peaks, water


def combine_data_arrays_to_rgba(
    color_data, alpha_data, color_sf=1.0, alpha_sf=1.0, cmap="gray"
):
    """
    Combine two arrays to determine the color based on one array and the alpha based
    on the othere

    Parameters
    ----------
    color_data : np.array
        Array of values to determine the color of the image
    alpha_data : np.array
        Array of values to determine the color of the image
    color_sf : float, optional
        Scale factor for color. The maximum color value is
        color_sf * (color_data.max() - color_data.min()) + color_data.min()
        Defaults to 1.0
    alpha_sf : float, optional
        Scale factor for alpha. The maximum alpha value is
        alpha_sf * (alpha_data.max() - alpha_data.min()) + alpha_data.min()
        Defaults to 1.0
    cmap : str, optional
        The colormap to use. Defaults to 'gray'
    """

    # Define the norm for color
    vmin_color = 0 * (color_data.max() - color_data.min()) + color_data.min()
    vmax_color = color_sf * (color_data.max() - color_data.min()) + color_data.min()
    norm_color = mpl.colors.Normalize(vmin=vmin_color, vmax=vmax_color)

    # Define the norm for alpha
    vmin_alpha = 0 * (alpha_data.max() - alpha_data.min()) + alpha_data.min()
    vmax_alpha = alpha_sf * (alpha_data.max() - alpha_data.min()) + alpha_data.min()
    # norm_alpha = mpl.colors.Normalize(vmin=vmin_alpha, vmax=vmax_alpha)

    # Compute color for each pixel based on elevation
    cmap_color = plt.get_cmap(cmap)
    color_array = cmap_color(norm_color(color_data))

    # # Compute alpha for each pixel
    dynamic_range = vmax_alpha - vmin_alpha
    if dynamic_range == 0:
        alpha_array = np.ones(alpha_data.shape)
    else:
        alpha_array = np.clip((alpha_data - vmin_alpha) / dynamic_range, 0, 1)

    # Keep the color from the color array and the alpha from the alpha array
    image_array = copy.copy(color_array)
    image_array *= np.repeat(alpha_array[..., np.newaxis], 4, axis=-1)
    image_array[..., -1] = 1

    return image_array


def main(
    gpx_files,
    trail_scale_fraction=1,
    dem_type=None,
    show_peaks=True,
    show_water=True,
    cmap="gray",
    color_sf=1.0,
    alpha_sf=1.0,
):
    """
    Create visualization of Trail

    Parameters
    ----------
    gpx_file : str
        File path to GPX file
    trail_scale_fraction : float
        Fraction of length of graphic that trail occupies
    dem_type : str
        DEM source.
    show_peaks : bool, optional
        Flag to show peaks on map. Defaults to True
    show_water : bool, optional
        Flag to show water on map. Defaults to True
    cmap : str, optional
        The colormap to use. Defaults to 'gray'
    color_sf : float, optional
        Scale factor for color. The maximum color value is
        color_sf * (color_data.max() - color_data.min()) + color_data.min()
        Defaults to 1.0
    alpha_sf : float, optional
        Scale factor for alpha. The maximum alpha value is
        alpha_sf * (alpha_data.max() - alpha_data.min()) + alpha_data.min()
        Defaults to 1.0
    """
    if len(gpx_files) == 1:
        name = os.path.basename(gpx_files[0]).replace(".gpx", "")
    else:
        name = os.path.basename(os.path.dirname(gpx_files[0]))

    # Load GPX data
    tracks, extents = load_gpxs(gpx_files)

    # Extend extents
    extents = extend_extents(extents, scale=1 / trail_scale_fraction)

    # Request topography data
    output_dir = os.path.dirname(gpx_files[0])
    topo_data, filename_topo = retrieve_topo(extents, output_dir, dem_type=dem_type)

    # Perform hillshade
    filename_shaded = filename_topo.replace(".tif", "_hillshade.tif")
    hill_data = hillshade(filename_topo, filename_shaded)

    # Find interesing markers
    peak_df, water_df = osm_locations(extents)

    fig, ax = plt.subplots(figsize=(12, 12))

    # Combine the slope for tha alpha channel and the color for the elevation
    slope = hill_data.values.squeeze().astype(np.float64)
    elev = topo_data.values.squeeze().astype(np.float64)
    no_data = elev == -999999
    elev[no_data] = elev[~no_data].min()
    image_array = combine_data_arrays_to_rgba(
        elev, slope, cmap=cmap, color_sf=color_sf, alpha_sf=alpha_sf
    )

    # # Mask over invalid elevation data
    # if elev.min() == -999999:
    #     image_array[elev <= 0, -1] = 0

    # Plot
    x = hill_data.x.values
    y = hill_data.y.values

    dx = (x[1] - x[0]) / 2.0
    dy = (y[1] - y[0]) / 2.0
    imshow_extent = [x[0] - dx, x[-1] + dx, y[0] - dy, y[-1] + dy]
    img = ax.imshow(image_array, extent=imshow_extent, origin="lower")

    # Crop image
    center_x = np.mean(imshow_extent[0:2])
    center_y = np.mean(imshow_extent[2:])
    radius = (imshow_extent[1] - imshow_extent[0]) / 2
    patch = patches.Circle(
        (center_x, center_y),
        radius=radius,
        transform=ax.transData,
    )
    img.set_clip_path(patch)

    tracks_combined = np.concatenate(tracks)
    alt_min = tracks_combined[:, 2].min()
    alt_max = tracks_combined[:, 2].max()
    # norm_size = mpl.colors.Normalize(vmin=alt_min, vmax=alt_max)
    for track in tracks:
        x = track[:, 0]
        y = track[:, 1]
        z = track[:, 2]
        # track_alt = interpolator(np.stack([x, y]).T)
        # line = colored_line_plot(ax, x, y, z, cmap="plasma", norm=None, linewidth=6)
        scatter = ax.scatter(
            x,
            y,
            c=z,
            marker="o",
            cmap="plasma",
            vmin=alt_min,
            vmax=alt_max,
            s=10,
            # edgecolors="black",
            # linewidth=0.1,
        )
    ax.set_aspect("equal")
    # ax.set_title(name)
    # cbar = fig.colorbar(scatter)
    # cbar.set_label("Altitude [m]")

    # Plot points of interest
    if show_water:
        for _, water in water_df.iterrows():
            x, y = water["geometry"].exterior.xy
            xc = water["geometry"].centroid.x
            yc = water["geometry"].centroid.y
            ax.fill(x, y, color=water["color"], alpha=0.5)
            # ax.text(xc, yc, water["name"], color="white", va="center", ha="center")
    if show_peaks:
        for _, peak in peak_df.iterrows():
            ax.plot(
                peak["lon"],
                peak["lat"],
                marker=peak["marker"],
                color=peak["color"],
                markersize=peak["markersize"],
                markeredgecolor=peak["markeredgecolor"],
            )
            ax.text(
                peak["lon"],
                peak["lat"],
                peak["name"],
                color=peak["color"],
                va="top",
                ha="center",
            )
    ax.set_xlim([extents["west"], extents["east"]])
    ax.set_ylim([extents["south"], extents["north"]])
    ax.set_axis_off()
    ax.set_facecolor("none")
    fig.set_facecolor("none")
    fig.tight_layout()

    output_filename = os.path.join(os.path.dirname(gpx_files[0]), f"{name}.png")
    fig.savefig(output_filename)

    return fig


if __name__ == "__main__":
    ARGS = ARG_PARSER.parse_args()
    main(
        glob.glob(ARGS.gpx_files),
        ARGS.trail_scale_fraction,
        ARGS.dem_type,
        show_peaks=ARGS.show_peaks,
        show_water=ARGS.show_water,
        cmap=ARGS.cmap,
        color_sf=ARGS.color_sf,
        alpha_sf=ARGS.alpha_sf,
    )
    if not ARGS.skip_plot:
        plt.show()
