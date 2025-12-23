"""

=======
trailpy
=======

Visualize trail GPX data

"""

import argparse
import gpxpy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os
import pprint
import subprocess

from bmi_topography import Topography
import rioxarray as rxr
from osgeo import gdal
from matplotlib.collections import LineCollection
from scipy.interpolate import RegularGridInterpolator
import osmnx as ox


ARG_PARSER = argparse.ArgumentParser(description="Visualize trail GPX data")

ARG_PARSER.add_argument("gpx_file", type=str, help="File path to GPX file")

ARG_PARSER.add_argument(
    "--trail_scale_fraction",
    "-s",
    type=float,
    help="Fraction of length trail occupies",
    default=0.8,
)

ARG_PARSER.add_argument(
    "--dem_type",
    "-d",
    type=str,
    help="Source os DEM",
    default="SRTMGL3",
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
            track_arrays.append(seg_points)

    # Stack all the tracks together
    tracks = np.stack(track_arrays)

    # Get the extreme values for the extents
    extents = {
        "south": tracks[..., 1].min(),
        "north": tracks[..., 1].max(),
        "west": tracks[..., 0].min(),
        "east": tracks[..., 0].max(),
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


def retrieve_topo(extents, output_dir, dem_type="SRTMGL1"):
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
    location.fetch()
    topo_data = location.load()

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
    options = ["-combined"]
    gdal.DEMProcessing(out_file, in_data, "hillshade", options=options)

    # Load back data
    raster_data = rxr.open_rasterio(out_file, masked=False)
    return raster_data


def colored_line_plot(ax, x, y, c_values, cmap="viridis", norm=None):
    """
    Plots a line with color varying based on c_values.

    Parameters
    ----------
    x : array-like
        X coordinates.
    y :
        Y coordinates.
    c_values : array-like, optional
        Values used for coloring (same size as x and y). Defaults to "viridis"
    cmap : str or Colormap
        Colormap to use.
    norm : Normalize
        Color normalization instance. If None, linear scaling is used.
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
    lc.set_linewidth(2)

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
    peaks = ox.features.features_from_bbox(
        (extents["west"], extents["south"], extents["east"], extents["north"]), tags
    )
    peaks = peaks.reset_index().dropna(subset="name").reset_index()
    peaks["lon"] = [geom.x for geom in peaks["geometry"].tolist()]
    peaks["lat"] = [geom.y for geom in peaks["geometry"].tolist()]
    peaks["marker"] = "^"
    peaks["label"] = "peak"
    peaks["color"] = "C2"
    peaks["markersize"] = 10
    peaks["markeredgecolor"] = "white"

    # Define the tags for peaks
    tags = {"natural": "water"}

    # Retrieve POIs using the bounding box
    water = ox.features.features_from_bbox(
        (extents["west"], extents["south"], extents["east"], extents["north"]), tags
    )
    water = water.reset_index().dropna(subset="name").reset_index()
    # water["centroid"] = water.geometry.centroid
    water["lon"] = [geom.x for geom in water.geometry.centroid]
    water["lat"] = [geom.y for geom in water.geometry.centroid]
    water["marker"] = "o"
    water["label"] = "water"
    water["color"] = "C0"

    return peaks, water


def main(gpx_file, trail_scale_fraction, dem_type):
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
    """

    # Load GPX data
    gpx, tracks, extents = load_gpx(gpx_file)

    # Extend extents
    extents = extend_extents(extents, scale=1 / trail_scale_fraction)

    # Request topography data
    output_dir = os.path.dirname(gpx_file)
    topo_data, filename_topo = retrieve_topo(extents, output_dir, dem_type=dem_type)

    # Perform hillshade
    filename_shaded = filename_topo.replace(".tif", "_hillshade.tif")
    hill_data = hillshade(filename_topo, filename_shaded)

    # Set maximum slope to nan
    slope = hill_data.values.squeeze().astype(np.float64)
    lakes = slope == 255
    slope[lakes] = np.nan
    lakes = lakes.astype(np.float64)
    lakes[lakes == False] = np.nan

    # Find interesing markers
    peak_df, water_df = osm_locations(extents)

    fig, ax = plt.subplots(figsize=(12, 12))
    ctour = ax.contourf(
        hill_data.x.values,
        hill_data.y.values,
        slope,
        cmap="Greys",
        levels=30,
    )
    # ax.contourf(
    #     hill_data.x.values,
    #     hill_data.y.values,
    #     lakes,
    #     cmap="Blues",
    #     levels=0,
    # )

    interpolator = RegularGridInterpolator(
        (topo_data.x.values, topo_data.y.values), topo_data.values.squeeze()
    )
    for track in tracks:
        # ax.plot(track[:, 0], track[:, 1], color="black", linewidth=5)
        x = track[:, 0]
        y = track[:, 1]
        track_alt = interpolator(np.stack([x, y]).T)
        line = colored_line_plot(
            ax,
            x,
            y,
            track_alt,
            cmap="plasma",
            norm=None,
        )
    ax.set_aspect("equal")
    ax.set_title(os.path.basename(gpx_file))
    cbar = fig.colorbar(line)
    cbar.set_label("Altitude [m]")

    # Plot points of interest
    for _, water in water_df.iterrows():
        x, y = water["geometry"].exterior.xy
        xc = water["geometry"].centroid.x
        yc = water["geometry"].centroid.y
        ax.fill(x, y, color=water["color"])
        # ax.text(xc, yc, water["name"], color="white", va="center", ha="center")
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

    # topo_data.plot()
    # hill_data.plot()
    # breakpoint()
    fig.tight_layout()


if __name__ == "__main__":
    ARGS = ARG_PARSER.parse_args()
    main(ARGS.gpx_file, ARGS.trail_scale_fraction, ARGS.dem_type)
    plt.show()
