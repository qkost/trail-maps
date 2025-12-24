"""

=======
run_all
=======

Create visualizations of all GPX files.

"""

import os
from trailpy import ARG_PARSER, main

ROOT_DIR = "data"


def run_all(**kwargs):

    kwargs.pop("gpx_files")
    kwargs.pop("skip_plot")

    for root, directories, files in os.walk(ROOT_DIR):
        path = root.split(os.sep)
        gpx_files = [os.path.join(root, f) for f in files if f.endswith(".gpx")]

        if any(gpx_files):
            filename = os.path.join(ROOT_DIR, path[-1] + ".png")
            print(f"{root} --> {filename}")
            fig = main(gpx_files, **kwargs)
            fig.savefig(filename)


if __name__ == "__main__":
    ARGS = ARG_PARSER.parse_args()
    run_all(**vars(ARGS))
