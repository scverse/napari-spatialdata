![SpatialData banner](https://github.com/scverse/spatialdata/blob/main/docs/_static/img/spatialdata_horizontal.png?raw=true)

# napari-spatialdata: interactive exploration and annotation of spatial omics data

[![License](https://img.shields.io/pypi/l/napari-spatialdata.svg?color=green)](https://github.com/scverse/napari-spatialdata/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-spatialdata.svg?color=green)](https://pypi.org/project/napari-spatialdata)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-spatialdata.svg?color=green)](https://python.org)
[![tests](https://github.com/scverse/napari-spatialdata/workflows/tests/badge.svg)](https://github.com/scverse/napari-spatialdata/actions)
[![codecov](https://codecov.io/gh/scverse/napari-spatialdata/branch/main/graph/badge.svg?token=ASqlOKnOj7)](https://codecov.io/gh/scverse/napari-spatialdata)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/scverse/napari-spatialdata/main.svg)](https://results.pre-commit.ci/latest/github/scverse/napari-spatialdata/main)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-spatialdata)](https://napari-hub.org/plugins/napari-spatialdata)
[![DOI](https://zenodo.org/badge/477021400.svg)](https://zenodo.org/badge/latestdoi/477021400)
[![Documentation][badge-pypi]][link-pypi]
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/napari-spatialdata/badges/version.svg)](https://anaconda.org/conda-forge/napari-spatialdata)

[badge-pypi]: https://badge.fury.io/py/napari-spatialdata.svg
[link-pypi]: https://pypi.org/project/napari-spatialdata/

This repository contains a napari plugin for interactively exploring and annotating
SpatialData objects. Here you can find the [napari-spatialdata documentation](https://spatialdata.scverse.org/projects/napari/en/stable/notebooks/spatialdata.html). `napari-spatialdata` is part of the `SpatialData` ecosystem. To learn more about SpatialData, please see the [spatialdata documentation](https://spatialdata.scverse.org/).

## Installation

You can install `napari-spatialdata` via [pip]:

    pip install napari-spatialdata[all]

The `all` command will install the qt bindings `PyQt5`.

Napari now also includes multiple triangulation backends. These improve the speed by which a napari 'Shapes' layer gets
loaded (this does not improve the speed of editing large numbers of shapes yet!). See also the napari
[documentation](https://napari.org/stable/guides/triangulation.html) and already slightly older [blog post](https://napari.org/island-dispatch/blog/triangles_speedup_beta.html). For installation via
pip:

    pip install napari-spatialdata[all, bermuda]

You can find more details on this in the [installation instructions](https://spatialdata.scverse.org/en/stable/installation.html).

## Using napari-spatialdata as default zarr reader

If you would like to use the plugin as the default zarr reader, in napari please go to `File` -> `Preferences`
-> `Plugins` and follow the instructions under `File extension readers`.

## Development Version

You can install `napari-spatialdata` from Github with:

    pip install git+https://github.com/scverse/napari-spatialdata

Or, you can also install in editable mode after cloning the repo by:

    git clone https://github.com/scverse/napari-spatialdata
    cd napari-spatialdata
    pip install -e .

Note: when performing an editable install of `napari-spatialdata`, `spatialdata`
will be reinstalled from `pip`. So, if you previously also made an editable install
of `spatialdata`, you need to re-run `pip install -e .` on the `spatialdata`
repository. Please find more details on this in the [installation instructions](https://spatialdata.scverse.org/en/stable/installation.html).

## Getting started

To learn how to use the `napari-spatialdata` plugin, please see the [documentation](https://spatialdata.scverse.org/projects/napari/en/stable/notebooks/spatialdata.html).
To learn how to integrate napari-spatialdata into your analysis workflows, please
see the [SpatialData tutorials](https://spatialdata.scverse.org/en/stable/tutorials/notebooks/notebooks.html). In particular:

- [Annotating regions of interest with napari](https://spatialdata.scverse.org/en/stable/tutorials/notebooks/notebooks/examples/napari_rois.html)
- [Use landmark annotations to align multiple -omics layers](https://spatialdata.scverse.org/en/stable/tutorials/notebooks/notebooks/examples/alignment_using_landmarks.html)

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"napari-spatialdata" is free and open source software

## Issues

If you encounter any problems, please file an [issue] along with a detailed description.

## Citation

Marconato, L., Palla, G., Yamauchi, K.A. et al. SpatialData: an open and universal data framework for spatial omics. Nat Methods (2024). https://doi.org/10.1038/s41592-024-02212-x

[napari]: https://github.com/napari/napari
[cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[mit]: http://opensource.org/licenses/MIT
[bsd-3]: http://opensource.org/licenses/BSD-3-Clause
[gnu gpl v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[gnu lgpl v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[apache software license 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[mozilla public license 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[pypi]: https://pypi.org/
[issue]: https://github.com/scverse/napari-spatialdata/issues
[//]: # "numfocus-fiscal-sponsor-attribution"

napari-spatialdata is part of the scverse® project ([website](https://scverse.org), [governance](https://scverse.org/about/roles)) and is fiscally sponsored by [NumFOCUS](https://numfocus.org/).
If you like scverse® and want to support our mission, please consider making a tax-deductible [donation](https://numfocus.org/donate-to-scverse) to help the project pay for developer time, professional services, travel, workshops, and a variety of other needs.

<div align="center">
<a href="https://numfocus.org/project/scverse">
  <img
    src="https://raw.githubusercontent.com/numfocus/templates/master/images/numfocus-logo.png"
    width="200"
  >
</a>
</div>
