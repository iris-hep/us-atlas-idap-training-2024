# [Coffea columnar analysis framework](https://indico.cern.ch/event/1376945/contributions/5787150/)

This module is designed to be run on a US ATLAS Coffea-Casa instance, like the [IRIS-HEP SSL deployment at the University of Chicago Analysis Facility](https://coffea.af.uchicago.edu/).

## Opening notebooks with `jupytext`

To open the notebook `.py` files as `.ipynb` files in Jupyter Lab, right click on the `.py` file and select "Open With -> Notebook".
`jupytext` will sync the state of your `.py` file to a `.ipynb` file and vise versa.

## Local environment with `pixi`

If you would like to run parts of the module notebooks locally in a standalone environment, you can use the provided [`pixi`](https://pixi.sh/) environment.
To do so:
* [Install `pixi`](https://pixi.sh/latest/#installation)
* Run `pixi run example`
