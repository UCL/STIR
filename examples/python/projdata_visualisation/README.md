# README ProjDataVisualisation.py

Author: Robert Twyman, 2022

## Brief Description
This python script may be used to visualize STIR projection data (ProjData) in an interactive GUI. This python script can be used to load STIR interfiles and the GUI contains sliders that allows the user to scroll through the following ProjData parameters:
- Segment number
- View number
- Axial position
- Tangential position (_not yet implemented_)

The visualization window will be automatically updated. Additionally, the ProjData may be visualized as Sinogram or Viewgram.


## Requirements
- Python 3
- STIR installed with python support (`${STIR_INSTALL_DIR}/python` in `$PYTHONPATH`)
- Matplotlib
- PyQT5

## Usage
```
python $STIR/examples/python/projdata_visualisation/ProjDataVisualisation.py
```
where `` is an optional argument of the name of a STIR projection data file. This filename can be set in the GUI.