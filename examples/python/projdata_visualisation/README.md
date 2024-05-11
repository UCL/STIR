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
### Command Line Interface Usage:

```
python ProjDataVisualisation.py <ProjData filename>
```
where `<ProjData filename>` is an optional filename CLI argument. Alternatively, this filename can be set in the GUI.

### Demonstration usage:

For usage using an existing python session, i.e. with projection data already loaded, see the example `demo_ProjDataVisualisation.py`.

