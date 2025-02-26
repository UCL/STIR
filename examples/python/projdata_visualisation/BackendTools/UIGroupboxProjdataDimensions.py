# Copyright 2022, 2024 University College London

# Author Robert Twyman

# This file is part of STIR.
#
# SPDX-License-Identifier: Apache-2.0
#
# See STIR/LICENSE.txt for details

from BackendTools.STIRInterface import ProjDataDims, ProjDataVisualisationBackend
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGroupBox, QGridLayout, QLabel, QSpinBox, QDoubleSpinBox, QSlider
import sys
import stir

# from https://stackoverflow.com/a/61439160 author: Siyh
# modification by KT to set interval in setRange
class DoubleSlider(QSlider):
    """
    slider that allows doubles as min/max/value

    WARNING: values will be discretised according to interval
    """

    def __init__(self, *args, **kargs):
        super(DoubleSlider, self).__init__( *args, **kargs)
        self._min = 0
        self._max = 99
        self.interval = 1

    def setValue(self, value):
        index = round((value - self._min) / self.interval)
        return super(DoubleSlider, self).setValue(index)

    def value(self):
        return self.index * self.interval + self._min

    @property
    def index(self):
        return super(DoubleSlider, self).value()

    def setIndex(self, index):
        return super(DoubleSlider, self).setValue(index)

    def setMinimum(self, value):
        self._min = value
        self.setInterval()

    def setMaximum(self, value):
        self._max = value
        self.setInterval()

    def setRange(self, min, max):
        self._min = min
        self._max = max
        self.setInterval()

    def setInterval(self, value = None):
        if value is None:
            value = (self._max - self._min) / 500
        self.interval = 1 if value == 0 else value
        self._range_adjusted()

    def _range_adjusted(self):
        number_of_steps = int((self._max - self._min) / self.interval)
        # doesn't make sense to have too many steps
        number_of_steps = max(min(number_of_steps, 500), 1)
        super(DoubleSlider, self).setMaximum(number_of_steps)
        self.interval = (self._max - self._min) / number_of_steps

class UIGroupboxProjDataDimensions:
    """Used to control the sinogram dimension slider, scroll box and values."""

    def __init__(self, stir_interface: ProjDataVisualisationBackend) -> QGroupBox:

        self.__external_UI_methods_on_connect = None
        self.stir_interface = stir_interface

        self.groupbox = QGroupBox("ProjData Dimensions")

        default_UI_config_dict = {
            ProjDataDims.SEGMENT_NUM: {
                'label': 'Segment number',
                'value': 0,
                'connect_method': self.segment_number_refresh,
                'int_values': True
            },
            ProjDataDims.AXIAL_POS: {
                'label': 'Axial position',
                'connect_method': self.axial_pos_refresh,
                'int_values': True
            },
            ProjDataDims.VIEW_NUMBER: {
                'label': 'View number',
                'value': 0,
                'connect_method': self.view_num_refresh,
                'int_values': True
            },
            ProjDataDims.TANGENTIAL_POS: {
                'label': 'Tangential position',
                'connect_method': self.tangential_pos_refresh,
                'int_values': True
            },
            ProjDataDims.TIMING_POS: {
                'label': 'TOF bin',
                'connect_method': self.timing_pos_refresh,
                'int_values': True
            },
            ProjDataDims.VMAX: {
                'label': 'vmax',
                'value': sys.float_info.max,
                'connect_method': self.vmax_refresh,
                'int_values': False
            }
        }

        self.UI_slider_spinboxes = self.__construct_slider_spinboxes(slider_spinbox_configurations=default_UI_config_dict)

        ##### LAYOUT ####
        layout = QGridLayout()
        # layout.addWidget(lineEdit, 0, 0, 1, 2)
        self.UI_slider_spinboxes[ProjDataDims.SEGMENT_NUM].add_item_to_layout(layout, row=0)
        self.UI_slider_spinboxes[ProjDataDims.AXIAL_POS].add_item_to_layout(layout, row=2)
        self.UI_slider_spinboxes[ProjDataDims.VIEW_NUMBER].add_item_to_layout(layout, row=4)
        self.UI_slider_spinboxes[ProjDataDims.TANGENTIAL_POS].add_item_to_layout(layout, row=6)
        self.UI_slider_spinboxes[ProjDataDims.TIMING_POS].add_item_to_layout(layout, row=8)
        self.UI_slider_spinboxes[ProjDataDims.VMAX].add_item_to_layout(layout, row=10)

        layout.setRowStretch(5, 1)
        self.groupbox.setLayout(layout)

    @staticmethod
    def _standardise_connect_methods(methods: list or callable) -> list:
        """
        Sets the external connect methods for the UI.
        Pass methods in a list of callable objects. These methods will be called when the UI is changed in order.
        """
        if not isinstance(methods, list):
            methods = [methods]

        for m in methods:
            if not callable(m):
                raise TypeError(f"{m} is not callable.")

        return methods

    def set_UI_configure_connect_methods(self, methods: list or callable) -> None:
        self.__external_UI_configure_methods_on_connect = self._standardise_connect_methods(methods)

    def set_UI_connect_methods(self, methods: list or callable) -> None:
        self.__external_UI_methods_on_connect = self._standardise_connect_methods(methods)

    def UI_controller_UI_configure_change_trigger(self):
        if self.__external_UI_configure_methods_on_connect is None:
            return
        for method in self.__external_UI_configure_methods_on_connect:
            method()

    def UI_controller_UI_change_trigger(self):
        if self.__external_UI_methods_on_connect is None:
            return
        for method in self.__external_UI_methods_on_connect:
            method()

    # Refresh methods on spinbox/slider changes
    def segment_number_refresh(self):
        """ This function is called when the user changes the segment number value.
        Because of the way the STIR segment data is handled, the segment_data needs to change first."""
        self.stir_interface.segment_data = self.stir_interface.projdata.get_segment_by_view(
            self.get_segment_indices_from_UI())
        self.UI_controller_UI_configure_change_trigger()

    def axial_pos_refresh(self):
        """This function is called when the user changes the axial position value."""
        self.UI_controller_UI_change_trigger()

    def view_num_refresh(self):
        """This function is called when the user changes the view number value."""
        self.UI_controller_UI_change_trigger()

    def tangential_pos_refresh(self):
        """This function is called when the user changes the tangential position value."""
        self.UI_controller_UI_change_trigger()

    def timing_pos_refresh(self):
        """This function is called when the user changes the TOF bin value."""
        self.stir_interface.segment_data = self.stir_interface.projdata.get_segment_by_view(
            self.get_segment_indices_from_UI())
        self.UI_controller_UI_change_trigger()

    def vmax_refresh(self):
        """This function is called when the user changes the vmax value."""
        self.UI_controller_UI_change_trigger()

    def get_segment_indices_from_UI(self) -> stir.SegmentIndices:
        """Returns the segment indices from the UI slider and spinboxes."""
        return stir.SegmentIndices(self.UI_slider_spinboxes[ProjDataDims.SEGMENT_NUM].value(),
                                   self.UI_slider_spinboxes[ProjDataDims.TIMING_POS].value())

    def refresh_sliders_and_spinboxes_ranges(self) -> None:
        """Update the sliders and spinboxes ranges based upon the stir_interface projdata."""
        if self.stir_interface.projdata is None:
            return
        # Update all slider and spinbox ranges, should start with segment number
        for dimension in ProjDataDims:
            limits = self.stir_interface.get_limits(dimension, self.stir_interface.get_current_segment_num())
            self.UI_slider_spinboxes[dimension].update_limits(limits=limits)

    def configure_enable_disable_sliders(self, is_sinogram_mode: bool):
        """Configure the sliders and spinboxes based upon the current mode and the projdata limits in each dimension."""

        self.disable(ProjDataDims.TANGENTIAL_POS)

        segment_limits = self.stir_interface.get_limits(ProjDataDims.SEGMENT_NUM,
                                                        self.stir_interface.get_current_segment_num())
        self.update_dimension_state(ProjDataDims.SEGMENT_NUM, segment_limits)

        if is_sinogram_mode:
            # No view number in sinogram mode
            self.disable(ProjDataDims.VIEW_NUMBER)
            axial_limits = self.stir_interface.get_limits(ProjDataDims.AXIAL_POS,
                                                          self.stir_interface.get_current_segment_num())
            self.update_dimension_state(ProjDataDims.AXIAL_POS, axial_limits)
        else:
            # No axial position in viewgram (not sinogram) mode
            self.disable(ProjDataDims.AXIAL_POS)
            view_limits = self.stir_interface.get_limits(ProjDataDims.VIEW_NUMBER,
                                                         self.stir_interface.get_current_segment_num())
            self.update_dimension_state(ProjDataDims.VIEW_NUMBER, view_limits)

        tof_limits = self.stir_interface.get_limits(ProjDataDims.TIMING_POS,
                                                    self.stir_interface.get_current_segment_num())
        self.update_dimension_state(ProjDataDims.TIMING_POS, tof_limits)

    def __construct_slider_spinboxes(self, slider_spinbox_configurations: dict) -> dict:
        """
        Constructs the UI for the slider and spinbox items based upon the configuration in my_dict.
        :param slider_spinbox_configurations: The configuration for the slider and spinbox items.
        :return: A dictionary of the slider and spinbox items.
        """
        UI_slider_spinboxes = {}
        for item in slider_spinbox_configurations.items():
            value = 0 if 'value' not in item[1] else item[1]['value']
            if self.stir_interface.projdata is None:
                max_range, min_range = value, 0
            else:
                min_range, max_range = \
                    self.stir_interface.get_limits(item[0], self.stir_interface.get_current_segment_num())

            UI_slider_spinboxes[item[0]] = UISliderSpinboxItem(groupbox=self.groupbox,
                                                               label=f"{item[1]['label']}:",
                                                               lower_limit=min_range,
                                                               upper_limit=max_range,
                                                               value=value,
                                                               connect_method=item[1]['connect_method'],
                                                               int_values=item[1]['int_values']
                                                               )
        return UI_slider_spinboxes

    def update_dimension_state(self, dimension, limits):
        """Updates the state of the slider and spinbox for the given dimension. If limits are equal, disable."""
        if limits[1] == limits[0]:
            self.disable(dimension)
        else:
            self.enable(dimension)

    def enable(self, dimension: ProjDataDims) -> None:
        """Enables the slider and spinbox for the given dimension."""
        self.UI_slider_spinboxes[dimension].enable()

    def disable(self, dimension: ProjDataDims) -> None:
        """Disables the slider and spinbox for the given dimension."""
        self.UI_slider_spinboxes[dimension].disable()

    def get_limits(self, dimension: ProjDataDims) -> tuple:
        """Returns the limits of the slider and spinbox for the given dimension."""
        return self.UI_slider_spinboxes[dimension].get_limits()

    def value(self, dimension: ProjDataDims) -> int|float:
        """Returns the value of the slider and spinbox for the given dimension."""
        return self.UI_slider_spinboxes[dimension].value()


class UISliderSpinboxItem:
    """Class for the UI of the ProjDataVisualisationBackend."""
    def __init__(self, groupbox: QGroupBox,
                 label: str,
                 lower_limit: int | float,
                 upper_limit: int | float,
                 value: int | float,
                 connect_method: callable,
                 int_values = True) -> None:
        """
        Constructor for the UISliderSpinboxItem class.
        Creates a slider and spinbox item, with a label, between limits.
        The slider and spinbox are connected to the connect_method.
        :param groupbox: The groupbox to add the label, slider and spinbox to.
        :param label: The label for the item, generally a short string.
        :param lower_limit: The minimum value for the spinbox/slider.
        :param upper_limit: The maximum value for the spinbox/slider.
        :param value: The initial value for the spinbox/slider.
        :param connect_method: Called method when the spinbox or slider value is changed.
        """

        self.__int_values = int_values

        # Connect method. This is called after the spinbox or slider value is changed.
        self._connect_method = connect_method

        # Label
        self.__label_str = label
        self.__label = QLabel(self.create_label_str((lower_limit, upper_limit)))

        # Spinbox
        if int_values:
            self.__spinbox = QSpinBox(groupbox)
        else:
            self.__spinbox = QDoubleSpinBox(groupbox)

        # Slider
        try: # Qt.Orientation.Horizontal required more recent versions of PyQt
            orientation = Qt.Orientation.Horizontal
        except AttributeError: # Qt.Horizontal required for PyQt version <= v5.10.1
            orientation = Qt.Horizontal
        if int_values:
            self.__slider = QSlider(orientation, groupbox)
        else:
            self.__slider = DoubleSlider(orientation, groupbox)
        self.__slider.setTickPosition(QSlider.TicksBelow)
        self.setRange(lower_limit, upper_limit)
        self.__spinbox.setValue(value)
        self.__slider.setValue(value)
        # enable "connectors" (do this after setValue() as otherwise that triggers the connectors again)
        self.__spinbox.valueChanged.connect(self.__spinbox_connect)
        self.__slider.valueChanged.connect(self.__slider_connect)

    def setValue(self, value: int|float):
        if value != self.value():
            self.__spinbox.setValue(value)
            # Possibly no need to update slider ourselves: "__spinbox_connect" should take care of it.
            # But better safe...
            self.__slider.setValue(value)

    def setRange(self, vmin: int|float, vmax: int|float) -> None:
        self.setMinimum(vmin)
        self.setMaximum(vmax)

    def setMinimum(self, value: int|float) -> None:
        self.__slider.setMinimum(value)
        self.__spinbox.setMinimum(value)

    def setMaximum(self, value: int|float) -> None:
        self.__slider.setMaximum(value)
        self.__spinbox.setMaximum(value)

    # ----------- UI configuration methods -----------
    def add_item_to_layout(self, layout: QGridLayout, row: int) -> None:
        """Adds the label, spinbox and slider to the give layout at row. Assumed that the layout is a QGridLayout."""
        layout.addWidget(self.__label, row, 0, 1, 1)
        layout.addWidget(self.__slider, row + 1, 0, 1, 1)
        layout.addWidget(self.__spinbox, row + 1, 1, 1, 1)

    def enable(self, enable=True) -> None:
        """Enables or disables the spinbox and slider."""
        self.__label.setEnabled(enable)
        self.__spinbox.setEnabled(enable)
        self.__slider.setEnabled(enable)

    def disable(self, disable=True) -> None:
        """Disables or enables the spinbox and slider."""
        self.enable(not disable)

    def update_limits(self, limits: tuple, value=None) -> None:
        """
        Updates the range of the spinbox and slider and adjusts the value to be within range.
        Also updates the label.
        :param limits: The new range (lower, upper) for the spinbox and slider.
        :param value: The new value for the spinbox. If None, the value is not changed unless out of limits.
        """
        self.setRange(limits[0], limits[1])
        self.__label.setText(self.create_label_str(limits))
        if value is None or value < limits[0] or value > limits[1]:
            value = min(limits[1], max(self.value(), limits[0]))
        self.setValue(value)

    def get_limits(self) -> tuple:
        """Returns the range (min, max) of the spinbox and slider."""
        return self.__spinbox.minimum(), self.__spinbox.maximum()

    def value(self) -> int | float:
        """Returns the value of the spinbox. Expected to be equal to slider value."""
        return self.__spinbox.value()

    def create_label_str(self, limits: tuple) -> str:
        """Returns the label string."""
        return f"{self.__label_str} {limits}"

    # ----------- Connect methods -----------
    # Calling slider.setValue() (or changing the slider) will call slider_connect, which calls
    # spinbox.setValue(), which will spinbox_connect. We therefore need to prevent that spinbox_connect
    # calls slider.setValue() again!
    def __spinbox_connect(self) -> None:
        """On spinbox connect, update the slider value and call __connect()."""
        if self.__spinbox.value() != self.__slider.value():  # Prevents duplicate connect calls
            self.__slider.setValue(self.__spinbox.value())
            self.__connect()

    def __slider_connect(self) -> None:
        """On slider connect, update the spinbox value and call __connect()."""
        if self.__slider.value() != self.__spinbox.value():  # Prevents duplicate connect calls
            self.__spinbox.setValue(self.__slider.value())
            self.__connect()

    def __connect(self) -> None:
        """Connects the spinbox and slider to the connect method. Used as safety incase _connect_method is None."""
        if self._connect_method is not None:
            self._connect_method()
