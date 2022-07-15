from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGroupBox, QGridLayout, QLabel, QSpinBox, QSlider

from projdata_visualisation.ProjDataVisualisationBackendTools.STIRInterface import ProjdataDims, \
    ProjDataVisualisationBackend


class UIGroupboxProjdataDimensions:
    """Used to control the sinogram dimension slider, scroll box and values."""

    def __init__(self, stir_interface: ProjDataVisualisationBackend) -> QGroupBox:

        self.stir_interface = stir_interface

        self.groupbox = QGroupBox("Projection Data Dimensions")

        default_UI_config_dict = {
            ProjdataDims.SEGMENT_NUM: {
                'label': 'Segment number',
                'value': 0,
                'connect_method': self.segment_number_refresh
            },
            ProjdataDims.AXIAL_POS: {
                'label': 'Axial position',
                'connect_method': self.axial_pos_refresh
            },
            ProjdataDims.VIEW_NUMBER: {
                'label': 'View number',
                'value': 0,
                'connect_method': self.view_num_refresh
            },
            ProjdataDims.TANGENTIAL_POS: {
                'label': 'Tangential position',
                'connect_method': self.tangential_pos_refresh
            }
        }

        self.UI_slider_spinboxes = self.construct_slider_spinboxes(slider_spinbox_configurations=default_UI_config_dict)

        ##### LAYOUT ####
        layout = QGridLayout()
        # layout.addWidget(lineEdit, 0, 0, 1, 2)
        self.UI_slider_spinboxes[ProjdataDims.SEGMENT_NUM].add_item_to_layout(layout, row=0)
        self.UI_slider_spinboxes[ProjdataDims.AXIAL_POS].add_item_to_layout(layout, row=2)
        self.UI_slider_spinboxes[ProjdataDims.VIEW_NUMBER].add_item_to_layout(layout, row=4)
        self.UI_slider_spinboxes[ProjdataDims.TANGENTIAL_POS].add_item_to_layout(layout, row=6)

        layout.setRowStretch(5, 1)
        self.groupbox.setLayout(layout)

    def set_UI_connect_methods(self, methods: (callable, list[callable])) -> None:
        """
        This function is called when the user changes the tangential position value.
        """
        if not isinstance(methods, list):
            methods = [methods]
        self.triggered_UI_changes = methods

    def UI_controller_UI_change_trigger(self):
        if len(self.triggered_UI_changes) == 0:
            print("triggered_UI_changes is empty")
        for method in self.triggered_UI_changes:
            method()

    def segment_number_refresh(self):
        """
        This function is called when the user changes the segment number value.
        """
        new_segment_num = self.UI_slider_spinboxes[ProjdataDims.SEGMENT_NUM].value()
        self.stir_interface.refresh_segment_data(new_segment_num)
        self.UI_controller_UI_change_trigger()

    def axial_pos_refresh(self):
        """
        This function is called when the user changes the axial position value.
        """
        self.UI_controller_UI_change_trigger()

    def view_num_refresh(self):
        """
        This function is called when the user changes the view number value.
        """
        self.UI_controller_UI_change_trigger()

    def tangential_pos_refresh(self):
        """
        This function is called when the user changes the tangential position value.
        """
        self.UI_controller_UI_change_trigger()

    def refresh_sliders_and_spinboxes_ranges(self) -> None:
        """
        Update the sliders and spinboxes ranges based upon the stir_interface projdata.
        """
        if self.stir_interface.proj_data_stream is None:
            return
        # Update all slider and spinbox ranges, should start with segment number
        for dimension in ProjdataDims:
            current_segment_num = self.stir_interface.get_current_segment_num()
            limits = self.stir_interface.get_limits(dimension, current_segment_num)
            self.UI_slider_spinboxes[dimension].update_limits(limits=limits)

    def update_enable_disable(self, sinogram_radio_button_state: bool):
        # Segment slider and scroll box handling
        segment_limits = self.stir_interface.get_limits(ProjdataDims.SEGMENT_NUM,
                                                        self.stir_interface.get_current_segment_num())
        if self.stir_interface.proj_data_stream is not None:
            if segment_limits[0] == 0 and segment_limits[1] == 0:
                self.disable(ProjdataDims.SEGMENT_NUM)
            else:
                self.enable(ProjdataDims.SEGMENT_NUM)

        # Check if sinogram or viewgram is selected and disable the appropriate sliders and spinboxs
        if sinogram_radio_button_state:
            # Disable the tangential position slider and spinbox
            self.disable(ProjdataDims.VIEW_NUMBER)
            axial_pos_limits = self.stir_interface.get_limits(ProjdataDims.AXIAL_POS,
                                                              self.stir_interface.get_current_segment_num())
            if (axial_pos_limits[0] - axial_pos_limits[1]) != 0:
                self.enable(ProjdataDims.AXIAL_POS)
            else:
                self.disable(ProjdataDims.AXIAL_POS)

        elif not sinogram_radio_button_state:
            self.disable(ProjdataDims.AXIAL_POS)
            self.enable(ProjdataDims.VIEW_NUMBER)

        if True:  # Until I work out what to do with tangential position
            self.disable(ProjdataDims.TANGENTIAL_POS)

    def construct_slider_spinboxes(self, slider_spinbox_configurations: dict) -> dict:
        """
        Constructs the UI for the slider and spinbox items based upon the configuration in my_dict.
        :param slider_spinbox_configurations: The configuration for the slider and spinbox items.
        :return: A dictionary of the slider and spinbox items.
        """
        UI_slider_spinboxes = {}
        for item in slider_spinbox_configurations.items():
            if self.stir_interface.proj_data_stream is None:
                max_range, min_range = 0, 0
            else:
                max_range, min_range = \
                    self.stir_interface.get_limits(item[0], self.stir_interface.get_current_segment_num())

            value = 0 if 'value' not in item[1] else item[1]['value']

            UI_slider_spinboxes[item[0]] = UISliderSpinboxItem(groupbox=self.groupbox,
                                                               label=f"{item[1]['label']}:",
                                                               min_range=min_range,
                                                               max_range=max_range,
                                                               value=value,
                                                               connect_method=item[1]['connect_method']
                                                               )
        return UI_slider_spinboxes

    def enable(self, dimension: ProjdataDims):
        self.UI_slider_spinboxes[dimension].enable()

    def disable(self, dimension: ProjdataDims):
        self.UI_slider_spinboxes[dimension].disable()

    def get_limits(self, dimension: ProjdataDims):
        return self.UI_slider_spinboxes[dimension].get_limits()

    def value(self, dimension: ProjdataDims):
        return self.UI_slider_spinboxes[dimension].value()


class UISliderSpinboxItem:
    """
    Class for the UI of the ProjDataVisualisationBackend.
    """

    def __init__(self, groupbox: QGroupBox,
                 label: str,
                 min_range: int,
                 max_range: int,
                 value: int,
                 connect_method: callable) -> None:
        """
        Constructor for the UISliderSpinboxItem class.
        :param groupbox: The groupbox to add the label, slider and spinbox to.
        :param label: The label for the item, generally a short string.
        :param min_range: The minimum value for the spinbox/slider.
        :param max_range: The maximum value for the spinbox/slider.
        :param value: The initial value for the spinbox/slider.
        :param connect_method: Called method when the spinbox or slider value is changed.
        """

        # Connect method. This is called after the spinbox or slider value is changed.
        self._connect_method = connect_method

        # Label
        self.label = QLabel(f"{label} {min_range, max_range}")

        # Spinbox
        self.spinbox = QSpinBox(groupbox)
        self.spinbox.setRange(min_range, max_range)
        self.spinbox.setValue(value)
        self.spinbox.valueChanged.connect(self.spinbox_connect)

        # Slider
        self.slider = QSlider(Qt.Orientation.Horizontal, groupbox)
        self.slider.setRange(min_range, max_range)
        self.slider.setValue(self.spinbox.value())
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.valueChanged.connect(self.slider_connect)

    # ----------- UI configuration methods -----------
    def add_item_to_layout(self, layout: QGridLayout, row: int) -> None:
        """
        Adds the label, spinbox and slider to the give layout at row.
        Assumed that the layout is a QGridLayout.
        """
        layout.addWidget(self.label, row, 0, 1, 1)
        layout.addWidget(self.slider, row + 1, 0, 1, 1)
        layout.addWidget(self.spinbox, row + 1, 1, 1, 1)

    def enable(self, enable=True) -> None:
        """
        Enables or disables the spinbox and slider.
        """
        self.label.setEnabled(enable)
        self.spinbox.setEnabled(enable)
        self.slider.setEnabled(enable)

    def disable(self, disable=True) -> None:
        """
        Disables or enables the spinbox and slider.
        """
        self.enable(not disable)

    def update_limits(self, limits: tuple, value=None) -> None:
        """
        Updates the range of the spinbox and slider and adjusts the value to be within range.
        """
        self.spinbox.setRange(limits[0], limits[1])
        self.slider.setRange(limits[0], limits[1])
        if value is None or value < limits[0] or value > limits[1]:
            value = min(limits[1], max(self.slider.value(), limits[0]))
        self.spinbox.setValue(value)
        self.slider.setValue(value)

    def get_limits(self) -> (int, int):
        """
        Returns the range (min, max) of the spinbox and slider.
        """
        return self.spinbox.minimum(), self.spinbox.maximum()

    def value(self) -> int:
        """
        Returns the value of the item.
        """
        return self.spinbox.value()

    # ----------- Connect methods -----------
    def spinbox_connect(self) -> None:
        """
        On spinbox connect, update the slider value and call _connect_method.
        """
        if self.spinbox.value() != self.slider.value():  # Prevents duplicate connect calls
            self.slider.setValue(self.spinbox.value())
            self.__connect()

    def slider_connect(self) -> None:
        """
        On slider connect, update the spinbox value and call _connect_method.
        """
        if self.slider.value() != self.spinbox.value():  # Prevents duplicate connect calls
            self.spinbox.setValue(self.slider.value())
            self.__connect()

    def __connect(self) -> None:
        """
        Connects the spinbox and slider to the connect method.
        Used as safety incase _connect_method is None.
        """
        if self._connect_method is not None:
            self._connect_method()
