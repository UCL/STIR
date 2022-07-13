from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QSpinBox, QSlider, QGroupBox, QGridLayout

from projdata_visualisation.ProjDataVisualisationBackend import ProjDataVisualisationBackend


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
    def add_to_layout(self, layout: QGridLayout, row: int) -> None:
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


def construct_slider_spinboxes(stir_interface: ProjDataVisualisationBackend,
                               UI_groupbox: QGroupBox,
                               configuration: dict) -> dict:
    """
    Constructs the UI for the slider and spinbox items based upon the configuration in my_dict.
    :param stir_interface: The stir interface  to get the values from.
    :param UI_groupbox: The groupbox to add the slider and spinbox to.
    :param configuration: The configuration for the slider and spinbox items.
    :return: A dictionary of the slider and spinbox items.
    """
    output_dict = {}
    for item in configuration.items():
        if stir_interface.proj_data_stream is None:
            max_range, min_range = 0, 0
        else:
            max_range, min_range = stir_interface.get_limits(item[0], stir_interface.segment_data.get_segment_number())

        value = 0
        if 'value' in item[1]:
            value = item[1]['value']

        output_dict[item[0]] = UISliderSpinboxItem(groupbox=UI_groupbox,
                                                   label=f"{item[1]['label']}:",
                                                   min_range=min_range,
                                                   max_range=max_range,
                                                   value=value,
                                                   connect_method=item[1]['connect_method']
                                                   )
    return output_dict
