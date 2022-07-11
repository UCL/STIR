#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ProjDataVisualisationUI.py
"""

import numpy as np
import stir
from PyQt5.QtCore import QDateTime, Qt, QTimer
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDateTimeEdit,
                             QDial, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                             QRadioButton, QPushButton, QScrollBar, QSizePolicy,
                             QSlider, QSpinBox, QStyleFactory, QTableWidget, QTabWidget, QTextEdit,
                             QVBoxLayout, QWidget)
from PyQt5.QtGui import QPixmap, QImage

## From demoPyQt5MatPlotLib2.py:
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

from ProjDataVisualisationBackend import ProjDataVisualisationBackend


def float32_to_uint8(array: np.ndarray) -> np.ndarray:
    """
    This function converts a float32 array to an uint8 array.
    """
    array = array.astype(np.float32)
    if array.max() == array.min():
        # Avoid division by zero
        array = array.fill(1)
    else:
        # Normalize the image between int(0) and int(255)
        array = (array - array.min()) / (array.max() - array.min())
        array = array * 255

    array = array.astype(np.uint8)
    return array


class WidgetGallery(QDialog):
    def __init__(self, parent=None):
        super(WidgetGallery, self).__init__(parent)

        self.originalPalette = QApplication.palette()

        self.configure_backend()

        styleComboBox = QComboBox()
        styleComboBox.addItems(QStyleFactory.keys())

        styleLabel = QLabel("&Style:")
        styleLabel.setBuddy(styleComboBox)

        self.useStylePaletteCheckBox = QCheckBox("&Use style's standard palette")
        self.useStylePaletteCheckBox.setChecked(True)

        # disableWidgetsCheckBox = QCheckBox("&Disable widgets")

        self.create_top_left_groupbox()
        self.create_top_right_groupbox()
        self.create_bottom_left_groupbox()

        styleComboBox.textActivated.connect(self.change_UI_style)
        self.useStylePaletteCheckBox.toggled.connect(self.change_UI_palette)

        topLayout = QHBoxLayout()
        topLayout.addStretch(1)

        mainLayout = QGridLayout()
        mainLayout.addLayout(topLayout, 0, 0, 1, 5)
        mainLayout.addWidget(self.topLeftGroupBox, 1, 0)
        mainLayout.addWidget(self.topRightGroupBox, 1, 1)
        # mainLayout.addWidget(self.bottomLeftTabWidget, 2, 0)
        mainLayout.addWidget(self.bottomLeftGroupBox, 2, 0)
        mainLayout.setRowStretch(1, 1)
        mainLayout.setRowStretch(2, 1)
        mainLayout.setColumnStretch(0, 1)
        mainLayout.setColumnStretch(1, 1)
        self.setLayout(mainLayout)

        self.change_UI_style('Fusion')

        self.update_UI_configuration()

    def configure_backend(self):
        ### Backend ###
        self.stir_interface = ProjDataVisualisationBackend(sys.argv)
        self.stir_interface.refresh_segment_data(0)

    def change_UI_style(self, styleName):
        QApplication.setStyle(QStyleFactory.create(styleName))
        self.change_UI_palette()

    def change_UI_palette(self):
        if (self.useStylePaletteCheckBox.isChecked()):
            QApplication.setPalette(QApplication.style().standardPalette())
        else:
            QApplication.setPalette(self.originalPalette)

    ##### GROUPBOX CREATION #####

    def create_top_left_groupbox(self):
        self.topLeftGroupBox = QGroupBox("FileName")

        # Creation group box entries
        filenameLabel = QLabel(f"Filename:\n{self.stir_interface.proj_data_filename}")

        gramTypeLabel = QLabel(f"Type of 2D data:")
        self.sinogram_radio_button = QRadioButton("Sinogram")
        self.viewgram_radio_button = QRadioButton("Viewgram")

        self.sinogram_radio_button.setChecked(True)
        self.sinogram_radio_button.toggled.connect(self.update_UI_configuration)
        self.viewgram_radio_button.toggled.connect(self.update_UI_configuration)

        # Configure Layout
        layout = QVBoxLayout()
        layout.addWidget(filenameLabel)

        layout.addWidget(gramTypeLabel)
        layout.addWidget(self.sinogram_radio_button)
        layout.addWidget(self.viewgram_radio_button)
        # layout.addWidget(radioButton3)

        layout.addStretch(1)
        self.topLeftGroupBox.setLayout(layout)

    def create_top_right_groupbox(self):
        self.topRightGroupBox = QGroupBox("Group 2")

        # a figure instance to plot on
        self.display_image_matplotlib_figure = plt.figure()

        # this is the Canvas Widget that
        # displays the 'figure'it takes the
        # 'figure' instance as a parameter to __init__
        self.display_image_matplotlib_canvas = FigureCanvas(self.display_image_matplotlib_figure)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.display_image_matplotlib_toolbar = NavigationToolbar(self.display_image_matplotlib_canvas, self)



        # self.displayImageLabel = QLabel(f"Image goes here...")
        # self.displayImageLabel.resize(640, 400)

        defaultPushButton = QPushButton("Default Push Button")
        defaultPushButton.setDefault(True)

        togglePushButton = QPushButton("Toggle Push Button")
        togglePushButton.setCheckable(True)
        togglePushButton.setChecked(True)

        flatPushButton = QPushButton("Flat Push Button")
        flatPushButton.setFlat(True)

        # creating a Vertical Box layout
        layout = QVBoxLayout()
        # layout.addWidget(self.displayImageLabel)
        # layout.addWidget(defaultPushButton)
        # layout.addWidget(togglePushButton)
        # layout.addWidget(flatPushButton)

        # adding tool bar to the layout
        layout.addWidget(self.display_image_matplotlib_toolbar)

        # adding canvas to the layout
        layout.addWidget(self.display_image_matplotlib_canvas)

        layout.addStretch(1)  # todo: Remove this line?
        self.topRightGroupBox.setLayout(layout)

    def create_bottom_left_groupbox(self):
        self.bottomLeftGroupBox = QGroupBox("Sinogram Positions")

        # Some default configurations
        initial_segment_number = 0  # Zero ring difference, start here
        initial_view_number = 0  # the first view, makes sense to start with this

        #### AXIAL POSITION ####
        max_axial_pos = self.stir_interface.proj_data_stream.get_max_axial_pos_num(initial_segment_number)
        min_axial_pos = self.stir_interface.proj_data_stream.get_min_axial_pos_num(initial_segment_number)
        self.axial_pos_label = QLabel(f"Axial position: {min_axial_pos, max_axial_pos}")
        self.axial_pos_spinbox = QSpinBox(self.bottomLeftGroupBox)
        self.axial_pos_spinbox.setRange(min_axial_pos, max_axial_pos)
        self.axial_pos_spinbox.setValue((max_axial_pos - min_axial_pos) // 2)
        self.axial_pos_spinbox.valueChanged.connect(self.axial_pos_spin_box_value_changed)

        self.axial_pos_slider = QSlider(Qt.Orientation.Horizontal, self.bottomLeftGroupBox)
        self.axial_pos_slider.setRange(0, max_axial_pos)
        self.axial_pos_slider.setValue(self.axial_pos_spinbox.value())
        self.axial_pos_slider.setTickPosition(QSlider.TicksBelow)
        self.axial_pos_slider.valueChanged.connect(self.axial_pos_slider_value_changed)

        if max_axial_pos == 0:
            self.axial_pos_spinbox.setEnabled(False)
            self.axial_pos_label.setEnabled(False)
            self.axial_pos_slider.setEnabled(False)

        #### VIEW NUMBER ####
        max_view_number = self.stir_interface.proj_data_stream.get_max_view_num()
        min_view_number = self.stir_interface.proj_data_stream.get_min_view_num()
        self.view_num_label = QLabel(f"View number: {min_view_number, max_view_number}")
        self.view_number_spinbox = QSpinBox(self.bottomLeftGroupBox)
        self.view_number_spinbox.setRange(min_view_number, max_view_number)
        self.view_number_spinbox.setValue(initial_view_number)
        self.view_number_spinbox.valueChanged.connect(self.view_num_spinbox_value_changed)

        self.view_number_slider = QSlider(Qt.Orientation.Horizontal, self.bottomLeftGroupBox)
        self.view_number_slider.setRange(min_view_number, max_view_number)
        self.view_number_slider.setValue(self.view_number_spinbox.value())
        self.view_number_slider.setTickPosition(QSlider.TicksBelow)
        self.view_number_slider.valueChanged.connect(self.view_num_slider_value_changed)

        #### TANGENTIAL POSITION ####
        max_tangential_pos = self.stir_interface.proj_data_stream.get_max_tangential_pos_num()
        min_tangential_pos = self.stir_interface.proj_data_stream.get_min_tangential_pos_num()
        self.tangential_pos_label = QLabel(f"Tangential position: {min_tangential_pos, max_tangential_pos}")
        self.tangential_pos_spinbox = QSpinBox(self.bottomLeftGroupBox)
        self.tangential_pos_spinbox.setRange(min_tangential_pos, max_tangential_pos)
        self.tangential_pos_spinbox.setValue((max_tangential_pos - min_tangential_pos) // 2)
        self.tangential_pos_spinbox.valueChanged.connect(self.tangential_pos_spin_box_value_changed)

        self.tangential_pos_slider = QSlider(Qt.Orientation.Horizontal, self.bottomLeftGroupBox)
        self.tangential_pos_slider.setRange(0, max_tangential_pos)
        self.tangential_pos_slider.setValue(self.tangential_pos_spinbox.value())
        self.tangential_pos_slider.setTickPosition(QSlider.TicksBelow)
        self.tangential_pos_slider.valueChanged.connect(self.tangential_pos_slider_value_changed)

        #### SEGMENT NUMBER ####
        max_segment_number = self.stir_interface.proj_data_stream.get_max_segment_num()
        min_segment_number = self.stir_interface.proj_data_stream.get_min_segment_num()
        self.segment_number_label = QLabel(f"Segment Number: {min_segment_number, max_segment_number}")
        self.segment_number_spinbox = QSpinBox(self.bottomLeftGroupBox)
        self.segment_number_spinbox.setRange(min_segment_number, max_segment_number)
        self.segment_number_spinbox.setValue(initial_segment_number)
        self.segment_number_spinbox.valueChanged.connect(self.segment_number_spin_box_value_changed)

        self.segment_number_slider = QSlider(Qt.Orientation.Horizontal, self.bottomLeftGroupBox)
        self.segment_number_slider.setRange(min_segment_number, max_segment_number)
        self.segment_number_slider.setValue(self.segment_number_spinbox.value())
        self.segment_number_slider.setTickPosition(QSlider.TicksBelow)
        self.segment_number_slider.valueChanged.connect(self.segment_number_slider_value_changed)

        # self.showSinogramPushButton = QPushButton("Show Sinogram")
        # self.showSinogramPushButton.setDefault(True)

        ##### LAYOUT ####
        layout = QGridLayout()
        # layout.addWidget(lineEdit, 0, 0, 1, 2)

        layout.addWidget(self.segment_number_label, 0, 0, 1, 1)
        layout.addWidget(self.segment_number_slider, 1, 0, 1, 1)
        layout.addWidget(self.segment_number_spinbox, 1, 1, 1, 1)

        layout.addWidget(self.axial_pos_label, 2, 0, 1, 1)
        layout.addWidget(self.axial_pos_slider, 3, 0, 1, 1)
        layout.addWidget(self.axial_pos_spinbox, 3, 1, 1, 1)

        # view number
        layout.addWidget(self.view_num_label, 4, 0, 1, 1)
        layout.addWidget(self.view_number_slider, 5, 0, 1, 1)
        layout.addWidget(self.view_number_spinbox, 5, 1, 1, 1)

        # tangential position
        layout.addWidget(self.tangential_pos_label, 6, 0, 1, 1)
        layout.addWidget(self.tangential_pos_slider, 7, 0, 1, 1)
        layout.addWidget(self.tangential_pos_spinbox, 7, 1, 1, 1)

        # layout.addWidget(self.showSinogramPushButton, 4, 0, 1, 2)

        layout.setRowStretch(5, 1)
        self.bottomLeftGroupBox.setLayout(layout)

    def segment_number_slider_value_changed(self):
        """
        This function is called when the user changes the segment number slider value.
        """
        self.segment_number_spinbox.setValue(self.segment_number_slider.value())
        self.stir_interface.refresh_segment_data(self.segment_number_slider.value())
        self.update_UI_configuration()

    def segment_number_spin_box_value_changed(self):
        """
        This function is called when the user changes the segment number spinbox value.
        """
        self.segment_number_slider.setValue(self.segment_number_spinbox.value())
        self.stir_interface.refresh_segment_data(self.segment_number_spinbox.value())
        self.update_UI_configuration()

    def axial_pos_slider_value_changed(self):
        """
        This function is called when the user changes the axial position slider value.
        """
        self.axial_pos_spinbox.setValue(self.axial_pos_slider.value())
        self.update_UI_configuration()

    def axial_pos_spin_box_value_changed(self):
        """
        This function is called when the user changes the axial position spinbox value.
        """
        self.axial_pos_slider.setValue(self.axial_pos_spinbox.value())
        self.update_UI_configuration()

    def tangential_pos_slider_value_changed(self):
        """
        This function is called when the user changes the tangential position slider value.
        """
        self.tangential_pos_spinbox.setValue(self.tangential_pos_slider.value())
        self.update_UI_configuration()

    def tangential_pos_spin_box_value_changed(self):
        """
        This function is called when the user changes the tangential position spinbox value.
        """
        self.tangential_pos_slider.setValue(self.tangential_pos_spinbox.value())
        self.update_UI_configuration()

    def view_num_slider_value_changed(self):
        """
        This function is called when the user changes the view num slider value.
        """
        self.view_number_spinbox.setValue(self.view_number_slider.value())
        self.update_UI_configuration()

    def view_num_spinbox_value_changed(self):
        """
        This function is called when the user changes the view num spinbox value.
        """
        self.view_number_slider.setValue(self.view_number_spinbox.value())
        self.update_UI_configuration()

    ######## UI CONFIGURATION CHANGES #########

    def update_axial_pos_spin_box_and_slider_range(self):
        """
        This function is called when the segment number is changed to automatically update the axial position slider
        and spinbox.
        Also, it updates the axial position value for both the slider and spinbox to keep within range.
        """
        min_axial_pos = self.stir_interface.segment_data.get_min_axial_pos_num()
        max_axial_pos = self.stir_interface.segment_data.get_max_axial_pos_num()
        new_axial_value = min(max_axial_pos, max(self.axial_pos_slider.value(), min_axial_pos))

        self.axial_pos_slider.setRange(min_axial_pos, max_axial_pos)
        self.axial_pos_spinbox.setRange(min_axial_pos, max_axial_pos)
        self.axial_pos_slider.setValue(new_axial_value)
        self.axial_pos_spinbox.setValue(new_axial_value)

    def update_UI_configuration(self):
        """
        This function is called when the user changes any of the projection data parameters.
        It updates the UI to reflect the new projection data parameters.
        It calls the updateDisplayImage function to update the display image.
        """

        # Segment slider and scroll box handling
        if self.stir_interface.proj_data_stream.get_max_segment_num() == 0 and \
                self.stir_interface.proj_data_stream.get_min_segment_num() == 0:
            self.segment_number_spinbox.setEnabled(False)
            self.segment_number_slider.setEnabled(False)
            self.segment_number_label.setEnabled(False)

        # Check if sinogram or viewgram is selected and disable the appropriate sliders and spinboxs
        if self.sinogram_radio_button.isChecked():
            # Disable the tangential position position slider and spinbox
            self.view_num_label.setEnabled(False)
            self.view_number_slider.setEnabled(False)
            self.view_number_spinbox.setEnabled(False)
            self.axial_pos_label.setEnabled(True)
            self.axial_pos_slider.setEnabled(True)
            self.axial_pos_spinbox.setEnabled(True)

        elif self.viewgram_radio_button.isChecked():
            self.axial_pos_label.setEnabled(False)
            self.axial_pos_slider.setEnabled(False)
            self.axial_pos_spinbox.setEnabled(False)
            self.view_num_label.setEnabled(True)
            self.view_number_spinbox.setEnabled(True)
            self.view_number_slider.setEnabled(True)

        if True:
            self.tangential_pos_label.setEnabled(False)
            self.tangential_pos_slider.setEnabled(False)
            self.tangential_pos_spinbox.setEnabled(False)

        self.update_axial_pos_spin_box_and_slider_range()
        self.update_display_image()

    def update_display_image(self):
        """
        This method updates the displayed image based uon the current UI configuration parameters.
        """

        # reset the figure
        self.display_image_matplotlib_figure.clear()
        ax = self.display_image_matplotlib_figure.add_subplot(111)

        # get the projection data numpy array from the stir interface
        if self.sinogram_radio_button.isChecked():
            image = self.get_sinogram_numpy_array()
            ax.title.set_text(f"Sinogram - Segment: {self.segment_number_spinbox.value()}, "
                              f"Axial Position: {self.axial_pos_spinbox.value()}")
            ax.yaxis.set_label_text("Views/projection angle")
            ax.xaxis.set_label_text("Tangential positions")
        elif self.viewgram_radio_button.isChecked():
            image = self.get_viewgram_numpy_array()
            ax.title.set_text(f"Sinogram - Segment: {self.segment_number_spinbox.value()},"
                              f"View Number: {self.view_number_spinbox.value()}")
            ax.yaxis.set_label_text("Axial positions")
            ax.xaxis.set_label_text("Tangential positions")
        else:
            msg = f"Error: No radio button is checked... How did you get here?\n"
            raise Exception(msg)

        # display the image
        ax.imshow(image,
                  # cmap='gray'
                  )
        self.display_image_matplotlib_canvas.draw()

    def get_sinogram_numpy_array(self):
        """
        This function returns the sinogram numpy array based on the current UI configuration parameters for segment 
        number and axial position number. 
        """
        return self.stir_interface.as_numpy(
            self.stir_interface.segment_data.get_sinogram(self.axial_pos_spinbox.value())
        )

    def get_viewgram_numpy_array(self):
        return self.stir_interface.as_numpy(
            self.stir_interface.segment_data.get_viewgram(self.view_number_spinbox.value())
        )


if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    gallery = WidgetGallery()
    gallery.show()
    sys.exit(app.exec())
