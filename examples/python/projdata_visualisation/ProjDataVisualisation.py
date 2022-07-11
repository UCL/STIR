#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ProjDataVisualisation.py
"""

from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                             QRadioButton, QPushButton, QStyleFactory, QVBoxLayout)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

from projdata_visualisation.ProjDataVisualisationBackend import ProjDataVisualisationBackend
from projdata_visualisation.ProjDataVisualisationUITools import UISliderSpinboxItem


class ProjDataVisualisationWidgetGallery(QDialog):
    def __init__(self, parent=None):
        super(ProjDataVisualisationWidgetGallery, self).__init__(parent)

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


        # creating a Vertical Box layout
        layout = QVBoxLayout()

        # adding tool bar to the layout
        layout.addWidget(self.display_image_matplotlib_toolbar)

        # adding canvas to the layout
        layout.addWidget(self.display_image_matplotlib_canvas)

        layout.addStretch(1)  # todo: Remove this line?
        self.topRightGroupBox.setLayout(layout)

    def create_bottom_left_groupbox(self):
        self.bottomLeftGroupBox = QGroupBox("Sinogram Positions")

        #### SEGMENT NUMBER ####
        max_segment_number = self.stir_interface.proj_data_stream.get_max_segment_num()
        min_segment_number = self.stir_interface.proj_data_stream.get_min_segment_num()
        initial_segment_number = 0  # Zero ring difference, start here
        self.UI_slider_spinbox_item_segment = \
            UISliderSpinboxItem(groupbox=self.bottomLeftGroupBox,
                                label=f"Segment Number:",
                                min_range=min_segment_number,
                                max_range=max_segment_number,
                                value=initial_segment_number,
                                connect_method=self.segment_number_refresh)

        #### AXIAL POSITION ####
        max_axial_pos = self.stir_interface.proj_data_stream.get_max_axial_pos_num(initial_segment_number)
        min_axial_pos = self.stir_interface.proj_data_stream.get_min_axial_pos_num(initial_segment_number)
        initial_axial_pos = (max_axial_pos - min_axial_pos) // 2
        self.UI_slider_spinbox_item_axial_pos = \
            UISliderSpinboxItem(groupbox=self.bottomLeftGroupBox,
                                label=f"Axial position:",
                                min_range=min_axial_pos,
                                max_range=max_axial_pos,
                                value=initial_axial_pos,
                                connect_method=self.axial_pos_refresh)
        if (max_axial_pos - min_axial_pos) == 0:
            self.UI_slider_spinbox_item_axial_pos.disable()

        #### VIEW NUMBER ####
        max_view_number = self.stir_interface.proj_data_stream.get_max_view_num()
        min_view_number = self.stir_interface.proj_data_stream.get_min_view_num()
        initial_view_number = 0  # the first view, makes sense to start with this
        self.UI_slider_spinbox_item_view_num = \
            UISliderSpinboxItem(groupbox=self.bottomLeftGroupBox,
                                label=f"View number:",
                                min_range=min_view_number,
                                max_range=max_view_number,
                                value=initial_view_number,
                                connect_method=self.view_num_refresh)

        #### TANGENTIAL POSITION ####
        max_tangential_pos = self.stir_interface.proj_data_stream.get_max_tangential_pos_num()
        min_tangential_pos = self.stir_interface.proj_data_stream.get_min_tangential_pos_num()
        initial_tangential_pos = (max_tangential_pos - min_tangential_pos) // 2
        self.UI_slider_spinbox_item_tangential_pos = \
            UISliderSpinboxItem(groupbox=self.bottomLeftGroupBox,
                                label=f"Tangential position:",
                                min_range=min_tangential_pos,
                                max_range=max_tangential_pos,
                                value=initial_tangential_pos,
                                connect_method=self.tangential_pos_refresh)

        ##### LAYOUT ####
        layout = QGridLayout()
        # layout.addWidget(lineEdit, 0, 0, 1, 2)
        self.UI_slider_spinbox_item_segment.add_to_layout(layout, row=0)
        self.UI_slider_spinbox_item_axial_pos.add_to_layout(layout, row=2)
        self.UI_slider_spinbox_item_view_num.add_to_layout(layout, row=4)
        self.UI_slider_spinbox_item_tangential_pos.add_to_layout(layout, row=6)

        layout.setRowStretch(5, 1)
        self.bottomLeftGroupBox.setLayout(layout)

    def segment_number_refresh(self):
        """
        This function is called when the user changes the segment number value.
        """
        self.stir_interface.refresh_segment_data(self.UI_slider_spinbox_item_segment.value())
        self.update_axial_pos_spin_box_and_slider_range()
        self.update_UI_configuration()

    def axial_pos_refresh(self):
        """
        This function is called when the user changes the axial position value.
        """
        self.update_UI_configuration()

    def view_num_refresh(self):
        """
        This function is called when the user changes the view number value.
        """
        self.update_UI_configuration()

    def tangential_pos_refresh(self):
        """
        This function is called when the user changes the tangential position value.
        """
        self.update_UI_configuration()

    ######## UI CONFIGURATION CHANGES #########

    def update_axial_pos_spin_box_and_slider_range(self):
        """
        This function is called when the segment number is changed to automatically update the axial position slider
        and spinbox.
        Also, it updates the axial position value for both the slider and spinbox to keep within range.
        """
        self.UI_slider_spinbox_item_axial_pos.update_range(self.stir_interface.segment_data.get_min_axial_pos_num(),
                                                           self.stir_interface.segment_data.get_max_axial_pos_num())

    def update_UI_configuration(self):
        """
        This function is called when the user changes any of the projection data parameters.
        It updates the UI to reflect the new projection data parameters.
        It calls the updateDisplayImage function to update the display image.
        """

        # Segment slider and scroll box handling
        if self.stir_interface.proj_data_stream.get_max_segment_num() == 0 and \
                self.stir_interface.proj_data_stream.get_min_segment_num() == 0:
            self.UI_slider_spinbox_item_segment.disable()

        # Check if sinogram or viewgram is selected and disable the appropriate sliders and spinboxs
        if self.sinogram_radio_button.isChecked():
            # Disable the tangential position position slider and spinbox
            self.UI_slider_spinbox_item_view_num.disable()
            self.UI_slider_spinbox_item_axial_pos.enable()

        elif self.viewgram_radio_button.isChecked():
            self.UI_slider_spinbox_item_axial_pos.disable()
            self.UI_slider_spinbox_item_view_num.enable()

        if True:  # Until I work out what to do with tangential position
            self.UI_slider_spinbox_item_tangential_pos.disable()

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
            ax.title.set_text(f"Sinogram - Segment: {self.UI_slider_spinbox_item_segment.value()}, "
                              f"Axial Position: {self.UI_slider_spinbox_item_axial_pos.value()}")
            ax.yaxis.set_label_text("Views/projection angle")
            ax.xaxis.set_label_text("Tangential positions")
        elif self.viewgram_radio_button.isChecked():
            image = self.get_viewgram_numpy_array()
            ax.title.set_text(f"Sinogram - Segment: {self.UI_slider_spinbox_item_segment.value()},"
                              f"View Number: {self.UI_slider_spinbox_item_view_num.value()}")
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
            self.stir_interface.segment_data.get_sinogram(self.UI_slider_spinbox_item_axial_pos.value())
        )

    def get_viewgram_numpy_array(self):
        return self.stir_interface.as_numpy(
            self.stir_interface.segment_data.get_viewgram(self.UI_slider_spinbox_item_view_num.value())
        )


if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    gallery = ProjDataVisualisationWidgetGallery()
    gallery.show()
    sys.exit(app.exec())
