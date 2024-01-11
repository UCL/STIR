#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2022, 2024 University College London

# Author Robert Twyman

# This file is part of STIR.
#
# SPDX-License-Identifier: Apache-2.0
#
# See STIR/LICENSE.txt for details

"""
Usage: python ProjDataVisualisation.py [filename]
"""
import sys

from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                             QRadioButton, QPushButton, QStyleFactory, QVBoxLayout, QFileDialog, QLineEdit)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

from BackendTools.STIRInterface import ProjDataVisualisationBackend, ProjDataDims
from BackendTools.UIGroupboxProjdataDimensions import UIGroupboxProjDataDimensions

import stir

class ProjDataVisualisationWidgetGallery(QDialog):
    def __init__(self, parent=None):
        super(ProjDataVisualisationWidgetGallery, self).__init__(parent)

        self.originalPalette = QApplication.palette()

        self.configure_backend()


        # #############
        # ### STYLE ###
        # #############
        styleComboBox = QComboBox()
        styleComboBox.addItems(QStyleFactory.keys())
        styleLabel = QLabel("&Style:")
        styleLabel.setBuddy(styleComboBox)
        self.useStylePaletteCheckBox = QCheckBox("&Use style's standard palette")
        self.useStylePaletteCheckBox.setChecked(True)
        try: # Required for PyQt version >= v5.14
            styleComboBox.textActivated.connect(self.change_UI_style)
        except AttributeError: 
            styleComboBox.activated.connect(self.change_UI_style)
        self.useStylePaletteCheckBox.toggled.connect(self.change_UI_palette)


        # #########################
        # ### Filename GroupBox ###
        # #########################
        self.FilenameControlGroupBox = QGroupBox("ProjData File")
        # Creation group box entries
        self.projdata_filename_box = QLineEdit(self.stir_interface.projdata_filename)
        push_button_browse_projdata = QPushButton("Browse")
        push_button_browse_projdata.clicked.connect(self.browse_file_system_for_projdata)
        push_button_load_projdata = QPushButton("Load")
        push_button_load_projdata.clicked.connect(self.load_projdata)
        self.load_projdata_status = QLabel(f"")

        # Sinogram and viewgram radio buttons
        gramTypeLabel = QLabel(f"Type of 2D data:")
        self.sinogram_radio_button = QRadioButton("Sinogram")
        self.viewgram_radio_button = QRadioButton("Viewgram")
        self.sinogram_radio_button.setChecked(True)  # Default to sinogram
        self.sinogram_radio_button.toggled.connect(self.refresh_UI_configuration)
        self.viewgram_radio_button.toggled.connect(self.refresh_UI_configuration)

        # Configure Layout
        layout = QGridLayout()
        layout.addWidget(self.projdata_filename_box, 0, 0, 1, 2)
        layout.addWidget(push_button_browse_projdata, 1, 0, 1, 1)
        layout.addWidget(push_button_load_projdata, 1, 1, 1, 1)
        layout.addWidget(self.load_projdata_status, 2, 0, 1, 2)
        layout.addWidget(gramTypeLabel, 3, 0, 1, 2)
        layout.addWidget(self.sinogram_radio_button, 4, 0, 1, 1)
        layout.addWidget(self.viewgram_radio_button, 4, 1, 1, 1)
        self.FilenameControlGroupBox.setLayout(layout)


        # #############################################
        # ### ProjData Dimentional Control GroupBox ###
        # #############################################
        self.UI_groupbox_projdata_dimensions = UIGroupboxProjDataDimensions(self.stir_interface)
        methods = [self.refresh_UI_configuration]
        self.UI_groupbox_projdata_dimensions.set_UI_connect_methods(methods=methods)


        # #####################################
        # ### Visualisation Window GroupBox ###
        # #####################################
        self.ProjDataVisualisationGroupBox = QGroupBox("ProjData Visualisation")
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
        # adding toolbar to the layout
        layout.addWidget(self.display_image_matplotlib_toolbar)
        # adding canvas to the layout
        layout.addWidget(self.display_image_matplotlib_canvas)
        layout.addStretch(1)  # todo: Remove this line?
        self.ProjDataVisualisationGroupBox.setLayout(layout)


        # #############################
        # ### Configure Main Layout ###
        # #############################
        topLayout = QHBoxLayout()
        # topLayout.addStretch(1)

        mainLayout = QGridLayout()
        mainLayout.addLayout(topLayout, 0, 0, 1, 5)
        mainLayout.addWidget(self.FilenameControlGroupBox, 2, 0)
        mainLayout.addWidget(self.ProjDataVisualisationGroupBox, 1, 0, 1, 2)
        # mainLayout.addWidget(self.bottomLeftTabWidget, 2, 0)
        mainLayout.addWidget(self.UI_groupbox_projdata_dimensions.groupbox, 2, 1)
        # mainLayout.setRowStretch(1, 1)
        # mainLayout.setRowStretch(2, 1)
        # mainLayout.setColumnStretch(0, 1)
        # mainLayout.setColumnStretch(1, 1)
        self.setLayout(mainLayout)

        self.change_UI_style('Fusion')

        self.refresh_UI_configuration()

    def configure_backend(self):
        ### Backend ###
        self.stir_interface = ProjDataVisualisationBackend(sys.argv)
        self.stir_interface.load_projdata()

    def change_UI_style(self, styleName):
        QApplication.setStyle(QStyleFactory.create(styleName))
        self.change_UI_palette()

    def change_UI_palette(self):
        if (self.useStylePaletteCheckBox.isChecked()):
            QApplication.setPalette(QApplication.style().standardPalette())
        else:
            QApplication.setPalette(self.originalPalette)


    ######## UI CONFIGURATION CHANGES #########

    def refresh_UI_configuration(self):
        """
        This function is called when the user changes any of the projection data parameters.
        It updates the UI to reflect the new projection data parameters.
        It calls the updateDisplayImage function to update the display image.
        """
        self.UI_groupbox_projdata_dimensions.refresh_sliders_and_spinboxes_ranges()
        self.UI_groupbox_projdata_dimensions.configure_enable_disable_sliders(self.sinogram_radio_button.isChecked())
        self.update_display_image()

    def update_display_image(self):
        """
        This method updates the displayed image based uon the current UI configuration parameters.
        """
        if self.stir_interface.projdata is None:
            return None

        # reset the figure
        self.display_image_matplotlib_figure.clear()
        ax = self.display_image_matplotlib_figure.add_subplot(111)

        # get the projection data numpy array from the stir interface
        if self.sinogram_radio_button.isChecked():
            image = self.get_sinogram_numpy_array()
            ax.title.set_text(
                f"Sinogram - Segment: {self.UI_groupbox_projdata_dimensions.value(ProjDataDims.SEGMENT_NUM)}, "
                f"Axial Position: {self.UI_groupbox_projdata_dimensions.value(ProjDataDims.AXIAL_POS)}, "
                f"TOF bin: {self.UI_groupbox_projdata_dimensions.value(ProjDataDims.TIMING_POS)}")
            ax.yaxis.set_label_text("Views/projection angle")
            ax.xaxis.set_label_text("Tangential positions")
            ax.xaxis.set_label_text("TOF bins")
        elif self.viewgram_radio_button.isChecked():
            image = self.get_viewgram_numpy_array()
            ax.title.set_text(
                f"Sinogram - Segment: {self.UI_groupbox_projdata_dimensions.value(ProjDataDims.SEGMENT_NUM)},"
                f"View Number: {self.UI_groupbox_projdata_dimensions.value(ProjDataDims.VIEW_NUMBER)}, "
                f"TOF bin: {self.UI_groupbox_projdata_dimensions.value(ProjDataDims.TIMING_POS)}")
            ax.yaxis.set_label_text("Axial positions")
            ax.xaxis.set_label_text("Tangential positions")
            ax.xaxis.set_label_text("TOF bins")
        else:
            msg = f"Error: No radio button is checked... How did you get here?\n"
            raise Exception(msg)

        # display the image
        ax.imshow(image,
                  # cmap='gray'
                  )
        self.display_image_matplotlib_canvas.draw()

    def get_bin(self) -> stir.Bin:
        view_num = self.UI_groupbox_projdata_dimensions.value(ProjDataDims.VIEW_NUMBER)
        axial_pos = self.UI_groupbox_projdata_dimensions.value(ProjDataDims.AXIAL_POS)
        segment_num = self.UI_groupbox_projdata_dimensions.value(ProjDataDims.SEGMENT_NUM)
        tangential_pos = self.UI_groupbox_projdata_dimensions.value(ProjDataDims.TANGENTIAL_POS)
        timing_pos = self.UI_groupbox_projdata_dimensions.value(ProjDataDims.TIMING_POS)
        return stir.Bin(segment_num, view_num, axial_pos, tangential_pos, timing_pos)

    def get_sinogram_numpy_array(self):
        """
        This function returns the sinogram numpy array based on the current UI configuration parameters for segment 
        number and axial position number. 
        """
        if self.stir_interface.projdata is None:
            return None

        return self.stir_interface.as_numpy(
            self.stir_interface.segment_data.get_sinogram(self.get_bin().axial_pos_num))

    def get_viewgram_numpy_array(self):
        if self.stir_interface.projdata is None:
            return None

        return self.stir_interface.as_numpy(self.stir_interface.segment_data.get_viewgram(self.get_bin().view_num))

    def browse_file_system_for_projdata(self):
        initial = self.projdata_filename_box.text()
        filename = QFileDialog.getOpenFileName(self, "Open ProjData File", "", "ProjData Files (*.hs)")
        if len(filename[0]) > 0:
            self.projdata_filename_box.setText(filename[0])
            self.load_projdata(filename[0])
        else:
            self.projdata_filename_box.setText(initial)

    def load_projdata(self, filename=None) -> None:
        """
        This function loads the projdata file and updates the UI.
        """
        if not filename:
            filename = self.projdata_filename_box.text()
        
        if filename is not None and filename != "":
            self.projdata_filename_box.setText(filename)

        data_load_successful = self.stir_interface.load_projdata(self.projdata_filename_box.text())

        if not data_load_successful:
            self.load_projdata_status.setText("STATUS: Failed to load ProjData from file.")
            return
        
        self.load_projdata_status.setText("STATUS: ProjData loaded successfully from file.")
        self.refresh_UI_configuration()
            

    def set_projdata(self, projdata):
        self.stir_interface.set_projdata(projdata)
        self.refresh_UI_configuration()
        self.projdata_filename_box.setText("ProjData set externally.")


def OpenProjDataVisualisation(projdata=None):
    """
    Function to open the ProjDataVisualisation GUI window. Will not exit python on window close.
    projdata: Proj data to be visualised. Can be either a stir.ProjData object, a file path (str) or None. If None, an empty GUI will be opened.
    """
    app = QApplication([])
    gallery = ProjDataVisualisationWidgetGallery()
    
    if isinstance(projdata, str):
        gallery.load_projdata(projdata)
    elif projdata is None:
        pass  # Do not set projdata in ProjDataVisualisationWidgetGallery
    else:
        import stir  # Nest if statement to avoid import if not needed
        if isinstance(projdata, stir.ProjData):
            gallery.set_projdata(projdata)
        else:
            raise TypeError("projdata must be stir.ProjData, None or str")

    gallery.show()
    app.exec_()
    print("ProjDataVisualisationWidgetGallery closed!")

def main():
    app = QApplication([])
    gallery = ProjDataVisualisationWidgetGallery()
    if len(sys.argv) > 1:
        gallery.load_projdata(sys.argv[1])
    gallery.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
