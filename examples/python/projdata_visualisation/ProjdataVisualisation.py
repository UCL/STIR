#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ProjdataVisualisation.py
"""
import sys

from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                             QRadioButton, QPushButton, QStyleFactory, QVBoxLayout, QFileDialog, QLineEdit)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

from ProjDataVisualisationBackendTools.STIRInterface import ProjDataVisualisationBackend, \
    ProjdataDims
from ProjDataVisualisationBackendTools.UIGroupboxProjdataDimensions import \
    UIGroupboxProjdataDimensions


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
        styleComboBox.textActivated.connect(self.change_UI_style)
        self.useStylePaletteCheckBox.toggled.connect(self.change_UI_palette)


        # #########################
        # ### Filename GroupBox ###
        # #########################
        self.filenameControlGroupBox = QGroupBox("ProjData File")
        # Creation group box entries
        self.projdata_filename_box = QLineEdit(self.stir_interface.proj_data_filename)
        push_button_browse_projdata = QPushButton("Browse")
        push_button_browse_projdata.clicked.connect(self.browse_file_system_for_projdata)
        push_button_load_projdata = QPushButton("Load")
        push_button_load_projdata.clicked.connect(self.load_projdata)

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
        layout.addWidget(gramTypeLabel, 2, 0, 1, 2)
        layout.addWidget(self.sinogram_radio_button, 3, 0, 1, 1)
        layout.addWidget(self.viewgram_radio_button, 3, 1, 1, 1)
        self.filenameControlGroupBox.setLayout(layout)


        # #############################################
        # ### ProjData Dimentional Control GroupBox ###
        # #############################################
        self.UI_groupbox_projdata_dimensions = UIGroupboxProjdataDimensions(self.stir_interface)
        methods = [self.refresh_UI_configuration]
        self.UI_groupbox_projdata_dimensions.set_UI_connect_methods(methods=methods)


        # #####################################
        # ### Visualisation Window GroupBox ###
        # #####################################
        self.projDataVisualisationGroupBox = QGroupBox("ProjData Visualisation")
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
        self.projDataVisualisationGroupBox.setLayout(layout)


        topLayout = QHBoxLayout()
        # topLayout.addStretch(1)

        mainLayout = QGridLayout()
        mainLayout.addLayout(topLayout, 0, 0, 1, 5)
        mainLayout.addWidget(self.filenameControlGroupBox, 2, 0)
        mainLayout.addWidget(self.projDataVisualisationGroupBox, 1, 1)
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
        self.stir_interface.refresh_segment_data(0)

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
        self.stir_interface
        self.UI_groupbox_projdata_dimensions.refresh_sliders_and_spinboxes_ranges()
        self.UI_groupbox_projdata_dimensions.update_enable_disable(self.sinogram_radio_button.isChecked())
        self.update_display_image()

    def update_display_image(self):
        """
        This method updates the displayed image based uon the current UI configuration parameters.
        """
        if self.stir_interface.proj_data_stream is None:
            return None

        # reset the figure
        self.display_image_matplotlib_figure.clear()
        ax = self.display_image_matplotlib_figure.add_subplot(111)

        # get the projection data numpy array from the stir interface
        if self.sinogram_radio_button.isChecked():
            image = self.get_sinogram_numpy_array()
            ax.title.set_text(
                f"Sinogram - Segment: {self.UI_groupbox_projdata_dimensions.value(ProjdataDims.SEGMENT_NUM)}, "
                f"Axial Position: {self.UI_groupbox_projdata_dimensions.value(ProjdataDims.AXIAL_POS)}")
            ax.yaxis.set_label_text("Views/projection angle")
            ax.xaxis.set_label_text("Tangential positions")
        elif self.viewgram_radio_button.isChecked():
            image = self.get_viewgram_numpy_array()
            ax.title.set_text(
                f"Sinogram - Segment: {self.UI_groupbox_projdata_dimensions.value(ProjdataDims.SEGMENT_NUM)},"
                f"View Number: {self.UI_groupbox_projdata_dimensions.value(ProjdataDims.VIEW_NUMBER)}")
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
        if self.stir_interface.proj_data_stream is None:
            return None

        axial_pos = self.UI_groupbox_projdata_dimensions.value(ProjdataDims.AXIAL_POS)
        return self.stir_interface.as_numpy(
            self.stir_interface.segment_data.get_sinogram(axial_pos))

    def get_viewgram_numpy_array(self):
        if self.stir_interface.proj_data_stream is None:
            return None

        view_num = self.UI_groupbox_projdata_dimensions.value(ProjdataDims.VIEW_NUMBER)
        return self.stir_interface.as_numpy(self.stir_interface.segment_data.get_viewgram(view_num))

    def browse_file_system_for_projdata(self):
        initial = self.projdata_filename_box.text()
        filename = QFileDialog.getOpenFileName(self, "Open ProjData File", "", "ProjData Files (*.hs)")
        if len(filename[0]) > 0:
            self.projdata_filename_box.setText(filename[0])
            self.load_projdata()
        else:
            self.projdata_filename_box.setText(initial)

    def load_projdata(self):
        """
        This function loads the projdata file and updates the UI.
        """
        self.stir_interface.load_projdata(self.projdata_filename_box.text())
        self.refresh_UI_configuration()


def main():
    app = QApplication(sys.argv)
    gallery = ProjDataVisualisationWidgetGallery()
    gallery.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
