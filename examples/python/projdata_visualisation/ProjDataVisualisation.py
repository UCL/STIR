#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ProjDataVisualisation.py
"""

from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                             QRadioButton, QPushButton, QStyleFactory, QVBoxLayout, QFileDialog, QLineEdit)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

from projdata_visualisation.ProjDataVisualisationBackendTools.STIRInterface import ProjDataVisualisationBackend, \
    SinogramDimensions
from projdata_visualisation.ProjDataVisualisationBackendTools.UITools import \
    construct_slider_spinboxes


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
        self.projdata_filename_box = QLineEdit(self.stir_interface.proj_data_filename)

        push_button_browse_projdata = QPushButton("Browse")
        push_button_browse_projdata.clicked.connect(self.browse_file_system_for_projdata)
        push_button_load_projdata = QPushButton("Load")
        push_button_load_projdata.clicked.connect(self.load_projdata)

        gramTypeLabel = QLabel(f"Type of 2D data:")
        self.sinogram_radio_button = QRadioButton("Sinogram")
        self.viewgram_radio_button = QRadioButton("Viewgram")

        self.sinogram_radio_button.setChecked(True)
        self.sinogram_radio_button.toggled.connect(self.update_UI_configuration)
        self.viewgram_radio_button.toggled.connect(self.update_UI_configuration)

        # Configure Layout
        layout = QGridLayout()
        layout.addWidget(self.projdata_filename_box, 0, 0, 1, 2)

        layout.addWidget(push_button_browse_projdata, 1, 0, 1, 1)
        layout.addWidget(push_button_load_projdata, 1, 1, 1, 1)

        layout.addWidget(gramTypeLabel, 2, 0, 1, 2)
        layout.addWidget(self.sinogram_radio_button, 3, 0, 1, 1)
        layout.addWidget(self.viewgram_radio_button, 3, 1, 1, 1)
        # layout.addWidget(radioButton3)

        # layout.addStretch(1)
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

        # adding toolbar to the layout
        layout.addWidget(self.display_image_matplotlib_toolbar)

        # adding canvas to the layout
        layout.addWidget(self.display_image_matplotlib_canvas)

        layout.addStretch(1)  # todo: Remove this line?
        self.topRightGroupBox.setLayout(layout)

    def create_bottom_left_groupbox(self):
        self.bottomLeftGroupBox = QGroupBox("Projection Data Dimensions")

        UI_config_dict = {
            SinogramDimensions.SEGMENT_NUM: {
                'label': 'Segment number',
                'value': 0,
                'connect_method': self.segment_number_refresh
            },
            SinogramDimensions.AXIAL_POS: {
                'label': 'Axial position',
                'connect_method': self.axial_pos_refresh
            },
            SinogramDimensions.VIEW_NUMBER: {
                'label': 'View number',
                'value': 0,
                'connect_method': self.view_num_refresh
            },
            SinogramDimensions.TANGENTIAL_POS: {
                'label': 'Tangential position',
                'connect_method': self.tangential_pos_refresh
            }
        }

        self.UI_slider_spinboxes = construct_slider_spinboxes(stir_interface=self.stir_interface,
                                                              UI_groupbox=self.bottomLeftGroupBox,
                                                              configuration=UI_config_dict)

        ##### LAYOUT ####
        layout = QGridLayout()
        # layout.addWidget(lineEdit, 0, 0, 1, 2)
        self.UI_slider_spinboxes[SinogramDimensions.SEGMENT_NUM].add_to_layout(layout, row=0)
        self.UI_slider_spinboxes[SinogramDimensions.AXIAL_POS].add_to_layout(layout, row=2)
        self.UI_slider_spinboxes[SinogramDimensions.VIEW_NUMBER].add_to_layout(layout, row=4)
        self.UI_slider_spinboxes[SinogramDimensions.TANGENTIAL_POS].add_to_layout(layout, row=6)

        layout.setRowStretch(5, 1)
        self.bottomLeftGroupBox.setLayout(layout)

    def segment_number_refresh(self):
        """
        This function is called when the user changes the segment number value.
        """
        self.stir_interface.refresh_segment_data(self.UI_slider_spinboxes[SinogramDimensions.SEGMENT_NUM].value())
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

    def update_sliders_and_spinboxs_ranges(self) -> None:
        """

        """
        if self.stir_interface.proj_data_stream is None:
            return
        for dimension in SinogramDimensions:
            self.UI_slider_spinboxes[dimension].update_limits(
                self.stir_interface.get_limits(dimension, self.stir_interface.get_current_segment_num()))

    def update_axial_pos_slider_spinbox_limits(self):
        """
        This function is called when the segment number is changed to automatically update the axial position slider
        and spinbox.
        Also, it updates the axial position value for both the slider and spinbox to keep within range.
        """
        self.UI_slider_spinboxes[SinogramDimensions.AXIAL_POS].update_limits(
            self.stir_interface.get_limits(SinogramDimensions.SEGMENT_NUM,
                                           self.stir_interface.get_current_segment_num()
                                           ))

    def update_UI_configuration(self):
        """
        This function is called when the user changes any of the projection data parameters.
        It updates the UI to reflect the new projection data parameters.
        It calls the updateDisplayImage function to update the display image.
        """
        self.update_sliders_and_spinboxs_ranges()

        # Segment slider and scroll box handling
        if self.stir_interface.proj_data_stream is not None:
            if self.stir_interface.proj_data_stream.get_max_segment_num() == 0 and \
                    self.stir_interface.proj_data_stream.get_min_segment_num() == 0:
                self.UI_slider_spinboxes[SinogramDimensions.SEGMENT_NUM].disable()
            else:
                self.UI_slider_spinboxes[SinogramDimensions.SEGMENT_NUM].enable()

        # Check if sinogram or viewgram is selected and disable the appropriate sliders and spinboxs
        if self.sinogram_radio_button.isChecked():
            # Disable the tangential position slider and spinbox
            self.UI_slider_spinboxes[SinogramDimensions.VIEW_NUMBER].disable()
            if 0 == (self.UI_slider_spinboxes[SinogramDimensions.AXIAL_POS].get_limits()[0] -
                     self.UI_slider_spinboxes[SinogramDimensions.AXIAL_POS].get_limits()[1]):
                self.UI_slider_spinboxes[SinogramDimensions.AXIAL_POS].disable()
            else:
                self.UI_slider_spinboxes[SinogramDimensions.AXIAL_POS].enable()

        elif self.viewgram_radio_button.isChecked():
            self.UI_slider_spinboxes[SinogramDimensions.AXIAL_POS].disable()
            self.UI_slider_spinboxes[SinogramDimensions.VIEW_NUMBER].enable()

        if True:  # Until I work out what to do with tangential position
            self.UI_slider_spinboxes[SinogramDimensions.TANGENTIAL_POS].disable()

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
                f"Sinogram - Segment: {self.UI_slider_spinboxes[SinogramDimensions.SEGMENT_NUM].value()}, "
                f"Axial Position: {self.UI_slider_spinboxes[SinogramDimensions.AXIAL_POS].value()}")
            ax.yaxis.set_label_text("Views/projection angle")
            ax.xaxis.set_label_text("Tangential positions")
        elif self.viewgram_radio_button.isChecked():
            image = self.get_viewgram_numpy_array()
            ax.title.set_text(f"Sinogram - Segment: {self.UI_slider_spinboxes[SinogramDimensions.SEGMENT_NUM].value()},"
                              f"View Number: {self.UI_slider_spinboxes[SinogramDimensions.VIEW_NUMBER].value()}")
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
        return self.stir_interface.as_numpy(
            self.stir_interface.segment_data.get_sinogram(
                self.UI_slider_spinboxes[SinogramDimensions.AXIAL_POS].value())
        )

    def get_viewgram_numpy_array(self):
        if self.stir_interface.proj_data_stream is None:
            return None
        return self.stir_interface.as_numpy(
            self.stir_interface.segment_data.get_viewgram(
                self.UI_slider_spinboxes[SinogramDimensions.VIEW_NUMBER].value())
        )

    def browse_file_system_for_projdata(self):
        initial = self.projdata_filename_box.text()
        fname = QFileDialog.getOpenFileName(self, "Open ProjData File", "", "ProjData Files (*.hs)")
        if len(fname[0]) > 0:
            self.projdata_filename_box.setText(fname[0])
            self.load_projdata()
        else:
            self.projdata_filename_box.setText(initial)

    def load_projdata(self):
        """
        This function loads the projdata file and updates the UI.
        """
        self.stir_interface.load_projdata(self.projdata_filename_box.text())
        self.update_UI_configuration()


if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    gallery = ProjDataVisualisationWidgetGallery()
    gallery.show()
    sys.exit(app.exec())
