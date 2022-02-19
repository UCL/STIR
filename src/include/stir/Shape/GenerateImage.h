//
//
/*
    Copyright (C) 2003-2011, Hammersmith Imanet Ltd
    Copyright (C) 2018-2022, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup Shape

  \brief Declaration of class stir::GenerateImage

  \author Kris Thielemans
  \author Sanida Mustafovic
  \author Robert Twyman

  *See generate_image utility documentation for parameter file example and usage documentation.
*/

#include "stir/Shape/Shape3D.h"
#include "stir/PatientPosition.h"
#include "stir/ImagingModality.h"
#include "stir/KeyParser.h"
#include "stir/is_null_ptr.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/IndexRange3D.h"
#include "stir/Succeeded.h"
#include "stir/IO/OutputFileFormat.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include <iostream>


#ifndef __stir_Shape_GenerateImage_h__
#define __stir_Shape_GenerateImage_h__



START_NAMESPACE_STIR

class GenerateImage : public KeyParser
{
public:
    //! Constructor requires a parameter filename passed. See \generate_image utility for details.
    explicit GenerateImage(const char * const par_filename);

    //! Computes the shapes onto a discretised density
    Succeeded compute();

    //! Writes the discretised density to file. Filename is set by initial parameters.
    Succeeded save_image();

    //! Returns the discretised density with computed shapes.
    shared_ptr<DiscretisedDensity<3, float>> get_output_sptr();

private:

    virtual void set_defaults();
    virtual void initialise_keymap();
    virtual bool post_processing();
    void set_imaging_modality();
    shared_ptr<ExamInfo> exam_info_sptr;
    std::string imaging_modality_as_string;
    ASCIIlist_type patient_orientation_values;
    ASCIIlist_type patient_rotation_values;
    int patient_orientation_index;
    int patient_rotation_index;

    shared_ptr<DiscretisedDensity<3, float> > out_density_ptr;

    std::vector<shared_ptr<Shape3D> > shape_ptrs;
    shared_ptr<Shape3D> current_shape_ptr;
    std::vector<float> values;
    float current_value;
    std::string output_filename;
    shared_ptr<OutputFileFormat<DiscretisedDensity<3,float> > > output_file_format_sptr;

    void increment_current_shape_num();

    int output_image_size_x;
    int output_image_size_y;
    int output_image_size_z;
    float output_voxel_size_x;
    float output_voxel_size_y;
    float output_voxel_size_z;

    CartesianCoordinate3D<int> num_samples;

    double image_duration;
    double rel_start_time;
};

END_NAMESPACE_STIR

#endif // __stir_Shape_GenerateImage_h__
