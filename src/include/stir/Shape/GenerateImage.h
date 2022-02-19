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

  \par Example .par file
  \code
  generate_image Parameters :=

  ;;;;; a number of keywords as in Interfile
  ; (possible values are given by using a | notation)

  ; optional: values: PET|nucmed *defaults to PET for backwards compatibility)
  imaging modality:=PET
  ; optional (but recommended). Needs to be a STIR supported name
  originating system := ECAT 931
  ; optional patient position keywords (defaulting to "unknown")
  ; orientation: allowed values: head_in|feet_in|other|unknown
  patient orientation := head_in
  ; rotation: allowed values: prone|supine|other|unknown
  patient rotation :=  supine
  ; optional keywords to set image timing
  image duration (sec) := 20 ; defaults to -1 (i.e. unknown)
  image relative start time (sec) := 0 ; defaults to zero

  ;;;;; specific keywords

  output filename:= somefile
  ; optional keyword to specify the output file format
  ; example below uses Interfile with 16-bit unsigned integers
  output file format type:= Interfile
  interfile Output File Format Parameters:=
    number format := unsigned integer
    number_of_bytes_per_pixel:=2
    ; fix the scale factor to 1
    ; comment out next line to let STIR use the full dynamic
    ; range of the output type
    scale_to_write_data:= 1
  End Interfile Output File Format Parameters:=

  X output image size (in pixels):= 13
  Y output image size (in pixels):= 13
  Z output image size (in pixels):= 15
  X voxel size (in mm):= 4
  Y voxel size (in mm):= 4
  Z voxel size (in mm):= 5

  ; parameters that determine subsampling of border voxels
  ; to obtain smooth edges
  ; setting these to 1 will just check if the centre of the voxel is in or out
  ; default to 5
  Z number of samples to take per voxel := 5
  Y number of samples to take per voxel := 5
  X number of samples to take per voxel := 5

  ; now starts an enumeration of objects
  ; Shape3D hierarchy for possible shapes and their parameters
  shape type:= ellipsoidal cylinder
     Ellipsoidal Cylinder Parameters:=
     radius-x (in mm):= 1
     radius-y (in mm):= 2
     length-z (in mm):= 3
     origin (in mm):= {z,y,x}
     END:=
  value := 10

  ; here comes another shape
  next shape:=
  shape type:= ellipsoid
  ; etc

  ; as many shapes as you want
  END:=
  \endcode

  \warning If the shape is smaller than the voxel-size, or the shape
  is at the edge of the image, the current
  mechanism of generating the image might miss the shape entirely.

  \warning Does not currently support interactive parsing, so a par file
  must be given on the command line.

  \todo Code duplicates things from stir::InterfileHeader. This is bad as it might
  miss new features being added there.
  \author Kris Thielemans
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
    //! Constructor requires a parameter filename passed. 
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
