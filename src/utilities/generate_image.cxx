/*
  Copyright (C) 2003-2011, Hammersmith Imanet Ltd
  Copyright (C) 2018-2020, University College London
  This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup utilities

  \brief A utility to generate images consistening of uniform objects added/subtracted together

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

START_NAMESPACE_STIR

class GenerateImage : public KeyParser {
public:
  GenerateImage(const char* const par_filename);

  Succeeded compute();

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

  std::vector<shared_ptr<Shape3D>> shape_ptrs;
  shared_ptr<Shape3D> current_shape_ptr;
  std::vector<float> values;
  float current_value;
  std::string output_filename;
  shared_ptr<OutputFileFormat<DiscretisedDensity<3, float>>> output_file_format_sptr;

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

void
GenerateImage::increment_current_shape_num() {
  if (!is_null_ptr(current_shape_ptr)) {
    shape_ptrs.push_back(current_shape_ptr);
    values.push_back(current_value);
    current_shape_ptr.reset();
  }
}

void
GenerateImage::set_defaults() {
  exam_info_sptr.reset(new ExamInfo);
  // need to default to PET for backwards compatibility
  exam_info_sptr->imaging_modality = ImagingModality::PT;
  patient_orientation_index = 3; // unknown
  patient_rotation_index = 5;    // unknown
  output_image_size_x = 128;
  output_image_size_y = 128;
  output_image_size_z = 1;
  output_voxel_size_x = 1;
  output_voxel_size_y = 1;
  output_voxel_size_z = 1;
  num_samples = CartesianCoordinate3D<int>(5, 5, 5);
  shape_ptrs.resize(0);
  values.resize(0);
  image_duration = -1.0;
  rel_start_time = 0;
  output_filename.resize(0);
  output_file_format_sptr = OutputFileFormat<DiscretisedDensity<3, float>>::default_sptr();
}

void
GenerateImage::set_imaging_modality() {
  set_variable();
  this->exam_info_sptr->imaging_modality = ImagingModality(imaging_modality_as_string);
}

void
GenerateImage::initialise_keymap() {
  add_start_key("generate_image Parameters");
  // copy of InterfileHeader (TODO)
  add_key("imaging modality", KeyArgument::ASCII, (KeywordProcessor)&GenerateImage::set_imaging_modality,
          &imaging_modality_as_string);
  add_key("originating system", &exam_info_sptr->originating_system);
  add_key("patient orientation", &patient_orientation_index, &patient_orientation_values);
  add_key("patient rotation", &patient_rotation_index, &patient_rotation_values);
  patient_orientation_values.push_back("head_in");
  patient_orientation_values.push_back("feet_in");
  patient_orientation_values.push_back("other");
  patient_orientation_values.push_back("unknown"); // default

  patient_rotation_values.push_back("supine");
  patient_rotation_values.push_back("prone");
  patient_rotation_values.push_back("right");
  patient_rotation_values.push_back("left");
  patient_rotation_values.push_back("other");
  patient_rotation_values.push_back("unknown"); // default

  add_key("output filename", &output_filename);
  add_parsing_key("output file format type", &output_file_format_sptr);
  add_key("X output image size (in pixels)", &output_image_size_x);
  add_key("Y output image size (in pixels)", &output_image_size_y);
  add_key("Z output image size (in pixels)", &output_image_size_z);
  add_key("X voxel size (in mm)", &output_voxel_size_x);
  add_key("Y voxel size (in mm)", &output_voxel_size_y);
  add_key("Z voxel size (in mm)", &output_voxel_size_z);

  add_key("Z number of samples to take per voxel", &num_samples.z());
  add_key("Y number of samples to take per voxel", &num_samples.y());
  add_key("X number of samples to take per voxel", &num_samples.x());

  add_key("image duration (sec)", &image_duration);
  add_key("image relative start time (sec)", &rel_start_time);

  add_parsing_key("shape type", &current_shape_ptr);
  add_key("value", &current_value);
  add_key("next shape", KeyArgument::NONE, (KeywordProcessor)&GenerateImage::increment_current_shape_num);
  add_stop_key("END");
}

bool
GenerateImage::post_processing() {
  assert(values.size() == shape_ptrs.size());

  if (patient_orientation_index < 0 || patient_rotation_index < 0)
    return true;
  // warning: relies on index taking same values as enums in PatientPosition
  exam_info_sptr->patient_position.set_rotation(static_cast<PatientPosition::RotationValue>(patient_rotation_index));
  exam_info_sptr->patient_position.set_orientation(static_cast<PatientPosition::OrientationValue>(patient_orientation_index));

  if (!is_null_ptr(current_shape_ptr)) {
    shape_ptrs.push_back(current_shape_ptr);
    values.push_back(current_value);
  }
  if (output_filename.size() == 0) {
    warning("You have to specify an output_filename\n");
    return true;
  }
  if (is_null_ptr(output_file_format_sptr)) {
    warning("You have specified an invalid output file format\n");
    return true;
  }
  if (output_image_size_x <= 0) {
    warning("X output_image_size should be strictly positive\n");
    return true;
  }
  if (output_image_size_y <= 0) {
    warning("Y output_image_size should be strictly positive\n");
    return true;
  }
  if (output_image_size_z <= 0) {
    warning("Z output_image_size should be strictly positive\n");
    return true;
  }
  if (output_voxel_size_x <= 0) {
    warning("X output_voxel_size should be strictly positive\n");
    return true;
  }
  if (output_voxel_size_y <= 0) {
    warning("Y output_voxel_size should be strictly positive\n");
    return true;
  }
  if (output_voxel_size_z <= 0) {
    warning("Z output_voxel_size should be strictly positive\n");
    return true;
  }
  if (num_samples.z() <= 0) {
    warning("number of samples to take in z-direction should be strictly positive\n");
    return true;
  }
  if (num_samples.y() <= 0) {
    warning("number of samples to take in y-direction should be strictly positive\n");
    return true;
  }
  if (num_samples.x() <= 0) {
    warning("number of samples to take in x-direction should be strictly positive\n");
    return true;
  }
  return false;
}

//! \brief parses parameters
/*! \warning Currently does not support interactive input, due to
    the use of the 'next shape' keyword.
*/
GenerateImage::GenerateImage(const char* const par_filename) {
  set_defaults();
  initialise_keymap();
  if (par_filename != 0) {
    if (parse(par_filename) == false)
      exit(EXIT_FAILURE);
  } else
    ask_parameters();

#if 0
  // doesn't work due to problem  Shape3D::parameter_info being ambiguous
  for (vector<shared_ptr<Shape3D> >::const_iterator iter = shape_ptrs.begin();
       iter != shape_ptrs.end();
       ++iter)
    {
      std::cerr << (**iter).parameter_info() << '\n';
    }
#endif
}

Succeeded
GenerateImage::compute() {
#if 0
  shared_ptr<DiscretisedDensity<3,float> > density_ptr(
						       read_from_file<DiscretisedDensity<3,float> >(template_filename));
  shared_ptr<DiscretisedDensity<3,float> > out_density_ptr = 
    density_ptr->clone();
  out_density_ptr->fill(0);
  VoxelsOnCartesianGrid<float> current_image =
    dynamic_cast<VoxelsOnCartesianGrid<float>& >(*density_ptr);

#else

  if (image_duration > 0.0) {
    std::vector<double> start_times(1, rel_start_time);
    std::vector<double> durations(1, image_duration);
    TimeFrameDefinitions frame_defs(start_times, durations);
    exam_info_sptr->set_time_frame_definitions(frame_defs);
  } else {
    warning("image duration not set, so time frame definitions will not be initialised");
  }
  VoxelsOnCartesianGrid<float> current_image(
      exam_info_sptr,
      IndexRange3D(0, output_image_size_z - 1, -(output_image_size_y / 2), -(output_image_size_y / 2) + output_image_size_y - 1,
                   -(output_image_size_x / 2), -(output_image_size_x / 2) + output_image_size_x - 1),
      CartesianCoordinate3D<float>(0, 0, 0),
      CartesianCoordinate3D<float>(output_voxel_size_z, output_voxel_size_y, output_voxel_size_x));
  shared_ptr<DiscretisedDensity<3, float>> out_density_ptr(current_image.clone());
#endif
  std::vector<float>::const_iterator value_iter = values.begin();
  for (std::vector<shared_ptr<Shape3D>>::const_iterator iter = shape_ptrs.begin(); iter != shape_ptrs.end();
       ++iter, ++value_iter) {
    std::cerr << "Next shape\n"; //(**iter).parameter_info() << '\n';
    current_image.fill(0);
    (**iter).construct_volume(current_image, num_samples);
    current_image *= *value_iter;
    *out_density_ptr += current_image;
  }
  return output_file_format_sptr->write_to_file(output_filename, *out_density_ptr);
}

END_NAMESPACE_STIR

/************************ main ************************/

USING_NAMESPACE_STIR
int
main(int argc, char* argv[]) {

  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " par_file\n";
    exit(EXIT_FAILURE);
  }
  GenerateImage application(argc == 2 ? argv[1] : 0);
  Succeeded success = application.compute();

  return success == Succeeded::yes ? EXIT_SUCCESS : EXIT_FAILURE;
}
