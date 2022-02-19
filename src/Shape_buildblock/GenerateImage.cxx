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
*/


#include "stir/Shape/GenerateImage.h"
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

void GenerateImage::
increment_current_shape_num()
{
  if (!is_null_ptr( current_shape_ptr))
  {
    shape_ptrs.push_back(current_shape_ptr);
    values.push_back(current_value);
    current_shape_ptr.reset();
  }
}

void
GenerateImage::
set_defaults()
{
  exam_info_sptr.reset(new ExamInfo);
  // need to default to PET for backwards compatibility
  exam_info_sptr->imaging_modality = ImagingModality::PT;
  patient_orientation_index = 3; //unknown
  patient_rotation_index = 5; //unknown
  output_image_size_x=128;
  output_image_size_y=128;
  output_image_size_z=1;
  output_voxel_size_x=1;
  output_voxel_size_y=1;
  output_voxel_size_z=1;
  num_samples = CartesianCoordinate3D<int>(5,5,5);
  shape_ptrs.resize(0);
  values.resize(0);
  image_duration = -1.0;
  rel_start_time = 0;
  output_filename.resize(0);
  output_file_format_sptr =
          OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr();
}

void GenerateImage::set_imaging_modality()
{
  set_variable();
  this->exam_info_sptr->imaging_modality = ImagingModality(imaging_modality_as_string);
}

void
GenerateImage::
initialise_keymap()
{
  add_start_key("generate_image Parameters");
  // copy of InterfileHeader (TODO)
  add_key("imaging modality",
          KeyArgument::ASCII, (KeywordProcessor)&GenerateImage::set_imaging_modality,
          &imaging_modality_as_string);
  add_key("originating system", &exam_info_sptr->originating_system);
  add_key("patient orientation",
          &patient_orientation_index,
          &patient_orientation_values);
  add_key("patient rotation",
          &patient_rotation_index,
          &patient_rotation_values);
  patient_orientation_values.push_back("head_in");
  patient_orientation_values.push_back("feet_in");
  patient_orientation_values.push_back("other");
  patient_orientation_values.push_back("unknown"); //default

  patient_rotation_values.push_back("supine");
  patient_rotation_values.push_back("prone");
  patient_rotation_values.push_back("right");
  patient_rotation_values.push_back("left");
  patient_rotation_values.push_back("other");
  patient_rotation_values.push_back("unknown"); //default

  add_key("output filename",&output_filename);
  add_parsing_key("output file format type",&output_file_format_sptr);
  add_key("X output image size (in pixels)",&output_image_size_x);
  add_key("Y output image size (in pixels)",&output_image_size_y);
  add_key("Z output image size (in pixels)",&output_image_size_z);
  add_key("X voxel size (in mm)",&output_voxel_size_x);
  add_key("Y voxel size (in mm)",&output_voxel_size_y);
  add_key("Z voxel size (in mm)",&output_voxel_size_z);

  add_key("Z number of samples to take per voxel", &num_samples.z());
  add_key("Y number of samples to take per voxel", &num_samples.y());
  add_key("X number of samples to take per voxel", &num_samples.x());

  add_key("image duration (sec)", &image_duration);
  add_key("image relative start time (sec)", &rel_start_time);

  add_parsing_key("shape type", &current_shape_ptr);
  add_key("value", &current_value);
  add_key("next shape", KeyArgument::NONE,
          (KeywordProcessor)&GenerateImage::increment_current_shape_num);
  add_stop_key("END");

}


bool
GenerateImage::
post_processing()
{
  assert(values.size() == shape_ptrs.size());

  if (patient_orientation_index<0 || patient_rotation_index<0)
    return true;
  // warning: relies on index taking same values as enums in PatientPosition
  exam_info_sptr->patient_position.set_rotation(static_cast<PatientPosition::RotationValue>(patient_rotation_index));
  exam_info_sptr->patient_position.set_orientation(static_cast<PatientPosition::OrientationValue>(patient_orientation_index));

  if (!is_null_ptr( current_shape_ptr))
  {
    shape_ptrs.push_back(current_shape_ptr);
    values.push_back(current_value);
  }
  if (output_filename.size()==0)
  {
    warning("You have to specify an output_filename\n");
    return true;
  }
  if (is_null_ptr(output_file_format_sptr))
  {
    warning("You have specified an invalid output file format\n");
    return true;
  }
  if (output_image_size_x<=0)
  {
    warning("X output_image_size should be strictly positive\n");
    return true;
  }
  if (output_image_size_y<=0)
  {
    warning("Y output_image_size should be strictly positive\n");
    return true;
  }
  if (output_image_size_z<=0)
  {
    warning("Z output_image_size should be strictly positive\n");
    return true;
  }
  if (output_voxel_size_x<=0)
  {
    warning("X output_voxel_size should be strictly positive\n");
    return true;
  }
  if (output_voxel_size_y<=0)
  {
    warning("Y output_voxel_size should be strictly positive\n");
    return true;
  }
  if (output_voxel_size_z<=0)
  {
    warning("Z output_voxel_size should be strictly positive\n");
    return true;
  }
  if (num_samples.z()<=0)
  {
    warning("number of samples to take in z-direction should be strictly positive\n");
    return true;
  }
  if (num_samples.y()<=0)
  {
    warning("number of samples to take in y-direction should be strictly positive\n");
    return true;
  }
  if (num_samples.x()<=0)
  {
    warning("number of samples to take in x-direction should be strictly positive\n");
    return true;
  }
  return false;
}

//! \brief parses parameters
/*! \warning Currently does not support interactive input, due to
    the use of the 'next shape' keyword.
*/
GenerateImage::
GenerateImage(const char * const par_filename)
{
  set_defaults();
  initialise_keymap();
  if (par_filename!=0)
  {
    if (parse(par_filename) == false)
      error("Failed to parse par_filename");
  }
  else
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
GenerateImage::
compute()
{
#if 0
  shared_ptr<DiscretisedDensity<3,float> > density_ptr(
						       read_from_file<DiscretisedDensity<3,float> >(template_filename));
  this->out_density_ptr =
    density_ptr->clone();
  out_density_ptr->fill(0);
  VoxelsOnCartesianGrid<float> current_image =
    dynamic_cast<VoxelsOnCartesianGrid<float>& >(*density_ptr);

#else

  if (image_duration>0.0)
  {
    std::vector<double> start_times(1, rel_start_time);
    std::vector<double> durations(1, image_duration);
    TimeFrameDefinitions frame_defs(start_times, durations);
    exam_info_sptr->set_time_frame_definitions(frame_defs);
  }
  else
  {
    warning("image duration not set, so time frame definitions will not be initialised");
  }
  VoxelsOnCartesianGrid<float>
          current_image(exam_info_sptr,
                        IndexRange3D(0,output_image_size_z-1,
                                     -(output_image_size_y/2),
                                     -(output_image_size_y/2)+output_image_size_y-1,
                                     -(output_image_size_x/2),
                                     -(output_image_size_x/2)+output_image_size_x-1),
                        CartesianCoordinate3D<float>(0,0,0),
                        CartesianCoordinate3D<float>(output_voxel_size_z,
                                                     output_voxel_size_y,
                                                     output_voxel_size_x));
  this->out_density_ptr.reset(current_image.clone());
#endif
  std::vector<float >::const_iterator value_iter = values.begin();
  for (std::vector<shared_ptr<Shape3D> >::const_iterator iter = shape_ptrs.begin();
          iter != shape_ptrs.end();
  ++iter, ++value_iter)
  {
    info("Processing next shape...", 2);
    current_image.fill(0);
    (**iter).construct_volume(current_image, num_samples);
    current_image *= *value_iter;
    *out_density_ptr += current_image;
  }
  return Succeeded::yes;
}

Succeeded
GenerateImage::
save_image()
{
  output_file_format_sptr->write_to_file(output_filename, *out_density_ptr);
  return Succeeded::yes;
}

shared_ptr<DiscretisedDensity<3, float>>
GenerateImage::
get_output_sptr()
{
  return out_density_ptr;
}

END_NAMESPACE_STIR
