/*
    Copyright (C) 2013, Institute for Bioengineering of Catalonia
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
  \ingroup IO
  \brief Implementation of class stir::ITKOutputFileFormat

  \author Berta Marti Fuster
  \author Kris Thielemans

  $Date$
  $Revision$
*/
#include "stir/VoxelsOnCartesianGrid.h"

#include "stir/IO/ITKOutputFileFormat.h"
#include "stir/IO/interfile.h"
#include "itkImage.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionIterator.h"

START_NAMESPACE_STIR


const char * const 
ITKOutputFileFormat::registered_name = "Nifti";

ITKOutputFileFormat::
ITKOutputFileFormat(const NumericType& type, 
                   const ByteOrder& byte_order) 
{
  base_type::set_defaults();
  set_type_of_numbers(type);
  set_byte_order(byte_order);
}

void 
ITKOutputFileFormat::
set_defaults()
{
  base_type::set_defaults();
}

void 
ITKOutputFileFormat::
initialise_keymap()
{
  parser.add_start_key("Nifti Output File Format Parameters");
  parser.add_stop_key("End Nifti Output File Format Parameters");
  base_type::initialise_keymap();
}

bool 
ITKOutputFileFormat::
post_processing()
{
  if (base_type::post_processing())
    return true;
  return false;
}

// note 'warn' commented below to avoid compiler warning message about unused variables
ByteOrder 
ITKOutputFileFormat::
set_byte_order(const ByteOrder& new_byte_order, const bool /* warn */) 
{
  this->file_byte_order = new_byte_order;
  return this->file_byte_order;
}



Succeeded  
ITKOutputFileFormat::
actual_write_to_file(string& filename, 
                     const DiscretisedDensity<3,float>& density) const
{
  const VoxelsOnCartesianGrid<float> image_ptr =
    dynamic_cast<const VoxelsOnCartesianGrid<float>& >(density);
	
  // TODO modify write_basic_interfile to return filename
  typedef itk::Image< float, 3> ImageType;
  typedef itk::ImageFileWriter<ImageType> WriterType;
  WriterType::Pointer writer = WriterType::New();

  //Creating the image
  ImageType::Pointer image = ImageType::New();
  ImageType::IndexType start;
  start[0] = 0; // first index on X
  start[1] = 0; // first index on Y
  start[2] = 0; // first index on Z

  ImageType::SizeType size;
  size[0] = image_ptr.get_x_size(); // size along X
  size[1] = image_ptr.get_y_size(); // size along Y
  size[2] = image_ptr.get_z_size(); // size along Z

  ImageType::SpacingType spacing;
  spacing[0] = image_ptr.get_voxel_size().x(); // size along X
  spacing[1] = image_ptr.get_voxel_size().y(); // size along Y
  spacing[2] = image_ptr.get_voxel_size().z(); // size along Z

  ImageType::RegionType region;
  region.SetSize( size );
  region.SetIndex( start );
	
  image->SetSpacing(spacing);
  image->SetRegions( region );
  image->Allocate();
	
  float * image_raw = new float [ size[0]*size[1]*size[2] ];
  std::copy(image_ptr.begin_all(),image_ptr.end_all(), image_raw);

  typedef itk::ImageRegionIterator< ImageType >	IteratorType;
  IteratorType it (image, image->GetLargestPossibleRegion() );
	
  int i = 0;
  for ( it = it.Begin(); !it.IsAtEnd(); ++it, ++i  ){

    it.Set(image_raw[i]);
  }

  writer->SetInput(image);
  writer->SetFileName(filename);
  writer->Update();
  
  Succeeded success =
    write_basic_interfile(filename, density, 
			  this->type_of_numbers, this->scale_to_write_data,
			  this->file_byte_order);
  if (success == Succeeded::yes)
    replace_extension(filename, ".hv");
  return success;
};

END_NAMESPACE_STIR


