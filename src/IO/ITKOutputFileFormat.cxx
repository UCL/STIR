/*
    Copyright (C) 2013, Institute for Bioengineering of Catalonia
    Copyright (C) 2013, University College London
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
*/
#include "stir/VoxelsOnCartesianGrid.h"

#include "stir/IO/ITKOutputFileFormat.h"
#include "stir/utilities.h"
#include "stir/Succeeded.h"
#include "itkImage.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionIterator.h"

START_NAMESPACE_STIR


const char * const 
ITKOutputFileFormat::registered_name = "ITK";

ITKOutputFileFormat::
ITKOutputFileFormat(const NumericType& type, 
                   const ByteOrder& byte_order) 
{
  this->set_defaults();
  this->set_type_of_numbers(type);
  this->set_byte_order(byte_order);
}

void 
ITKOutputFileFormat::
set_defaults()
{
  base_type::set_defaults();
  this->default_extension = ".nhdr";
}

void 
ITKOutputFileFormat::
initialise_keymap()
{
  parser.add_start_key("ITK Output File Format Parameters");
  parser.add_key("default extension", &this->default_extension);
  parser.add_stop_key("End ITK Output File Format Parameters");
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
actual_write_to_file(std::string& filename, 
                     const DiscretisedDensity<3,float>& density) const
{
#if 0
  TODO use:
                              this->type_of_numbers, this->scale_to_write_data,
                              this->file_byte_order);
#endif
  try
    {
      add_extension(filename, this->default_extension);

      const VoxelsOnCartesianGrid<float> image_ptr =
        dynamic_cast<const VoxelsOnCartesianGrid<float>& >(density);
      CartesianCoordinate3D<int> min_indices;
      CartesianCoordinate3D<int> max_indices;
      if (!density.get_regular_range(min_indices, max_indices))
	{
	  warning("ITK writer: can handle only regular index ranges.");
	  return Succeeded::no;
	}
	
      typedef itk::Image< float, 3> ImageType;
      typedef itk::ImageFileWriter<ImageType> WriterType;
      WriterType::Pointer writer = WriterType::New();

      //Creating the image
      ImageType::Pointer image = ImageType::New();
      
      // use 0 start indices in ITK
      ImageType::IndexType start;
      start[0] = 0; // first index on X
      start[1] = 0; // first index on Y
      start[2] = 0; // first index on Z

      // find ITK origin (i.e. coordinates of first voxel)
      ImageType::PointType origin;
      CartesianCoordinate3D<float> stir_offset = density.get_physical_coordinates_for_indices(min_indices);
      origin[0] = stir_offset.x();
      origin[1] = stir_offset.y();
      // Note: need to use - for z-coordinate because of different axis conventions
      origin[2] = -stir_offset.z();
      
      // find ITK size
      ImageType::SizeType size;
      size[0] = image_ptr.get_x_size(); // size along X
      size[1] = image_ptr.get_y_size(); // size along Y
      size[2] = image_ptr.get_z_size(); // size along Z

      // find ITK voxel size
      ImageType::SpacingType spacing;
      spacing[0] = image_ptr.get_voxel_size().x(); // size along X
      spacing[1] = image_ptr.get_voxel_size().y(); // size along Y
      spacing[2] = image_ptr.get_voxel_size().z(); // size along Z

      // ITK orientation (RAI)
      ImageType::DirectionType matrix;
      matrix.Fill(0.F);
      matrix(0,0) = 1.F;
      matrix(1,1) = 1.F;
      matrix(2,2) = -1.F;

      ImageType::RegionType region;
      region.SetSize( size );
      region.SetIndex( start );

      image->SetSpacing(spacing);
      image->SetRegions( region );
      image->SetOrigin(origin);
      image->SetDirection( matrix );
      image->Allocate();
	
      // copy data
      typedef itk::ImageRegionIterator< ImageType >	IteratorType;
      IteratorType it (image, image->GetLargestPossibleRegion() );	
      DiscretisedDensity<3,float>::const_full_iterator stir_iter = density.begin_all_const();
      for ( it = it.Begin(); !it.IsAtEnd(); ++it, ++stir_iter  ){

        it.Set(*stir_iter);
      }

      // write it!
      writer->SetInput(image);
      writer->SetFileName(filename);
      writer->Update();

      return Succeeded::yes;
    }
  catch (...)
    {
      return Succeeded::no;
    }

}

END_NAMESPACE_STIR


