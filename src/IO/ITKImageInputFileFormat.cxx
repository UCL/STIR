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
  \brief Declaration of class stir::ITKImageInputFileFormat

  \author Berta Marti Fuster
  \author Kris Thielemans
*/

#include "stir/IO/ITKImageInputFileFormat.h"
#include "stir/DiscretisedDensity.h"

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkCastImageFilter.h"
#include "itkImageRegionIterator.h"

START_NAMESPACE_STIR

//! Class for reading images in ITK file-format.
/*! \ingroup IO
    \preliminary

*/

typedef itk::Image<float, 3>		FinalImageType;
  
bool 
ITKImageInputFileFormat::actual_can_read(const FileSignature& signature,
                                         std::istream& input) const
{
  return false;
}

bool
ITKImageInputFileFormat::can_read(const FileSignature& signature,
                                  std::istream& input) const
{
  return this->actual_can_read(signature, input);
}

bool 
ITKImageInputFileFormat::can_read(const FileSignature& /*signature*/,
                                  const std::string& filename) const
{
  typedef itk::ImageFileReader<FinalImageType> ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName(filename);
  try 
    { 
      reader->Update(); 
      return true;
    } 
  catch( itk::ExceptionObject & err ) 
    { 
		
      return false;
    } 
}
 
std::auto_ptr< DiscretisedDensity<3,float> >
ITKImageInputFileFormat::read_from_file(std::istream& input) const
{
  error("read_from_file for ITK with istream not implemented %s:%d. Sorry",
        __FILE__, __LINE__);
  return
    std::auto_ptr<DiscretisedDensity<3,float> >
    (0);
}

std::auto_ptr< DiscretisedDensity<3,float> >
ITKImageInputFileFormat::read_from_file(const std::string& filename) const
{
  typedef itk::Image< float, 3> ImageType;
  return
    std::auto_ptr<DiscretisedDensity<3,float> >
    (read_file_itk< ImageType >(filename));
}

//To read any file format
template<typename TImageType> VoxelsOnCartesianGrid<float>*
ITKImageInputFileFormat::read_file_itk(std::string filename) const
{
  typedef itk::ImageFileReader<TImageType> ReaderType;
  typename ReaderType::Pointer reader = ReaderType::New();

  reader->SetFileName(filename);
  reader->Update();

  typedef itk::CastImageFilter<TImageType, FinalImageType> CastType;
  typename CastType::Pointer castFilter = CastType::New();
  castFilter->SetInput(reader->GetOutput());
  castFilter->Update();

  // find voxel size
  CartesianCoordinate3D<float> voxel_size(static_cast<float>(castFilter->GetOutput()->GetSpacing()[2]), 
                                          static_cast<float>(castFilter->GetOutput()->GetSpacing()[1]), 
                                          static_cast<float>(castFilter->GetOutput()->GetSpacing()[0]));

  // find index range in usual STIR convention
  const int z_size =  castFilter->GetOutput()->GetLargestPossibleRegion().GetSize()[2];
  const int y_size =  castFilter->GetOutput()->GetLargestPossibleRegion().GetSize()[1];
  const int x_size =  castFilter->GetOutput()->GetLargestPossibleRegion().GetSize()[0];
  const BasicCoordinate<3,int> min_indices = 
    make_coordinate(0, -y_size/2, -x_size/2);
  const BasicCoordinate<3,int> max_indices = 
    min_indices + make_coordinate(z_size, y_size, x_size) - 1;

  // find STIR origin
  CartesianCoordinate3D<float> origin(static_cast<float>(castFilter->GetOutput()->GetOrigin()[2]), 
				      static_cast<float>(castFilter->GetOutput()->GetOrigin()[1]), 
				      static_cast<float>(castFilter->GetOutput()->GetOrigin()[0]));
  {
    // make sure that origin is such that 
    // first_pixel_offsets =  min_indices*voxel_size + origin
    origin -= voxel_size * BasicCoordinate<3,float>(min_indices);
  }

  // create STIR image
  VoxelsOnCartesianGrid<float>* image_ptr =
    new VoxelsOnCartesianGrid<float>
    (IndexRange<3>(min_indices, max_indices),
     origin,
     voxel_size);
  VoxelsOnCartesianGrid<float>::full_iterator stir_iter = image_ptr->begin_all();

  // copy data
  typedef itk::ImageRegionIterator< FinalImageType >	IteratorType;
  IteratorType it (castFilter->GetOutput(), castFilter->GetOutput()->GetLargestPossibleRegion() );
	
  for ( it = it.Begin(); !it.IsAtEnd(); ++it, ++stir_iter  ){

    *stir_iter = it.Get();
  }

  return image_ptr;
}



END_NAMESPACE_STIR

