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
  \brief Declaration of class stir::ITKInputFileFormat

  \author Berta Marti Fuster

*/

#include "stir/IO/ITKImageInputFileFormat.h"
#include "stir/IO/InputFileFormat.h"
#include "stir/DiscretisedDensity.h"

#include "stir/IO/interfile.h"

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
ITKImageInputFileFormat::can_read(const FileSignature& signature,
                                  const std::string& filename) const
{
  typedef itk::ImageFileReader<FinalImageType> ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName(filename);
  try 
    { 
      reader->Update(); 
    } 
  catch( itk::ExceptionObject & err ) 
    { 
		
      return false;
    } 
}
 
   std::auto_ptr< DiscretisedDensity<3,float> >
	ITKImageInputFileFormat::read_from_file(std::istream& input) const
  {
    return
      std::auto_ptr<DiscretisedDensity<3,float> >
      (read_interfile_image(input));
  }
   std::auto_ptr< DiscretisedDensity<3,float> >
   ITKImageInputFileFormat::read_from_file(const std::string& filename) const
   {
     typedef itk::ImageIOBase::IOComponentType ScalarPixelType;

     itk::ImageIOBase::Pointer imageIO =
       itk::ImageIOFactory::CreateImageIO(filename.c_str(),
                                          itk::ImageIOFactory::ReadMode);

     //Set the inputFileName
     imageIO->SetFileName(filename);
     //Read the information using ImageIOFactory
     imageIO->ReadImageInformation();

     //Get the pixel Type to create a itkImage
     const ScalarPixelType pixelType = imageIO->GetComponentType();
     //Get the Dimension of the image to create a itkImage
     const size_t numDimensions =  imageIO->GetNumberOfDimensions();
	   
     switch ( pixelType )
       {
       case 1: //Unsigned char
         {
           typedef itk::Image< unsigned char, 3> ImageType;
           return
             std::auto_ptr<DiscretisedDensity<3,float> >
             (read_file_itk< ImageType >(filename));
         }

       case 2: //Char
         {
           typedef itk::Image< char, 3> ImageType;
           return
             std::auto_ptr<DiscretisedDensity<3,float> >
             (read_file_itk< ImageType >(filename));
         }

       case 3: //Unsigned short
         {
           typedef itk::Image< unsigned short, 3> ImageType;
           return
             std::auto_ptr<DiscretisedDensity<3,float> >
             (read_file_itk< ImageType >(filename));
         }
       case 4: // Short
         {
           typedef itk::Image< short, 3> ImageType;
           return
             std::auto_ptr<DiscretisedDensity<3,float> >
             (read_file_itk< ImageType >(filename));
         }
       case 5: //Unsigned Integer
         {
           typedef itk::Image< unsigned int, 3> ImageType;
           return
             std::auto_ptr<DiscretisedDensity<3,float> >
             (read_file_itk< ImageType >(filename));
         }

       case 6: //Integer
         {
           typedef itk::Image< int, 3> ImageType;
           return
             std::auto_ptr<DiscretisedDensity<3,float> >
             (read_file_itk< ImageType >(filename));
         }

       case 7: //Unsigned long
         {
           typedef itk::Image< unsigned long, 3> ImageType;
           return
             std::auto_ptr<DiscretisedDensity<3,float> >
             (read_file_itk< ImageType >(filename));
         }

       case 8: //Long
         {
           typedef itk::Image< long, 3> ImageType;
           return
             std::auto_ptr<DiscretisedDensity<3,float> >
             (read_file_itk< ImageType >(filename));
         }

       case 9: //Float
         {
           typedef itk::Image< float, 3> ImageType;
           return
             std::auto_ptr<DiscretisedDensity<3,float> >
             (read_file_itk< ImageType >(filename));
         }

       case 10:
         {
           typedef itk::Image< double, 3> ImageType;
           return
             std::auto_ptr<DiscretisedDensity<3,float> >
             (read_file_itk< ImageType >(filename));
         }
       }
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
  CastType::Pointer castFilter = CastType::New();
  castFilter->SetInput(reader->GetOutput());
  castFilter->Update();

  CartesianCoordinate3D<float> voxel_size(static_cast<float>(castFilter->GetOutput()->GetSpacing()[2]), 
                                          static_cast<float>(castFilter->GetOutput()->GetSpacing()[1]), 
                                          static_cast<float>(castFilter->GetOutput()->GetSpacing()[0]));

  const int z_size =  castFilter->GetOutput()->GetLargestPossibleRegion().GetSize()[2];
  const int y_size =  castFilter->GetOutput()->GetLargestPossibleRegion().GetSize()[1];
  const int x_size =  castFilter->GetOutput()->GetLargestPossibleRegion().GetSize()[0];
  const BasicCoordinate<3,int> min_indices = 
    make_coordinate(0, -y_size/2, -x_size/2);
  const BasicCoordinate<3,int> max_indices = 
    min_indices + make_coordinate(z_size, y_size, x_size) - 1;

  CartesianCoordinate3D<float> origin(0,0,0);
  //if (hdr.first_pixel_offsets[2] != InterfileHeader::double_value_not_set)
  //{
  //	// make sure that origin is such that 
  //	// first_pixel_offsets =  min_indices*voxel_size + origin
  //	origin =
  //		make_coordinate(float(hdr.first_pixel_offsets[2]),
  //		float(hdr.first_pixel_offsets[1]),
  //		float(hdr.first_pixel_offsets[0]))
  //		- voxel_size * BasicCoordinate<3,float>(min_indices);
  //	// TODO remove
  //	if (norm(origin)>.01)
  //		warning("interfile parsing: setting origin to (z=%g,y=%g,x=%g)",
  //		origin.z(), origin.y(), origin.x());
  //}

  VoxelsOnCartesianGrid<float>* image_ptr =
    new VoxelsOnCartesianGrid<float>
    (IndexRange<3>(min_indices, max_indices),
     origin,
     voxel_size);
  VoxelsOnCartesianGrid<float>::full_iterator stir_iter = image_ptr->begin_all();

  typedef itk::ImageRegionIterator< FinalImageType >	IteratorType;
  IteratorType it (castFilter->GetOutput(), castFilter->GetOutput()->GetLargestPossibleRegion() );
	
  for ( it = it.Begin(); !it.IsAtEnd(); ++it, ++stir_iter  ){

    *stir_iter = it.Get();
  }

  return image_ptr;
}



END_NAMESPACE_STIR

