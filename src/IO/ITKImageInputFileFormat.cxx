/*
    Copyright (C) 2013, Institute for Bioengineering of Catalonia
    Copyright (C) 2013-2014, University College London

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
#include "stir/utilities.h"

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageRegionIterator.h"
#include "itkGDCMImageIO.h"
#include "itkGDCMSeriesFileNames.h"
#include "itkImageSeriesReader.h"

START_NAMESPACE_STIR

//! Class for reading images in ITK file-format.
/*! \ingroup IO
    \preliminary

*/

typedef itk::Image<float, 3>		FinalImageType;

template<typename ImageTypePtr>
VoxelsOnCartesianGrid<float>*
convert_ITK_to_STIR(const ImageTypePtr itk_image);

template<typename TImageType> VoxelsOnCartesianGrid<float>*
read_file_itk(std::string filename);
  
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
  catch( itk::ExceptionObject & /*err*/ ) 
    { 
		
      return false;
    } 
}

unique_ptr< DiscretisedDensity<3,float> >
ITKImageInputFileFormat::read_from_file(std::istream& input) const
{
  error("read_from_file for ITK with istream not implemented %s:%d. Sorry",
        __FILE__, __LINE__);
  return
    unique_ptr<DiscretisedDensity<3,float> >();
}

unique_ptr< DiscretisedDensity<3,float> >
ITKImageInputFileFormat::read_from_file(const std::string& filename) const
{
  return
    unique_ptr<DiscretisedDensity<3,float> >
    (read_file_itk< FinalImageType >(filename));
}


//To read any file format via ITK
template<typename TImageType> VoxelsOnCartesianGrid<float>*
read_file_itk(std::string filename)
{
  typedef itk::GDCMImageIO       ImageIOType;
  ImageIOType::Pointer dicomIO = ImageIOType::New();
  try
    {
      if (!dicomIO->CanReadFile(filename.c_str()))
        {
          // Not a DICOM file, so we just read a single image
          typedef itk::ImageFileReader<TImageType> ReaderType;
          typename ReaderType::Pointer reader = ReaderType::New();

          reader->SetFileName(filename);
          reader->Update();
          typename TImageType::Pointer itk_image = reader->GetOutput();
          return convert_ITK_to_STIR(itk_image);
        }
      else
        {
          // It's a DICOM file (I hope).

          // For this, we need to read all slices in a series.
          // We use code from ITK's Examples/IO/DicomSeriesReadImageWrite2.cxx
          // to do this.
          // However, we change it to read the series which contains the filename that was passed.
          
          // find all series in the directory
          // This is by default based on unique
          // \item[0020 0011] Series Number
          // \item[0018 0024] Sequence Name
          // \item[0018 0050] Slice Thickness
          // \item[0028 0010] Rows
          // \item[0028 0011] Columns
          typedef itk::GDCMSeriesFileNames NamesGeneratorType;
          typename NamesGeneratorType::Pointer nameGenerator = NamesGeneratorType::New();
          nameGenerator->SetUseSeriesDetails( true );
          // Make sure we read only data from a single frame and gate
          nameGenerator->AddSeriesRestriction("0008|0022" ); // AcquisitionDate
          nameGenerator->AddSeriesRestriction("0008|0032" ); // AcquisitionTime
          nameGenerator->AddSeriesRestriction("0018|1060" ); // TriggerTime
          nameGenerator->AddSeriesRestriction("0018|1063" ); // FrameTime

          const std::string dir_name = get_directory_name(filename);
          nameGenerator->SetDirectory( dir_name.c_str() );
          typedef std::vector< std::string >    SeriesIdContainer;
          const SeriesIdContainer & seriesUIDs = nameGenerator->GetSeriesUIDs();

          // We've found all "series" (i.e. different data-sets according to above restrictions). Now see which one we should read.
          // We do this by checking which one contains the original filename.
          typedef std::vector< std::string >   FileNamesContainer;
          FileNamesContainer fileNames;
          // Loop through all "series"
          for (SeriesIdContainer::const_iterator iter=seriesUIDs.begin(); iter!= seriesUIDs.end(); ++iter)
            {
              fileNames = nameGenerator->GetFileNames( iter->c_str() );
              // check if filename is present
              if (std::find(fileNames.begin(), fileNames.end(), filename) != fileNames.end())
                break; // yes, get out of series-loop
            }
          
          // ok. we know which filenames are in the same "series", so let's read them
          typedef itk::ImageSeriesReader< TImageType > ReaderType;
          typename ReaderType::Pointer reader = ReaderType::New();

          reader->SetImageIO( dicomIO );
          reader->SetFileNames( fileNames );
          reader->Update();
          typename TImageType::Pointer itk_image = reader->GetOutput();

          // Finally, convert to STIR!
          return convert_ITK_to_STIR(itk_image);
     
        }
    }
  catch (std::exception &ex)
    {
      error(ex.what());
      return 0;
    }

}

// Helper class to be able to work-around ITK's Pointer stuff
// We will need to find the type of the object pointed to. I don't know how to do this
// in ITK, so we do it using the usual way to get rid of pointers, but now including itk::SmartPointer
// (we might only need the latter, but I'm not sure)
template <class PtrType>
struct removePtr
{
  typedef PtrType type;
};

template <class Type>
struct removePtr<Type *>
{
  typedef Type type;
};

template <class Type>
struct removePtr<itk::SmartPointer<Type> >
{
  typedef Type type;
};

// Actual conversion function
// WARNING: COMPLETELY IGNORES ORIENTATION
template<typename ImageTypePtr>
VoxelsOnCartesianGrid<float>*
convert_ITK_to_STIR(const ImageTypePtr itk_image)
  {
  // find voxel size
  CartesianCoordinate3D<float> voxel_size(static_cast<float>(itk_image->GetSpacing()[2]), 
                                          static_cast<float>(itk_image->GetSpacing()[1]), 
                                          static_cast<float>(itk_image->GetSpacing()[0]));

  // find index range in usual STIR convention
  const int z_size =  itk_image->GetLargestPossibleRegion().GetSize()[2];
  const int y_size =  itk_image->GetLargestPossibleRegion().GetSize()[1];
  const int x_size =  itk_image->GetLargestPossibleRegion().GetSize()[0];
  const BasicCoordinate<3,int> min_indices = 
    make_coordinate(0, -y_size/2, -x_size/2);
  const BasicCoordinate<3,int> max_indices = 
    min_indices + make_coordinate(z_size, y_size, x_size) - 1;

  // find STIR origin
  CartesianCoordinate3D<float> origin(static_cast<float>(itk_image->GetOrigin()[2]), 
				      static_cast<float>(itk_image->GetOrigin()[1]), 
				      static_cast<float>(itk_image->GetOrigin()[0]));
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

  // copy data
  VoxelsOnCartesianGrid<float>::full_iterator stir_iter = image_ptr->begin_all();
  typedef typename removePtr<ImageTypePtr>::type ImageType;
  typedef itk::ImageRegionConstIterator< ImageType > IteratorType;
  IteratorType it (itk_image, itk_image->GetLargestPossibleRegion() );
  for ( it.GoToBegin(); !it.IsAtEnd(); ++it, ++stir_iter  )
    {
      *stir_iter = static_cast<float>(it.Get());
    }

  return image_ptr;
}



END_NAMESPACE_STIR

