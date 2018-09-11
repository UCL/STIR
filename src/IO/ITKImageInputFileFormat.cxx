/*
    Copyright (C) 2013, Institute for Bioengineering of Catalonia
    Copyright (C) 2013-2014, University College London
    Copyright (C) 2018, University College London

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
  \author Richard Brown
*/

#include "stir/IO/ITKImageInputFileFormat.h"
#include "stir/DiscretisedDensity.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/utilities.h"

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageRegionIterator.h"
#include "itkGDCMImageIO.h"
#include "itkGDCMSeriesFileNames.h"
#include "itkImageSeriesReader.h"
#include "itkOrientImageFilter.h"
#include "stir/info.h"

START_NAMESPACE_STIR

//! Class for reading images in ITK file-format.
/*! \ingroup IO
    \preliminary

*/

typedef itk::Image<float, 3>                                 ITKImageSingle;
typedef itk::VectorImage<float, 3>                           ITKImageMulti;
typedef DiscretisedDensity<3,float>                          STIRImageSingle;
typedef VoxelsOnCartesianGrid<CartesianCoordinate3D<float> > STIRImageMulti;

STIRImageSingle*
convert_ITK_to_STIR(const ITKImageSingle::Pointer itk_image);

STIRImageMulti*
convert_ITK_to_STIR(const ITKImageMulti::Pointer itk_image);

template<typename STIRImageType>
static
STIRImageType *
read_file_itk(const std::string &filename);
  
CartesianCoordinate3D<float>
ITK_coordinates_to_STIR(const itk::ImageBase<3>::PointType &itk_coord,
        CartesianCoordinate3D<float> &voxel_size,
        const BasicCoordinate<3,int> &min_indices,
        bool is_displacement_field = false);

template<typename ITKImageType>
void
orient_ITK_image(typename ITKImageType::Pointer &itk_image,
                 CartesianCoordinate3D<float> &voxel_size,
                 BasicCoordinate<3,int> &min_indices,
                 BasicCoordinate<3,int> &max_indices,
                 CartesianCoordinate3D<float> &origin,
                 const typename ITKImageType::Pointer itk_image_orig);

template <typename STIRImageType>
bool 
ITKImageInputFileFormat<STIRImageType>::actual_can_read(const FileSignature& signature,
                                         std::istream& input) const
{
  return false;
}

template <typename STIRImageType>
bool
ITKImageInputFileFormat<STIRImageType>::can_read(const FileSignature& signature,
                                  std::istream& input) const
{
  return this->actual_can_read(signature, input);
}

template <typename STIRImageType>
bool 
ITKImageInputFileFormat<STIRImageType>::can_read(const FileSignature& /*signature*/,
                                  const std::string& filename) const
{
  typedef itk::ImageFileReader<ITKImageSingle> ReaderType;
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

template <typename STIRImageType>
unique_ptr<STIRImageType>
ITKImageInputFileFormat<STIRImageType>::read_from_file(std::istream& input) const
{
  error("read_from_file for ITK with istream not implemented %s:%d. Sorry",
        __FILE__, __LINE__);
  return
    unique_ptr<STIRImageType>();
}

template<typename STIRImageType>
unique_ptr<STIRImageType>
ITKImageInputFileFormat<STIRImageType>::
read_from_file(const std::string& filename) const
{
  return
    unique_ptr<STIRImageType>
    (read_file_itk< STIRImageType >(filename));
}

// Actual conversion function
STIRImageSingle*
convert_ITK_to_STIR(const ITKImageSingle::Pointer itk_image_orig)
{
    ITKImageSingle::Pointer itk_image;
    CartesianCoordinate3D<float> voxel_size;
    BasicCoordinate<3,int> min_indices, max_indices;
    CartesianCoordinate3D<float> origin;

    // orientate the ITK image
    orient_ITK_image<ITKImageSingle>(
                itk_image, voxel_size, min_indices, max_indices, origin, itk_image_orig);

    // create STIR image
    VoxelsOnCartesianGrid<float>* image_ptr =
        new VoxelsOnCartesianGrid<float>
            (IndexRange<3>(min_indices, max_indices),
            origin,
            voxel_size);

    // copy data
    VoxelsOnCartesianGrid<float>::full_iterator stir_iter = image_ptr->begin_all();
    typedef itk::ImageRegionConstIterator< ITKImageSingle > IteratorType;
    IteratorType it (itk_image, itk_image->GetLargestPossibleRegion() );
    for ( it.GoToBegin(); !it.IsAtEnd(); ++it, ++stir_iter  ) {
        *stir_iter = static_cast<float>(it.Get());
    }

    return image_ptr;
}

// Actual conversion function
STIRImageMulti*
convert_ITK_to_STIR(const ITKImageMulti::Pointer itk_image_orig)
{
    ITKImageMulti::Pointer itk_image;
    CartesianCoordinate3D<float> voxel_size;
    BasicCoordinate<3,int> min_indices, max_indices;
    CartesianCoordinate3D<float> origin;

    // orientate the ITK image
    orient_ITK_image<ITKImageMulti>(
                itk_image, voxel_size, min_indices, max_indices, origin, itk_image_orig);

    // create STIR image
    STIRImageMulti* image_ptr =
        new STIRImageMulti
            (IndexRange<3>(min_indices, max_indices),
             origin,
             voxel_size);

    // copy data
    STIRImageMulti::full_iterator stir_iter = image_ptr->begin_all();
    typedef itk::ImageRegionConstIterator< ITKImageMulti > IteratorType;
    IteratorType it (itk_image, itk_image->GetLargestPossibleRegion() );
    for ( it.GoToBegin(); !it.IsAtEnd(); ++it, ++stir_iter) {
        itk::Point<double,3U> itk_coord;
        itk_coord[0] = -it.Get()[0]; // Seems we need an extra flip in the x-axis. don't know why...
        itk_coord[1] = -it.Get()[1]; // Seems we need an extra flip in the y-axis. don't know why...
        itk_coord[2] =  it.Get()[2];
        *stir_iter =
            ITK_coordinates_to_STIR(
                itk_coord,
                voxel_size,
                min_indices,
                true);
    }

    return image_ptr;
}

//To read any file format via ITK
template<>
STIRImageSingle*
read_file_itk(const std::string &filename)
{
  typedef itk::GDCMImageIO       ImageIOType;
  ImageIOType::Pointer dicomIO = ImageIOType::New();
  try
    {
      if (!dicomIO->CanReadFile(filename.c_str()))
        {
          // Not a DICOM file, so we just read a single image
          typedef itk::ImageFileReader<ITKImageSingle> ReaderType;
          typename ReaderType::Pointer reader = ReaderType::New();

          reader->SetFileName(filename);
          reader->Update();
          ITKImageSingle::Pointer itk_image = reader->GetOutput();

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
          typedef itk::ImageSeriesReader< ITKImageSingle > ReaderType;
          ReaderType::Pointer reader = ReaderType::New();

          reader->SetImageIO( dicomIO );
          reader->SetFileNames( fileNames );
          reader->Update();
          ITKImageSingle::Pointer itk_image = reader->GetOutput();

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

//To read any file format via ITK
template<>
STIRImageMulti*
read_file_itk(const std::string &filename)
{
  typedef itk::GDCMImageIO       ImageIOType;

  try
    {
      // Not a DICOM file, so we just read a single image
      typedef itk::ImageFileReader<ITKImageMulti> ReaderType;
      ReaderType::Pointer reader = ReaderType::New();
      reader->SetFileName(filename);
      reader->Update();

      if (reader->GetImageIO()->GetPixelType() != itk::ImageIOBase::VECTOR) {
          error("read_file_itk: Image type should be vector %s:%d.",
                __FILE__, __LINE__);
          return NULL; }

      ITKImageMulti::Pointer itk_image = reader->GetOutput();

      return convert_ITK_to_STIR(itk_image);

    }
  catch (std::exception &ex)
    {
      error(ex.what());
      return 0;
    }
}

template<typename ITKImageType>
void
orient_ITK_image(typename ITKImageType::Pointer &itk_image,
                 CartesianCoordinate3D<float> &voxel_size,
                 BasicCoordinate<3,int> &min_indices,
                 BasicCoordinate<3,int> &max_indices,
                 CartesianCoordinate3D<float> &origin,
                 const typename ITKImageType::Pointer itk_image_orig)
{
    // Only works for HFS!
    typedef itk::OrientImageFilter<ITKImageType,ITKImageType> OrienterType;
    typename OrienterType::Pointer orienter = OrienterType::New();
    orienter->UseImageDirectionOn();
    orienter->SetInput(itk_image_orig);
    orienter->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RAS);
    orienter->Update();
    itk_image = orienter->GetOutput();

    // find voxel size
    voxel_size = CartesianCoordinate3D<float>(static_cast<float>(itk_image->GetSpacing()[2]),
        static_cast<float>(itk_image->GetSpacing()[1]),
        static_cast<float>(itk_image->GetSpacing()[0]));

    // find index range in usual STIR convention
    const int z_size =  itk_image->GetLargestPossibleRegion().GetSize()[2];
    const int y_size =  itk_image->GetLargestPossibleRegion().GetSize()[1];
    const int x_size =  itk_image->GetLargestPossibleRegion().GetSize()[0];
    min_indices = BasicCoordinate<3,int>(make_coordinate(0, -y_size/2, -x_size/2));
    max_indices = min_indices + make_coordinate(z_size, y_size, x_size) - 1;

    // find STIR origin
    origin = CartesianCoordinate3D<float>(
                ITK_coordinates_to_STIR(
                    itk_image->GetOrigin(),
                    voxel_size,
                    min_indices));
#ifndef NDEBUG
  info(boost::format("ITK image origin: %1%") % itk_image->GetOrigin());
  info(boost::format("STIR image origin: [%1%, %2%, %3%]") % origin[1] % origin[2] % origin[3]);
#endif
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

CartesianCoordinate3D<float>
ITK_coordinates_to_STIR(
        const itk::ImageBase<3>::PointType &itk_coord,
        CartesianCoordinate3D<float> &voxel_size,
        const BasicCoordinate<3,int> &min_indices,
        bool is_displacement_field)
{
  // find STIR origin
  // Note: need to use - for z-coordinate because of different axis conventions
  // Only works for HFS!
  CartesianCoordinate3D<float> stir_coord(-static_cast<float>(itk_coord[2]),
                                           static_cast<float>(itk_coord[1]),
                                           static_cast<float>(itk_coord[0]));

  // The following is not required for displacement field images
  if (!is_displacement_field)
  {
    // make sure that origin is such that
    // first_pixel_offsets =  min_indices*voxel_size + origin
    stir_coord -= voxel_size * BasicCoordinate<3,float>(min_indices);
  }

  return stir_coord;
}

// explicit instantiations
template class ITKImageInputFileFormat<STIRImageSingle>;
template class ITKImageInputFileFormat<STIRImageMulti>;

END_NAMESPACE_STIR
