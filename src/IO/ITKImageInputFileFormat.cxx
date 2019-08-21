/*
    Copyright (C) 2013, Institute for Bioengineering of Catalonia
    Copyright (C) 2013-2014, University College London
    Copyright (C) 2018, University College London
    Copyright (C) 2018, Commonwealth Scientific and Industrial Research Organisation
                        Australian eHealth Research Centre

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
  \author Ashley Gillman
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
#include "boost/format.hpp"

START_NAMESPACE_STIR

//! Class for reading images in ITK file-format.
/*! \ingroup IO
    \preliminary

*/

typedef itk::Image<float, 3>                                 ITKImageSingle;
typedef itk::VectorImage<float, 3>                           ITKImageMulti;
typedef DiscretisedDensity<3, float>                         STIRImageSingle;
typedef VoxelsOnCartesianGrid<float>                         STIRImageSingleConcrete;
typedef VoxelsOnCartesianGrid<CartesianCoordinate3D<float> > STIRImageMulti;
typedef itk::MetaDataObject< std::string >                   MetaDataStringType;

template<typename STIRImageType>
static
STIRImageType *
read_file_itk(const std::string &filename);

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

/* Convert ITK (LPS) coordinates into STIR physical coordinates and
   accounting for the change of origin by default.

   However, if `is_relative_coordinate` is true, coordinates are
   interpreted as being displacement vectors and hence the change of
   origin is ignored.
 */
template<typename ITKPointType, typename STIRImageType>
static inline
CartesianCoordinate3D<float>
ITK_coordinates_to_STIR_physical_coordinates
(const ITKPointType &itk_coord,
 const STIRImageType &stir_image, bool is_relative_coordinate=false)
{
  // find STIR origin
  // Note: need to use - for z-coordinate because of different axis conventions
  CartesianCoordinate3D<float> stir_coord
    = stir_image.get_physical_coordinates_for_LPS_coordinates
      (CartesianCoordinate3D<float>(static_cast<float>(itk_coord[2]),
                                    static_cast<float>(itk_coord[1]),
                                    static_cast<float>(itk_coord[0])));

  // The following is not required for displacement vectors, such as a displacement field, as
  // the coordinates are relative.
  if (!is_relative_coordinate)
  {
    CartesianCoordinate3D<float> stir_origin_index(0, 0, 0);

    // assuming we previously oriented the ITK image, min_indices is the
    // index where ITK origin points to in physical space
    const CartesianCoordinate3D<float> stir_origin_wrt_itk_origin
      = stir_image.get_physical_coordinates_for_indices(stir_origin_index)
      - stir_image.get_physical_coordinates_for_indices(stir_image.get_min_indices());

    stir_coord += stir_origin_wrt_itk_origin;
  }

  return stir_coord;
}

/* Convert an ITK Pixel (i.e., float) to a STIR Pixel. */
template<typename STIRPixelType, typename ITKPixelType, typename STIRImageType>
static inline
STIRPixelType
ITK_pixel_to_STIR_pixel(ITKPixelType itk_pixel,
                        const STIRImageType &stir_image,
                        bool)
{
  return static_cast<STIRPixelType>(itk_pixel);
}

/* Specialisation if the pixel is a vector and we want a multi-image */
template<>
inline
typename STIRImageMulti::full_value_type
ITK_pixel_to_STIR_pixel(typename ITKImageMulti::PixelType itk_pixel,
                        const STIRImageMulti &stir_image,
                        bool is_displacement_field)
{
  // ITK VariableLengthVector to ITK FixedArray
  // We know it is length 3
  assert(itk_pixel.GetSize() == 3);
  // TODO: currently this is only for deformation/displacement images
  //       However, dynamic images may be other lengths.
  typename ITKImageMulti::PointType itk_coord;
  for (unsigned int i=0; i<3; ++i)
    itk_coord[i] = itk_pixel[i];
  return ITK_coordinates_to_STIR_physical_coordinates
    (itk_coord, stir_image, is_displacement_field);
}

/* Calculate the STIR index range from an ITK image. */
template<typename ITKImagePtrType>
static inline
IndexRange<3>
calc_stir_index_range(const ITKImagePtrType itk_image)
{
  // find index range in usual STIR convention
  const int z_size = itk_image->GetLargestPossibleRegion().GetSize()[2];
  const int y_size = itk_image->GetLargestPossibleRegion().GetSize()[1];
  const int x_size = itk_image->GetLargestPossibleRegion().GetSize()[0];
  const BasicCoordinate<3, int> min_indices
    = BasicCoordinate<3,int>(make_coordinate(0, -y_size/2, -x_size/2));
  const BasicCoordinate<3, int> max_indices
    = min_indices + make_coordinate(z_size, y_size, x_size) - 1;
  return IndexRange<3>(min_indices, max_indices);
}

/* Calculate the STIR origin for a given voxel_size and index_range from an ITK
   image.
 */
template<typename ITKImagePtrType>
static inline
const CartesianCoordinate3D<float>
calc_stir_origin(CartesianCoordinate3D<float> voxel_size,
                 IndexRange<3> index_range,
                 const ITKImagePtrType itk_image)
{
  const CartesianCoordinate3D<float> stir_origin_index(0, 0, 0);
  // dummy image that has minumum to be able to find ITK -> STIR origin vector
  const VoxelsOnCartesianGrid<float> dummy_image
    (index_range, stir_origin_index, voxel_size);

  return ITK_coordinates_to_STIR_physical_coordinates
    (itk_image->GetOrigin(), dummy_image);
}

/* Constructs an exam info object from an ITK meta data dictionary.
   Uses fields:
   - (0018, 5100) Patient Position
 */
static inline
shared_ptr<ExamInfo>
construct_exam_info_from_metadata_dictionary(itk::MetaDataDictionary dictionary)
{
  shared_ptr<ExamInfo> exam_info_sptr (new ExamInfo());

  // Patient Position
  PatientPosition patient_position(PatientPosition::unknown_position);
  std::string patient_position_str = "";
  const std::string patient_position_tag = "0018|5100";
  const itk::MetaDataDictionary::ConstIterator maybe_patient_position_iter
    = dictionary.Find(patient_position_tag);
  // Did we actually find it?
  if (maybe_patient_position_iter != dictionary.End()) {
    const MetaDataStringType::ConstPointer maybe_patient_position_data
      = dynamic_cast<const MetaDataStringType*>
      (maybe_patient_position_iter->second.GetPointer());
    // Is it actually a string value?
    if(maybe_patient_position_data) {
      patient_position_str = maybe_patient_position_data->GetMetaDataObjectValue();
    }
  }
  // Now patient_positon_str is empty or the value, but is it a valid value?
  // If so, update patient_position
  for (unsigned int position_idx = 0;
       (position_idx < PatientPosition::unknown_position)
         && (patient_position.get_position() == PatientPosition::unknown_position);
       ++position_idx) {
    PatientPosition possible_position
      (static_cast<PatientPosition::PositionValue>(position_idx));
    if (patient_position_str.find(possible_position.get_position_as_string())
        != std::string::npos) {
      patient_position = possible_position;
    }
  }
  // warn if we got nothing
  if (patient_position.get_position() == PatientPosition::unknown_position) {
    warning("Unable to determine patient position. "
            "Internally this will generally be handled by assuming HFS");
  }
  exam_info_sptr->patient_position = patient_position;

  return exam_info_sptr;
}

/* Constructs an empty STIR image with correct geometrical- and meta-data.
   This method expects that itk_image is already oriented to be consistent with
   STIR x, y, z axes.
 */
template<typename ITKImagePtrType, typename STIRImageType>
static inline
STIRImageType*
construct_empty_stir_image(const ITKImagePtrType itk_image,
                           shared_ptr<ExamInfo> exam_info_sptr)
{
  // find voxel size
  const CartesianCoordinate3D<float> voxel_size
    (static_cast<float>(itk_image->GetSpacing()[2]),
     static_cast<float>(itk_image->GetSpacing()[1]),
     static_cast<float>(itk_image->GetSpacing()[0]));

  // find info STIR image geometrical metadata
  const IndexRange<3> index_range = calc_stir_index_range(itk_image);
  const CartesianCoordinate3D<float> stir_origin = calc_stir_origin
    (voxel_size, index_range, itk_image);

  // create STIR image
  STIRImageType* image_ptr = new STIRImageType
    (exam_info_sptr, index_range, stir_origin, voxel_size);
  return image_ptr;
}

/* Copy a single ITK Image to a single STIR Image.
   This method expects that itk_image is already oriented to be consistent with
   STIR x, y, z axes.
*/
template<typename ITKImageType, typename STIRImageType>
static inline
void copy_ITK_data_to_STIR_image(const typename ITKImageType::Pointer itk_image,
                                 STIRImageType& stir_image,
                                 bool is_displacement_field)
{
  typename STIRImageType::full_iterator stir_iter = stir_image.begin_all();
  typedef itk::ImageRegionConstIterator<ITKImageType> IteratorType;
  IteratorType it (itk_image, itk_image->GetLargestPossibleRegion());
  for (it.GoToBegin(); !it.IsAtEnd(); ++it, ++stir_iter)
  {
    *stir_iter = ITK_pixel_to_STIR_pixel
      <typename STIRImageType::full_value_type, typename ITKImageType::PixelType, STIRImageType>
      (it.Get(), stir_image, is_displacement_field);
  }
}

template<typename ITKImageType>
typename ITKImageType::Pointer
orient_ITK_image(const typename ITKImageType::Pointer itk_image_orig,
                 const shared_ptr<ExamInfo> exam_info_sptr)
{
  typedef itk::OrientImageFilter<ITKImageType,ITKImageType> OrienterType;
  typename OrienterType::Pointer orienter = OrienterType::New();
  orienter->UseImageDirectionOn();
  orienter->SetInput(itk_image_orig);

  // We need the origin to be in the minimum x, y, z corner. This
  // depends on the patient position
  switch (exam_info_sptr->patient_position.get_position()) {
  case PatientPosition::unknown_position:
    // If unknown, assume HFS
  case PatientPosition::HFS:
    // HFS means currently in LPI
    // So origin is in RAS direction
    orienter
      ->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RAS);
    break;

  case PatientPosition::HFP:
    // HFP means currently in RAI
    // So origin is in LPS direction
    orienter
      ->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_LPS);
    break;

  case PatientPosition::FFS:
    // FFS means currently in RPS
    // So origin is in LAI direction
    orienter
      ->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_LAI);
    break;

  case PatientPosition::FFP:
    // FFP means currently in LAS
    // So origin is in RPI direction
    orienter
      ->SetDesiredCoordinateOrientation(itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RPI);
    break;

  default:
    throw std::runtime_error("Unsupported patient position, not sure how to read.");
  }

  orienter->Update();
  return orienter->GetOutput();
}

/* Convert an ITK image into an internal STIR one. */
template<typename ITKImageType, typename STIRImageType>
static inline
STIRImageType*
convert_ITK_to_STIR(const typename ITKImageType::Pointer itk_image,
                    bool is_displacement_field=false)
{
  // Construct extra metadata
  const shared_ptr<ExamInfo> exam_info_sptr
    = construct_exam_info_from_metadata_dictionary(itk_image->GetMetaDataDictionary());
  // Reorient the ITK image to align with STIR axes
  typename ITKImageType::Pointer reor_itk_image
    = orient_ITK_image<ITKImageType>(itk_image, exam_info_sptr);
  // Make the STIR Image
  STIRImageType* stir_image_ptr = construct_empty_stir_image
    <typename ITKImageType::Pointer, STIRImageType>(reor_itk_image, exam_info_sptr);
  // Copy the ITK image data into the STIR Image
  copy_ITK_data_to_STIR_image<ITKImageType, STIRImageType>
    (reor_itk_image, *stir_image_ptr, is_displacement_field);
  return stir_image_ptr;
}

//To read any file format via ITK
template<>
inline
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
          ReaderType::Pointer reader = ReaderType::New();

          reader->SetFileName(filename);
          reader->Update();
          ITKImageSingle::Pointer itk_image = reader->GetOutput();

          return convert_ITK_to_STIR
            <ITKImageSingle, STIRImageSingleConcrete>
            (itk_image);
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
          // Reads complete series.
          nameGenerator->AddSeriesRestriction("0008|0022" ); // AcquisitionDate
          //nameGenerator->AddSeriesRestriction("0008|0032" ); // AcquisitionTime
          //nameGenerator->AddSeriesRestriction("0018|1060" ); // TriggerTime
          //nameGenerator->AddSeriesRestriction("0018|1063" ); // FrameTime

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

          // Update custom patient position tag in metadata
          itk_image->SetMetaDataDictionary(dicomIO->GetMetaDataDictionary());

          // Finally, convert to STIR!
          return convert_ITK_to_STIR
            <ITKImageSingle, STIRImageSingleConcrete>
            (itk_image);
     
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
inline
STIRImageMulti*
read_file_itk(const std::string &filename)
{
  try
    {
      // Not a DICOM file, so we just read a single image
      typedef itk::ImageFileReader<ITKImageMulti> ReaderType;
      ReaderType::Pointer reader = ReaderType::New();
      reader->SetFileName(filename);
      reader->Update();

      // Only support Nifti for now
      if (strcmp(reader->GetImageIO()->GetNameOfClass(), "NiftiImageIO") != 0) {
          error("read_file_itk: Only Nifti images are currently support for multicomponent images %s:%d.",
                __FILE__, __LINE__);
          return NULL; }

      if (reader->GetImageIO()->GetPixelType() != itk::ImageIOBase::VECTOR) {
          error("read_file_itk: Image type should be vector %s:%d.",
                __FILE__, __LINE__);
          return NULL; }

      warning("Only displacement fields are currently supported in STIR (not deformations). "
              "There is no way of verifying this from the nifti_image metadata, so you need to "
              "make sure that the image you are supplying is a displacement field image.");

      ITKImageMulti::Pointer itk_image = reader->GetOutput();

      return convert_ITK_to_STIR<ITKImageMulti, STIRImageMulti>
        (itk_image, true);

    }
  catch (std::exception &ex)
    {
      error(ex.what());
      return 0;
    }
}

// explicit instantiations
template class ITKImageInputFileFormat<STIRImageSingle>;
template class ITKImageInputFileFormat<STIRImageMulti>;

END_NAMESPACE_STIR
