// 
// $Id$
//
#ifndef __Interfile_h__
#define __Interfile_h__
/*!
  \file 
  \ingroup buildblock
 
  \brief  Declarations of functions which read/write Interfile data

  \author Kris Thielemans 
  \author Sanida Mustafovic
  \author PARAPET project

  $Date$
  $Revision$
    
*/

#include "NumericType.h"
#include <iostream>
#include <string>

#ifndef TOMO_NO_NAMESPACES
using std::string;
using std::istream;
using std::ios;
#endif


START_NAMESPACE_TOMO

class ByteOrder;
template <int num_dimensions, typename elemT> class Array;
template <int num_dimensions, typename elemT> class DiscretisedDensity;
template <typename elemT> class VectorWithOffset;
template <typename elemT> class CartesianCoordinate3D;
template <typename elemT> class Coordinate3D;
template <typename elemT> class VoxelsOnCartesianGrid;
class ProjDataFromStream;

//! This reads the first 3d image in an Interfile header file, given as a stream
/*!
 If there is trouble interpreting the header, 
 VoxelsOnCartesianGrid<float>::ask_parameters() is called instead
 If the name for the data file is not an absolute pathname,
 \c directory_for_data is prepended (if not NULL).

  \warning it is up to the caller to deallocate the image

  This should normally never be used. Use DiscretisedDensity::read_from_file() instead.
 */
VoxelsOnCartesianGrid<float>* read_interfile_image(istream& input, 
				      const string& directory_for_data = "");

//! This reads the first 3d image in an Interfile header file, given as a filename.
/*!
 The function first opens a stream from 'filename' and then calls the previous function
 with 'directory_for_data' set to the directory part of 'filename'.

  \warning it is up to the caller to deallocate the image

  This should normally never be used. Use DiscretisedDensity::read_from_file() instead.
*/
VoxelsOnCartesianGrid<float>* read_interfile_image(const string& filename);

//! This outputs an Interfile header for an image.
/*!
 A .hv extension will be added to the header_file_name if none is present.
 \return Succeeded::yes when succesful, Succeeded::no otherwise.
 
 In fact, at the moment 2 headers are output:
 <ul>
 <li>'header_file_name' is 'new-style' (suitable for Mediman for instance), 
 <li>*.ahv is 'old-style' (suitable for Analyze for instance)
 </ul>
 They both point to the same file with binary data.
 */

Succeeded 
write_basic_interfile_image_header(const string& header_file_name,
				   const string& image_file_name,
				   const CartesianCoordinate3D<int>& dimensions,
				   const CartesianCoordinate3D<float>& voxel_size,
				   const NumericType output_type,
				   const ByteOrder byte_order,
				   const VectorWithOffset<float>& scaling_factors,
				   const VectorWithOffset<unsigned long>& file_offsets);


//! a utility function that computes the file offsets of subsequent images
const VectorWithOffset<unsigned long> 
compute_file_offsets(int number_of_time_frames,
		 const NumericType output_type,
		 const Coordinate3D<int>& dim,
		 unsigned long initial_offset = 0);


//! This outputs an Interfile header and data for a Array<3,elemT> object.
/*!
 Extension .v will be added to the parameter 'filename' (if no extension present).
 Extensions .hv (and .ahv) will be used for the header filename. 
 \return 'true' when succesful, 'false' otherwise.
*/

template <class elemT>
Succeeded 
write_basic_interfile(const string&filename, 
			   const Array<3,elemT>& image,
			   const CartesianCoordinate3D<float>& voxel_size,
			   const NumericType output_type = NumericType::FLOAT);

//! This outputs an Interfile header and data for a Array<3,elemT> object, assuming unit voxel sizes
/*!
 Extension .v will be added to the parameter 'filename' (if no extension present).
 Extensions .hv (and .ahv) will be used for the header filename. 
 \return 'true' when succesful, 'false' otherwise.

 \warning For Visual Studio, only the float version is defined to work around a 
  compiler bug. (Otherwise, the float version is not instantiated for some reason).
*/

#ifndef _MSC_VER
template <class elemT>
#else
#define elemT float
#endif
Succeeded 
write_basic_interfile(const string& filename, 
		      const Array<3,elemT>& image,
		      const NumericType output_type = NumericType::FLOAT);
#ifdef _MSC_VER
#undef elemT 
#endif

//! This outputs an Interfile header and data for a VoxelsOnCartesianGrid<float> object
/*!
 Extension .v will be added to the parameter 'filename' (if no extension present).
 Extensions .hv (and .ahv) will be used for the header filename. 
 \return 'true' when succesful, 'false' otherwise.
*/
Succeeded 
write_basic_interfile(const string& filename, 
		      const VoxelsOnCartesianGrid<float>& image,
		      const NumericType output_type = NumericType::FLOAT);


//! This outputs an Interfile header and data for a DiscretisedDensity<3,float> object
/*!
 Extension .v will be added to the parameter 'filename' (if no extension present).
 Extensions .hv (and .ahv) will be used for the header filename. 
 \return 'true' when succesful, 'false' otherwise.

  Currently the DiscretisedDensity<3,float>& object has to be a reference to a 
  VoxelsOnCartesianGrid<float> object.
*/
Succeeded 
write_basic_interfile(const string& filename, 
		      const DiscretisedDensity<3,float>& image,
		      const NumericType output_type = NumericType::FLOAT);

//! This reads the first 3D sinogram from an Interfile header, given as a stream
/*!
  If there is trouble interpreting the header, 
  ProjDataFromStream::ask_parameters() is called instead

  \param input A stream giving the Interfile header.

  \param directory_for_data If the name for the data file is not an absolute pathname,
  this string is prepended.
  
  \param openmode Mode for opening the data file. ios::binary will be added by the code.

  \warning it is up to the caller to deallocate the object  
*/
ProjDataFromStream* read_interfile_PDFS(istream& input,
 				        const string& directory_for_data = "",
					const ios::openmode openmode = ios::in);

//! This reads the first 3D sinogram from an Interfile header, given as a filename
/*! This first opens a stream and then calls the previous function
  with 'directory_for_data' set to the directory part of 'filename'.

  \warning it is up to the caller to deallocate the object

  This should normally never be used. Use ProjData::read_from_file() instead.
*/
ProjDataFromStream* read_interfile_PDFS(const string& filename,
					const ios::openmode open_mode);

//! This writes an Interfile header appropriate for the ProjDataFromStream object.
/*! A .hs extension will be added to the header_file_name if none is present.
 \return Succeeded::yes when succesful, Succeeded::no otherwise.
*/
 Succeeded write_basic_interfile_PDFS_header(const string& header_filename,
      			                const string& data_filename,
				        const ProjDataFromStream& pdfs);

//! This function writes an Interfile header for the pdfs object.
/*! The header_filename is found by replacing the extension in the 
   data_filename with .hs
   \return Succeeded::yes when succesful, Succeeded::no otherwise.
 */
Succeeded write_basic_interfile_PDFS_header(const string& data_filename,
			    const ProjDataFromStream& pdfs);

END_NAMESPACE_TOMO

#endif // __Interfile_h__



