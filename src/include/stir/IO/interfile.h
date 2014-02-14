// 
//
#ifndef __stir_Interfile_h__
#define __stir_Interfile_h__

/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2011, Hammersmith Imanet Ltd
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
  \ingroup InterfileIO
 
  \brief  Declarations of functions which read/write Interfile data

  \author Kris Thielemans 
  \author Sanida Mustafovic
  \author PARAPET project

*/

#include "stir/NumericType.h"
// note that I had to include Succeeded.h instead of just forward 
// declaring the class. Otherwise every file that used write_*interfile*
// has to include Succeeded.h (even if it doesn't use the return value).
#include "stir/Succeeded.h"
#include "stir/ByteOrder.h"
#include <iostream>
#include <string>

#ifndef STIR_NO_NAMESPACES
using std::string;
using std::istream;
using std::ios;
#endif


START_NAMESPACE_STIR

template <int num_dimensions> class IndexRange;
template <int num_dimensions, typename elemT> class Array;
template <int num_dimensions, typename elemT> class DiscretisedDensity;
template <typename elemT> class VectorWithOffset;
template <typename elemT> class CartesianCoordinate3D;
template <typename elemT> class Coordinate3D;
template <typename elemT> class VoxelsOnCartesianGrid;
class ProjDataFromStream;

//! Checks if the signature corresponds to the start of an interfile header 
/*! 
  \ingroup InterfileIO
  The first line of an Interfile header should contain the 
    \verbatim 
    INTERFILE :=
    \endverbatim
    (potentially preceded by an exclamation mark, and maybe in 
    mixed or lower case).
    This function checks if the list of characters pointed to by
    \a signature satifies these requirements.

    \warning The parameter \a signature should be 0 terminated.
*/
bool is_interfile_signature(const char * const signature);

//! This reads the first 3d image in an Interfile header file, given as a stream
/*!
  \ingroup InterfileIO
 If there is trouble interpreting the header, 
 VoxelsOnCartesianGrid<float>::ask_parameters() is called instead
 If the name for the data file is not an absolute pathname,
 \a directory_for_data is prepended (if not NULL).

  \warning it is up to the caller to deallocate the image

  This should normally never be used. Use read_from_file<DiscretisedDensity<3,float> >() instead.
 */
VoxelsOnCartesianGrid<float>* read_interfile_image(istream& input, 
				      const string& directory_for_data = "");

//! This reads the first 3d image in an Interfile header file, given as a filename.
/*!
  \ingroup InterfileIO
 The function first opens a stream from 'filename' and then calls the previous function
 with 'directory_for_data' set to the directory part of 'filename'.

  \warning it is up to the caller to deallocate the image

  This should normally never be used. Use read_from_file<DiscretisedDensity<3,float> >() instead.
*/
VoxelsOnCartesianGrid<float>* read_interfile_image(const string& filename);

//! This outputs an Interfile header for an image.
/*!
  \ingroup InterfileIO
 A .hv extension will be added to the header_file_name if none is present.
 \return Succeeded::yes when succesful, Succeeded::no otherwise.
 
 In fact, at the moment 2 headers are output:
 <ul>
 <li>'header_file_name' is 'new-style' (suitable for Mediman for instance), 
 <li>*.ahv is 'old-style' (Interfile version 3.3) (suitable for Analyze for instance)
 </ul>
 They both point to the same file with binary data.

 \warning The .ahv file contains a fix such that Analyze reads the data
 with the correct voxel size (in z), which is probably non-confirming, and
 so will get other programs to read the voxel size incorrectly. 
 A relevant comment is written in each .ahv file.
 */

Succeeded 
write_basic_interfile_image_header(const string& header_file_name,
				   const string& image_file_name,
				   const IndexRange<3>& index_range,
				   const CartesianCoordinate3D<float>& voxel_size,
				   const CartesianCoordinate3D<float>& origin,
				   const NumericType output_type,
				   const ByteOrder byte_order,
				   const VectorWithOffset<float>& scaling_factors,
				   const VectorWithOffset<unsigned long>& file_offsets);


//! a utility function that computes the file offsets of subsequent images
/*!
   \ingroup InterfileIO
*/
const VectorWithOffset<unsigned long> 
compute_file_offsets(int number_of_time_frames,
		 const NumericType output_type,
		 const Coordinate3D<int>& dim,
		 unsigned long initial_offset = 0);


//! This outputs an Interfile header and data for a Array<3,elemT> object.
/*!
  \ingroup InterfileIO
 Extension .v will be added to the parameter 'filename' (if no extension present).
 Extensions .hv (and .ahv) will be used for the header filename. 
*/

template <class elemT>
Succeeded 
write_basic_interfile(const string&filename, 
		      const Array<3,elemT>& image,
		      const CartesianCoordinate3D<float>& voxel_size,
		      const CartesianCoordinate3D<float>& origin,
		      const NumericType output_type = NumericType::FLOAT,
		      const float scale= 0,
		      const ByteOrder byte_order=ByteOrder::native);

//! This outputs an Interfile header and data for a Array<3,elemT> object, assuming unit voxel sizes
/*!
  \ingroup InterfileIO
 Extension .v will be added to the parameter 'filename' (if no extension present).
 Extensions .hv (and .ahv) will be used for the header filename. 

 \warning For Visual Studio 7.0 or earlier, only the float version is defined
   to work around a 
  compiler bug. (Otherwise, the float version is not instantiated for some reason).
*/

#if !defined(_MSC_VER) || (_MSC_VER > 1300)
template <class elemT>
#else
#define elemT float
#endif
Succeeded 
write_basic_interfile(const string& filename, 
		      const Array<3,elemT>& image,
		      const NumericType output_type = NumericType::FLOAT,
		      const float scale= 0,
		      const ByteOrder byte_order=ByteOrder::native);
#if defined(_MSC_VER) && (_MSC_VER <= 1300)
#undef elemT 
#endif

//! This outputs an Interfile header and data for a VoxelsOnCartesianGrid<float> object
/*!
  \ingroup InterfileIO
 Extension .v will be added to the parameter 'filename' (if no extension present).
 Extensions .hv (and .ahv) will be used for the header filename. 
*/
Succeeded 
write_basic_interfile(const string& filename, 
		      const VoxelsOnCartesianGrid<float>& image,
		      const NumericType output_type = NumericType::FLOAT,
		      const float scale= 0,
		      const ByteOrder byte_order=ByteOrder::native);


//! This outputs an Interfile header and data for a DiscretisedDensity<3,float> object
/*!
  \ingroup InterfileIO
 Extension .v will be added to the parameter 'filename' (if no extension present).
 Extensions .hv (and .ahv) will be used for the header filename. 

  Currently the DiscretisedDensity<3,float>& object has to be a reference to a 
  VoxelsOnCartesianGrid<float> object.
*/
Succeeded 
write_basic_interfile(const string& filename, 
		      const DiscretisedDensity<3,float>& image,
		      const NumericType output_type = NumericType::FLOAT,
		      const float scale= 0,
		      const ByteOrder byte_order=ByteOrder::native);

//! This reads the first 3D sinogram from an Interfile header, given as a stream
/*!
  \ingroup InterfileIO
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
/*!
  \ingroup InterfileIO
  This first opens a stream and then calls the previous function
  with 'directory_for_data' set to the directory part of 'filename'.

  \warning it is up to the caller to deallocate the object

  This should normally never be used. Use ProjData::read_from_file() instead.
*/
ProjDataFromStream* read_interfile_PDFS(const string& filename,
					const ios::openmode open_mode);

//! This writes an Interfile header appropriate for the ProjDataFromStream object.
/*!
  \ingroup InterfileIO
  A .hs extension will be added to the header_file_name if none is present.
 \return Succeeded::yes when succesful, Succeeded::no otherwise.
*/
 Succeeded write_basic_interfile_PDFS_header(const string& header_filename,
      			                const string& data_filename,
				        const ProjDataFromStream& pdfs);

//! This function writes an Interfile header for the pdfs object.
/*! 
  \ingroup InterfileIO
  The header_filename is found by replacing the extension in the 
   data_filename with .hs
   \return Succeeded::yes when succesful, Succeeded::no otherwise.
 */
Succeeded write_basic_interfile_PDFS_header(const string& data_filename,
			    const ProjDataFromStream& pdfs);

END_NAMESPACE_STIR

#endif // __Interfile_h__
