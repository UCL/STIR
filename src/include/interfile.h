// 
// $Id$: $Date$
//
#ifndef __Interfile_h__
#define __Interfile_h__

#include "KeyParser.h"
#include "NumericInfo.h"
#include "pet_common.h"
#include "imagedata.h"
// KT 14/01/2000 added sinodata.h
#include "sinodata.h"
// SM&KT 14/02/2000 changed
#include "CartesianCoordinate3D.h"

START_NAMESPACE_TOMO

// This reads the first image from an Interfile header
// If there is trouble interpreting the header, 
// ask_image_details() is called instead
// If the name for the data file is not an absolute pathname,
// 'directory_for_data' is prepended (if not NULL).
// KT 14/01/2000 added directory capability
PETImageOfVolume read_interfile_image(istream& input, 
				      const char * const directory_for_data = 0);

// KT 13/11/98 new
// This first opens a stream and then calls the previous function
// with 'directory_for_data' set to the directory part of 'filename'.
PETImageOfVolume read_interfile_image(const char *const filename);

// SM&KT 17/01/2000 new
// This outputs an Interfile header for an image.
// No extensions will be added to the filenames.
// Return is 'true' when succesful, 'false' otherwise.
// 
// In fact, at the moment 2 headers are output:
// 'header_file_name' is 'new-style' (suitable for Mediman for instance), 
// *.ahv is 'old-style' (suitable for Analyze for instance)
// They both point to the same file with binary data.
bool 
write_basic_interfile_image_header(const string& header_file_name,
				   const string& image_file_name,
				   const Coordinate3D<int>& dimensions,
				   const Point3D& voxel_size,
				   const NumericType output_type,
				   const ByteOrder byte_order,
				   const VectorWithOffset<float>& scaling_factors,
				   const VectorWithOffset<unsigned long>& file_offsets);

// SM&KT 17/01/2000 new
// a utility function that computes the file offsets of subsequent images
const VectorWithOffset<unsigned long> 
compute_file_offsets(int number_of_time_frames,
		 const NumericType output_type,
		 const Coordinate3D<int>& dim,
		 unsigned long initial_offset = 0);

// This outputs an Interfile header and data for a Tensor3D object.
// Extensions .hv (and .ahv) and .v will be added to the parameter 'filename' 
// Return is 'true' when succesful, 'false' otherwise.
template <class NUMBER>
bool write_basic_interfile(const char * const filename, 
			   const Tensor3D<NUMBER>& image,
			   const Point3D& voxel_size,
			   const NumericType output_type = NumericType::FLOAT);

// KT 09/11/98 added this for 'old' interface
template <class NUMBER>
inline bool 
write_basic_interfile(const char * const filename, 
		      const Tensor3D<NUMBER>& image,
		      const NumericType output_type = NumericType::FLOAT)
{
  return
    write_basic_interfile(filename, 
			  image, 
			  Point3D(1,1,1), 
			  output_type);
}

// This preserves voxel_size
bool 
write_basic_interfile(const char * const filename, 
		      const PETImageOfVolume& image,
		      const NumericType output_type = NumericType::FLOAT);


// This reads the first 3D sinogram from an Interfile header
// If there is trouble interpreting the header, 
// ask_PSOV_details() is called instead
// If the name for the data file is not an absolute pathname,
// 'directory_for_data' is prepended (if not NULL).
// KT 14/01/2000 added directory capability
PETSinogramOfVolume read_interfile_PSOV(istream& input,
 				        const char * const directory_for_data = 0);

// This first opens a stream and then calls the previous function
// with 'directory_for_data' set to the directory part of 'filename'.
PETSinogramOfVolume read_interfile_PSOV(const char *const filename);

bool write_basic_interfile_PSOV_header(const string& header_file_name,
				       const string& image_file_name,
				       const PETSinogramOfVolume& psov);

//CL26/01/2000 New add 
bool 
write_basic_interfile_PSOV_header(const string& header_file_name,
				  const PETSegment& segment);

END_NAMESPACE_TOMO

#endif // __Interfile_h__

