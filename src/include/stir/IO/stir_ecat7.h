//
// $Id$
//
/*!
  \file
  \ingroup ECAT

  \brief Declaration of routines which convert CTI things into our 
  building blocks and vice versa.

  \author Kris Thielemans
  \author PARAPET project

  $Date$

  $Revision$

*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/

#ifndef __stir_IO_stir_ecat7_H__
#define __stir_IO_stir_ecat7_H__


#include "stir/IO/stir_ecat_common.h"
#include "stir/NumericType.h"

#ifdef HAVE_LLN_MATRIX
#ifdef STIR_NO_NAMESPACES
// terrible trick to avoid conflict between stir::Sinogram and Sinogram defined in matrix.h
// when we do have namespaces, the conflict can be resolved by using ::Sinogram
#define Sinogram CTISinogram
#else
#define CTISinogram ::Sinogram
#endif

#include "matrix.h"

#ifdef STIR_NO_NAMESPACES
#undef Sinogram
#endif

#include <string>
#include <iostream>

#ifndef STIR_NO_NAMESPACES
using std::string;
using std::iostream;
#endif

START_NAMESPACE_STIR

class Succeeded;
class ByteOrder;
class Scanner;
template <int num_dimensions, typename elemT> class DiscretisedDensity;
template <typename elemT> class VoxelsOnCartesianGrid;
template <typename elemT> class Sinogram;
template <typename T> class shared_ptr;
class ProjData;
class ProjDataInfo;
class ProjDataFromStream;

START_NAMESPACE_ECAT
START_NAMESPACE_ECAT7

/*!
  \brief checks if the file is in ECAT7 format
  \ingroup ECAT
  This partly relies on the implementation of matrix_open in the LLN matrix library.
  Additional checks are made on the main header. Current checks are:
  <ol>
  <li> sw_version field between 70 and 79
  <li> file_type field one of the values in the enum MatFileType
  <li> num_frames field > 0
  </ol>
  \warning When the file is not readable, error() is called.
*/
bool is_ECAT7_file(const string& filename);
/*!
  \brief checks if the file is in ECAT6 format and if the file contains images
  \ingroup ECAT
*/
bool is_ECAT7_image_file(const string& filename);
/*!
  \brief checks if the file is in ECAT6 format and
  if the file contains emission sinograms (or blank/transmision)
  \ingroup ECAT
*/
bool is_ECAT7_emission_file(const string& filename);
/*!
  \brief checks if the file is in ECAT6 format and 
  if the file contains attenuation correction factors
  \ingroup ECAT
*/
bool is_ECAT7_attenuation_file(const string& filename);


//! determine scanner type from the main_header
/*! 
  \ingroup ECAT
  Returns a Unknown_Scanner if it does not recognise it. */
void find_scanner(shared_ptr<Scanner> & scanner_ptr,const Main_header& mhead);


//! Create a new ECAT7 image file and write the data in there
/*!
  \ingroup ECAT
*/
Succeeded 
DiscretisedDensity_to_ECAT7(DiscretisedDensity<3,float> const & density, 
                            string const & cti_name, string const& orig_name,
			    const Scanner& scanner,
                            const int frame_num = 1, const int gate_num = 1, 
			    const int data_num = 0, const int bed_num = 0);

//! Write an (extra) image to an existing ECAT7 file 
/*! 
  \ingroup ECAT
  Some consistency checks are performed between the image and the data in the main header
  \warning This does NOT write the main header.
  */
Succeeded 
DiscretisedDensity_to_ECAT7(MatrixFile *mptr,
                            DiscretisedDensity<3,float> const & density, 
			    const int frame_num = 1, const int gate_num = 1, 
			    const int data_num = 0, const int bed_num = 0);

//! Create a new ECAT7  sinogram file and write the data in there
/*! 
  \ingroup ECAT
  Note that not all \a output_type are supported by the ECAT7 format.
  If a wrong type is used, it will be forced to floats.
*/
Succeeded 
ProjData_to_ECAT7(ProjData const& proj_data, NumericType output_type,
		  string const & cti_name, string const & orig_name,
		  const int frame_num = 1, const int gate_num = 1, 
		  const int data_num = 0, const int bed_num = 0,
		  const bool write_as_attenuation = false,
                  float scale_factor = 0.0F);

//! Write an (extra) set of sinograms to an existing ECAT7 file 
/*! 
  \ingroup ECAT
   Some consistency checks are performed between the proj_data and the data
   in the main header pointer of \a mptr.

   For ECAT7, the data type for the output is determined by the \c file_type field
   of the main header.

  \warning This does NOT write the main header.
*/
Succeeded 
ProjData_to_ECAT7(MatrixFile *mptr,
		  ProjData const& proj_data, 
		  const int frame_num = 1, const int gate_num = 1, 
                  const int data_num = 0, const int bed_num = 0,
                  float scale_factor = 0.0F);


//! Fill in most of the main header given a Scanner object and orig_name.
/*!
  \ingroup ECAT
*/
void make_ECAT7_main_header(Main_header&, 
			    const Scanner&,
                            const string& orig_name                     
                            );

//! Fill in most of the main header given a Scanner object and orig_name and an image
/*!
  \ingroup ECAT
  Sets num_planes, plane_separation as well*/
void make_ECAT7_main_header(Main_header& mhead,
			    Scanner const& scanner,
                            const string& orig_name,
                            DiscretisedDensity<3,float> const & density
                            );

//! Fill in most of the main header given an orig_name and a proj_data_info
/*! 
  \ingroup ECAT
  It gets the scanner from the proj_data_info object.
  Sets num_planes, plane_separation as well and attempts septa_state.

  \return the actual NumericType that should be used for further IO.
  This is necessary because the file_type in the ECAT7 main header  
  depends on the \a output_type (sign).

  Defaults mean that it will set to file_type Float3dSinogram.

  \warning   Note that not all \a output_type are supported by the ECAT7 format.
  If a wrong type is used, it will be forced to floats.
  \warning   The \c acquisition_type field will be set to either 
  \c TransmissionScan or \c StaticEmission, depending on \a write_as_attenuation. This is not necessarily correct.

*/
NumericType 
make_ECAT7_main_header(Main_header& mhead,
		       const string& orig_name,
		       ProjDataInfo const & proj_data_info,
		       const bool write_as_attenuation = false,
		       NumericType output_type = NumericType::FLOAT
		       );

//! Fill in most of the subheader
/*! 
  \ingroup ECAT
  \warning data_type and volume_mode has still to be set */
void
make_subheader_for_ECAT7(Attn_subheader& shead, 
                         const Main_header& mhead,
                         const ProjDataInfo& proj_data_info
                         );
//! Fill in most of the subheader
/*!
  \ingroup ECAT
  \warning data_type and volume_mode has still to be set */
void
make_subheader_for_ECAT7(Scan3D_subheader& shead, 
                         const Main_header& mhead,
                         const ProjDataInfo& proj_data_info
                         );

//! Make a ProjDataFromStream object that 'points' into an ECAT7 file
/*! 
  \ingroup ECAT
    \arg mptr is the LLN structure that has the file pointer etc for the ECAT7 file
    \arg matrix encodes frame, gate, data, bed numbers
    \arg stream_ptr is a pointer to a stream object corresponding to the SAME file as the
        ECAT7 file. This argument is necessary as it isn't possible to make a stream from 
        a FILE pointer.
    \return NULL when it is an unsupported data type
    
    Only data types AttenCor, Byte3dSinogram, Short3dSinogram, Float3dSinogram can be handled.
    Other sinogram formats have subheaders interleaved with the data which makes it 
    impossible to have a corresponding ProjDataFromStream
*/
ProjDataFromStream * 
make_pdfs_from_matrix(MatrixFile * const mptr, 
                      MatrixData * const matrix, 
                      const shared_ptr<iostream>&  stream_ptr);

//! Writes an Interfile header that 'points' into an ECAT7 file
/*! 
  \ingroup ECAT
  Only data types AttenCor, Byte3dSinogram, Short3dSinogram, Float3dSinogram,
  ByteVolume, PetVolume can be handled.

    \a interfile_header_name will be set to the header name used. It will be of the form
    ECAT7_filename_extension_f1g1d0b0.hs or .hv. For example, for ECAT7_filename test.S, and
    frame=2, gate=3, data=4, bed=5, the header name will be test_S_f2g3d4b5.hs
*/
Succeeded 
write_basic_interfile_header_for_ECAT7(string& interfile_header_name,
                                       const string& ECAT7_filename,
				       const int frame_num, const int gate_num, const int data_num, const int bed_num);

/*
  \brief Read an image from an ECAT7 file.
  \ingroup ECAT
  \warning do not use directly, but use DiscretisedDensity<3,float>::read_from_file().

  \return a pointer to a newly allocated image, or 0 if it failed.
*/
VoxelsOnCartesianGrid<float> * 
ECAT7_to_VoxelsOnCartesianGrid(const string& ECAT7_filename,
			       const int frame_num, const int gate_num, const int data_num, const int bed_num);
/* 
  \brief Read projection data from an ECAT7 file
  \ingroup ECAT
  \warning do not use directly, but use ProjData::read_from_file().

  \return a pointer to a newly allocated ProjDataFromStream object, or 0 if it failed.
*/
ProjDataFromStream*
ECAT7_to_PDFS(const string& ECAT7_filename,
		   const int frame_num, const int gate_num, const int data_num, const int bed_num);

END_NAMESPACE_ECAT7
END_NAMESPACE_ECAT
END_NAMESPACE_STIR

#endif

#endif
