//
// $Id$
//
/*!

  \file

  \brief Declaration of routines which convert CTI things into our 
  building blocks and vice versa.

  \author Kris Thielemans
  \author PARAPET project

  $Date$

  $Revision$

  \warning matrix.h has to be included before this
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#ifndef __stir_CTI_stir_ecat7_H__
#define __stir_CTI_stir_ecat7_H__

#include "stir/common.h"
#include <string>

#ifndef STIR_NO_NAMESPACES
using std::string;
#endif

//*************** namespace macros
#if 0 //ndef STIR_NO_NAMESPACE
# define START_NAMESPACE_ECAT7 namespace ecat7 {
# define END_NAMESPACE_ECAT7 }
# define USING_NAMESPACE_ECAT7 using namespace ecat7;
#else
# define START_NAMESPACE_ECAT7 
# define END_NAMESPACE_ECAT7 
# define USING_NAMESPACE_ECAT7 
#endif

START_NAMESPACE_STIR
START_NAMESPACE_ECAT7

class NumericType;
class ByteOrder;
class Scanner;
template <int num_dimensions, typename elemT> class DiscretisedDensity;
template <typename elemT> class VoxelsOnCartesianGrid;
template <typename elemT> class Sinogram;
template <typename T> class shared_ptr;
class ProjData;
class ProjDataFromStream;


//! determine scanner type from the main_header
/*! Returns a Unknown_Scanner if it does not recognise it. */
void find_scanner(shared_ptr<Scanner> & scanner_ptr,const Main_header& mhead);
//! Find out which NumericType and ByteOrder corresponds to a CTI data type
void find_data_type(NumericType& data_type, ByteOrder& byte_order, const short ecat_data_type);

//! Find out which CTI data type corresponds to a certain NumericType and ByteOrder
/*! Returns 0 when it does not recognise it */
short find_cti_data_type(const NumericType& type, const ByteOrder& byte_order);


//! Create a new ECAT7 image file and write the data in there
Succeeded 
DiscretisedDensity_to_ECAT7(DiscretisedDensity<3,float> const & density, 
                            string const & cti_name, string const& orig_name,
			    const Scanner& scanner,
                            const int frame_num = 1, const int gate_num = 1, 
			    const int data_num = 0, const int bed_num = 0);

//! Write an (extra) image to an existing ECAT7 file 
/*! 
  Some consistency checks are performed between the image and the data in the main header
  \warning This does NOT write the main header.
  */
Succeeded 
DiscretisedDensity_to_ECAT7(MatrixFile *mptr,
                            DiscretisedDensity<3,float> const & density, 
			    const int frame_num = 1, const int gate_num = 1, 
			    const int data_num = 0, const int bed_num = 0);

//! Create a new ECAT7  sinogram file and write the data in there
Succeeded ProjData_to_ECAT7(ProjData const& proj_data, 
                            string const & cti_name, string const & orig_name,
                            const int frame_num = 1, const int gate_num = 1, 
			    const int data_num = 0, const int bed_num = 0,
                            const bool write_as_attenuation = false);

//! Write an (extra) set of sinograms to an existing ECAT7 file 
/*! 
   Some consistency checks are performed between the proj_data and the data in the main header
  \warning This does NOT write the main header.
*/
Succeeded 
ProjData_to_ECAT7(MatrixFile *mptr, ProjData const& proj_data, 
                  const int frame_num = 1, const int gate_num = 1, 
                  const int data_num = 0, const int bed_num = 0);


//! Fill in most of the main header given a Scanner object and orig_name.
void make_ECAT7_main_header(Main_header&, 
			    const Scanner&,
                            const string& orig_name                     
                            );

//! Fill in most of the main header given a Scanner object and orig_name and an image
/*! Sets num_planes, plane_separation as well*/
void make_ECAT7_main_header(Main_header& mhead,
			    Scanner const& scanner,
                            const string& orig_name,
                            DiscretisedDensity<3,float> const & density
                            );

//! Fill in most of the main header given an orig_name and a proj_data_info
/*! It gets the scanner from the proj_data_info object.
    Sets num_planes, plane_separation as well*/
void make_ECAT7_main_header(Main_header& mhead,
			    const string& orig_name,
                            ProjDataInfo const & proj_data_info
                            );

//! Fill in most of the subheader
/*! \warning data_type and volume_mode has still to be set */
void
make_subheader_for_ecat7(Attn_subheader& shead, 
                         const Main_header& mhead,
                         const ProjDataInfo& proj_data_info
                         );
//! Fill in most of the subheader
/*! \warning data_type and volume_mode has still to be set */
void
make_subheader_for_ecat7(Scan3D_subheader& shead, 
                         const Main_header& mhead,
                         const ProjDataInfo& proj_data_info
                         );

//! Make a ProjDataFromStream object that 'points' into an ECAT7 file
/*! 
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
/*! Only data types AttenCor, Byte3dSinogram, Short3dSinogram, Float3dSinogram,
    ByteVolume, PetVolume can be handled.
*/
Succeeded 
write_basic_interfile_header_for_ecat7(const string& ecat7_filename,
                                       int frame, int gate, int data,
                                       int bed);

#if 0
/*
  \brief Convert image data
*/
VoxelsOnCartesianGrid<float> * 
ECAT7_to_VoxelsOnCartesianGrid(const int frame_num, const int gate_num, const int data_num, const int bed_num,
                      MatrixFile *mptr, const Main_header & v_mhead);
/* 
  \brief Convert sinogram data
  \param max_ring_diff if less than 0, the maximum is used (i.e. num_rings-1)
  \param arccorrected tells the function if the data is (assumed to be) arc-corrected. Note 
         that the ECAT7 file format does not have any flag to indicate this.
*/
void ECAT7_to_PDFS(const int frame_num, const int gate_num, const int data_num, const int bed_num,
		   int max_ring_diff, bool arccorrected,
		   char *v_data_name, MatrixFile *mptr, const Main_header & v_mhead);
#endif

END_NAMESPACE_ECAT7

END_NAMESPACE_STIR

#endif
