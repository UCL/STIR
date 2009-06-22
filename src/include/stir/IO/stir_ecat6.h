//
// $Id$
//
/*!
  \file
  \ingroup ECAT

  \brief Declaration of routines which convert ECAT6 things into our 
  building blocks and vice versa.

  \author Kris Thielemans
  \author PARAPET project

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
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

#ifndef __stir_IO_stir_ecat6_H__
#define __stir_IO_stir_ecat6_H__

#include "stir/IO/stir_ecat_common.h"
#include "stir/IO/ecat6_types.h"
#include <string>
#include <stdio.h>

#ifndef STIR_NO_NAMESPACES
using std::string;
#endif


START_NAMESPACE_STIR

class Succeeded;
class NumericType;
class ByteOrder;
class Scanner;
template <int num_dimensions, typename elemT> class DiscretisedDensity;
template <typename elemT> class VoxelsOnCartesianGrid;
template <typename elemT> class Sinogram;
template <typename T> class shared_ptr;
class ProjData;
class ProjDataInfo;

START_NAMESPACE_ECAT
START_NAMESPACE_ECAT6


/*!
  \brief checks if the file is in ECAT6 format
  \ingroup ECAT

  As ECAT6 doesn't have a 'magic number' this check is somewhat heuristic.
  Checks are only on the main header. Current checks are:
  <ol>
  <li> sw_version field between 0 and 69
  <li> file_type field one of the values in the enum MatFileType
  <li> num_frames field > 0
  </ol>
*/
bool is_ECAT6_file(const string& filename);
/*!
  \brief checks if the file is in ECAT6 format and if the file contains images
  \ingroup ECAT
*/
bool is_ECAT6_image_file(const string& filename);
/*!
  \brief checks if the file is in ECAT6 format and
  if the file contains emission sinograms (or blank/transmision)
  \ingroup ECAT
*/
  bool is_ECAT6_emission_file(const string& filename);
/*!
  \brief checks if the file is in ECAT6 format and 
  if the file contains attenuation correction factors
  \ingroup ECAT
*/
bool is_ECAT6_attenuation_file(const string& filename);

/*
  \brief Convert image data
  \ingroup ECAT
  \param cti_fptr a FILE pointer to the ECAT6 file.
  \param mhead the ECAT6 main header. Note that this parameter will be used
         to get system and size info, not the main header in the file.
*/
VoxelsOnCartesianGrid<float> * 
ECAT6_to_VoxelsOnCartesianGrid(const int frame_num, const int gate_num, const int data_num, const int bed_num,
                      FILE *cti_fptr, const ECAT6_Main_header & mhead);
/* 
  \brief Convert sinogram data
  \ingroup ECAT
  \param max_ring_diff if less than 0, the maximum is used (i.e. num_rings-1)
  \param arccorrected tells the function if the data is (assumed to be) arc-corrected. Note 
         that the ECAT6 file format does not have any flag to indicate this.
  \param output_file_name filename for output. A .s extension will be added (for the
         binary file) if no extension is present.
  \param cti_fptr a FILE pointer to the ECAT6 file.
  \param mhead the ECAT6 main header. Note that this parameter will be used
         to get system and size info, not the main header in the file.
  \warning multiplies the data with the loss_correction_factor in the subheader.
*/
void ECAT6_to_PDFS(const int frame_num, const int gate_num, const int data_num, const int bed_num,
		   int max_ring_diff, bool arccorrected,
		   const string& output_file_name, 
                   FILE *cti_fptr, const ECAT6_Main_header & mhead);

//! determine scanner type from the ECAT6_Main_header
/*! 
  \ingroup ECAT
  Returns a Unknown_Scanner if it does not recognise it. */
Scanner * find_scanner_from_ECAT6_Main_header(const ECAT6_Main_header& mhead);

//! Create a new ECAT6 image file and write the data in there
/*! \ingroup ECAT*/
Succeeded 
DiscretisedDensity_to_ECAT6(DiscretisedDensity<3,float> const & density, 
                            string const & cti_name, string const& orig_name,
			    const Scanner& scanner,
                            const int frame_num = 1, const int gate_num = 1, 
			    const int data_num = 0, const int bed_num = 0);

//! Write an (extra) image to an existing ECAT6 file 
/*! 
  \ingroup ECAT
  Some consistency checks are performed between the image and the data in the main header
  \warning This does NOT write the main header.
  */
Succeeded 
DiscretisedDensity_to_ECAT6(FILE *fptr,
                            DiscretisedDensity<3,float> const & density, 
			    const ECAT6_Main_header& mhead,
                            const int frame_num = 1, const int gate_num = 1, 
			    const int data_num = 0, const int bed_num = 0);

//! Create a new ECAT6  sinogram file and write the data in there
/*! 
  \ingroup ECAT
  \warning Only data without axial compression can be handled by the ECAT6 3D 
  sinogram format, and hence also by this function (CTI span==1), except
  when write_2D_sinograms==true
 */
Succeeded ProjData_to_ECAT6(ProjData const& proj_data, 
                            string const & cti_name, string const & orig_name,
                            const int frame_num = 1, const int gate_num = 1, 
			    const int data_num = 0, const int bed_num = 0,
			    const bool write_2D_sinograms = false);

//! Write an (extra) set of sinograms to an existing ECAT6 file 
/*! 
  \ingroup ECAT
   Some consistency checks are performed between the proj_data and the data in the main header
  \warning Only data without axial compression can be handled by the ECAT6 3D 
  sinogram format, and hence also by this function (CTI span==1), except
  when write_2D_sinograms==true
  \warning This does NOT write the main header.
*/
Succeeded 
ProjData_to_ECAT6(FILE *fptr, ProjData const& proj_data, 
                  const ECAT6_Main_header& mhead,
                  const int frame_num = 1, const int gate_num = 1, 
                  const int data_num = 0, const int bed_num = 0,
			    const bool write_2D_sinograms = false);


//! Fill in most of the main header given a Scanner object and orig_name.
/*!
  \ingroup ECAT
*/
void make_ECAT6_Main_header(ECAT6_Main_header&, 
			    const Scanner&,
                            const string& orig_name                     
                            );

//! Fill in most of the main header given a Scanner object and orig_name and an image
/*! 
  \ingroup ECAT
  Sets file_type, num_planes, plane_separation as well*/
void make_ECAT6_Main_header(ECAT6_Main_header& mhead,
			    Scanner const& scanner,
                            const string& orig_name,
                            DiscretisedDensity<3,float> const & density
                            );

//! Fill in most of the main header given an orig_name and a proj_data_info
/*! 
  \ingroup ECAT
  It gets the scanner from the proj_data_info object.
   Sets file_type, num_planes, plane_separation as well*/
void make_ECAT6_Main_header(ECAT6_Main_header& mhead,
			    const string& orig_name,
                            ProjDataInfo const & proj_data_info
                            );
END_NAMESPACE_ECAT
END_NAMESPACE_ECAT6
END_NAMESPACE_STIR
#endif
