//
//
/*
    Copyright (C) 2022 National Physical Laboratory
    Copyright (C) 2022 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*
  \ingroup projdata
  \file Declaration of stir::interpolate_axial_position
  
  \author Daniel Deidda
  \author Kris Thielemans

*/

START_NAMESPACE_STIR

class ProjData;
template <int num_dimensions, class T> class BasicCoordinate;
template <class elemT> class Sinogram;
template <class elemT> class SegmentBySinogram;


//! \brief Perform linear Interpolation
/*! 
  \ingroup projdata
  \param[out] proj_data_out Its projection_data_info is used to 
  determine output characteristics. Data will be 'put' in here using 
  ProjData::set_sinogram().
  \param[in] proj_data_in input data 
  This function interpolates from a axially downsampled projdata to a full scanner.
  This mostly is useful in the scatter sinogram expansion.
*/  
//@{
Succeeded 
interpolate_axial_position(ProjData& proj_data_out,
		     const ProjData& proj_data_in);
//@}

END_NAMESPACE_STIR



