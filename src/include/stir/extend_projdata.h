//
//
/*
    Copyright (C) 2005- 2009, Hammersmith Imanet Ltd
    Copyright 2023, Positrigo AG, Zurich
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*
  \ingroup projdata
  \file  Functions that extend a direct sinogram or segment.
  
  \author Charalampos Tsoumpas
  \author Kris Thielemans
  \author Markus Jehl
*/
START_NAMESPACE_STIR

//@{
/*!						  
\ingroup projdata
\brief Extension of direct projection data.

Functions that extend the given sinogram or segment in the view direction taking 
periodicity into account, if exists. If the sinogram is not symmetric in
tangential position, the values are extrapolated by nearest neighbour known values.

This is probably only useful before calling interpolation routines, or for FORE.
*/

/*!
 \brief Generic function to extend a segment in any or all directions. Axially and
        tangentially, the segment is filled with the nearest existing value.
        In view direction, the function wraps around for stir::ProjData that cover 180°
        or 360° degrees, and throws an error for other angular coverages.
 \ingroup numerics
 \param[in,out] segment segment to be extended.
 \param[in] view_extension how many views to add either side of the segment
 \param[in] axial_extension how many axial bins to add either side of the segment
 \param[in] tangential_extension how many tangential bins to add either side of the segment
*/
Array<3,float>
extend_segment(const SegmentBySinogram<float>& segment, const int view_extension = 5,
               const int axial_extension = 5, const int tangential_extension = 5);
//@}

END_NAMESPACE_STIR
