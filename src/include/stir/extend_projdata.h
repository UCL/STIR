//
//
/*
    Copyright (C) 2005- 2009, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*
  \ingroup projdata
  \file  Functions that extend a direct sinogram or segment in the view direction
  
  \author Charalampos Tsoumpas
  \author Kris Thielemans

*/
START_NAMESPACE_STIR

//@{
/*!						  
\ingroup projdata
\brief Extension of direct projection data in view direction.

Functions that extend the given sinogram or segment in the view direction taking 
periodicity into account, if exists. If the sinogram is not symmetric in
tangential position, the values are extrapolated by nearest neighbour known values.

This is probably only useful before calling interpolation routines, or for FORE.
*/
Array<3,float>
extend_segment_in_views(const SegmentBySinogram<float>& sino, 
			const int min_view_extension, const int max_view_extension);
Array<2,float>
extend_sinogram_in_views(const Sinogram<float>& sino,
			 const int min_view_extension, const int max_view_extension);
//@}

END_NAMESPACE_STIR


