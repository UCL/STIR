//
// $Id$
//
/*
    Copyright (C) 2005- $Date$, Hammersmith Imanet Ltd
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
/*
  \ingroup projdata
  \file  Functions that extend a direct sinogram or segment in the view direction
  
  \author Charalampos Tsoumpas
  \author Kris Thielemans

  $Date$
  $Revision$
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


