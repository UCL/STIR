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
  \file  
  
  \author Charalampos Tsoumpas
  \author Kris Thielemans

  $Date$
  $Revision$
*/
START_NAMESPACE_STIR

//@{
/*!						  
\ingroup projdata
\brief Extension of projection data in view direction.
	Functions that extend the given sinogram or segment in the direction of views taking 
	periodicity into account, if exists. If the sinogram is not symmetric the values 
	are extrapolated by nearest neighbour known values.
*/
inline
Array<3,float>
extend_segment(const SegmentBySinogram<float>& sino, 
						const int min_view_extension, const int max_view_extension);
inline
Array<2,float>
extend_sinogram(const Sinogram<float>& sino,
						const int min_view_extension, const int max_view_extension);
//@}

END_NAMESPACE_STIR

#include "local/stir/extend_projdata.inl"


