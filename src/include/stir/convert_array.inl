//
// $Id$
//
/*
    Copyright (C) 2006 - $Date$, Hammersmith Imanet Ltd
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
  \ingroup Array
  \brief implementation of stir::convert_array

  \author Kris Thielemans

  $Date$

  $Revision$

*/
#include "stir/NumericInfo.h"
#include "stir/Array.h"
#include "stir/convert_range.h"
START_NAMESPACE_STIR

template <int num_dimensions, class T1, class T2, class scaleT>
void
find_scale_factor(scaleT& scale_factor,
		  const Array<num_dimensions,T1>& data_in, 
		  const NumericInfo<T2> info_for_out_type)
{
  find_scale_factor(scale_factor, data_in.begin_all(), data_in.end_all(), info_for_out_type);
}



template <int num_dimensions, class T1, class T2, class scaleT>
Array<num_dimensions, T2>
convert_array(scaleT& scale_factor,
	      const Array<num_dimensions, T1>& data_in, 
	      const NumericInfo<T2> info_for_out_type)
{
  Array<num_dimensions,T2> data_out(data_in.get_index_range());

  convert_array(data_out, scale_factor, data_in);
  return data_out;    
}

template <int num_dimensions, class T1, class T2, class scaleT>
void 
convert_array(Array<num_dimensions, T2>& data_out,
	      scaleT& scale_factor,
	      const Array<num_dimensions, T1>& data_in)
{
  convert_range(data_out.begin_all(), scale_factor, 
		data_in.begin_all(), data_in.end_all());   
}

END_NAMESPACE_STIR
