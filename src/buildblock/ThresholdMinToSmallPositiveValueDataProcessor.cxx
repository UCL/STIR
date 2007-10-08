//
// $Id$
//
/*
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
/*!

  \file
  \ingroup DataProcessor
  \brief Implementations for class stir::ThresholdMinToSmallPositiveValueDataProcessor

  \author Kris Thielemans

  $Date$
  $Revision$
*/
#include "stir/ThresholdMinToSmallPositiveValueDataProcessor.h"
#include "stir/thresholding.h"
#include "stir/DiscretisedDensity.h"

START_NAMESPACE_STIR

  
template <typename DataT>
Succeeded
ThresholdMinToSmallPositiveValueDataProcessor<DataT>::
virtual_set_up(const DataT& density)

{
  return Succeeded::yes;  
}


template <typename DataT>
void
ThresholdMinToSmallPositiveValueDataProcessor<DataT>::
virtual_apply(DataT& data) const

{     
  threshold_min_to_small_positive_value(data.begin_all(), data.end_all(), 0.000001F);
  //threshold_min_to_small_positive_value_and_truncate_rim(data, 0);
}


template <typename DataT>
void
ThresholdMinToSmallPositiveValueDataProcessor<DataT>::
virtual_apply(DataT& out_data, 
	  const DataT& in_data) const
{
  out_data = in_data;
  threshold_min_to_small_positive_value(out_data.begin_all(), out_data.end_all(), 0.000001F);
}

template <typename DataT>
ThresholdMinToSmallPositiveValueDataProcessor<DataT>::
ThresholdMinToSmallPositiveValueDataProcessor()
{
  set_defaults();
}

template <typename DataT>
void
ThresholdMinToSmallPositiveValueDataProcessor<DataT>::
set_defaults()
{
  base_type::set_defaults();
}

template <typename DataT>
void 
ThresholdMinToSmallPositiveValueDataProcessor<DataT>::
initialise_keymap()
{
  base_type::initialise_keymap();
  this->parser.add_start_key("Threshold Min To Small Positive Value Parameters");
  this->parser.add_stop_key("END Threshold Min To Small Positive Value Parameters");
}



template<class DataT>
const char * const 
ThresholdMinToSmallPositiveValueDataProcessor<DataT>::
  registered_name =
  "Threshold Min To Small Positive Value";


#  ifdef _MSC_VER
// prevent warning message on reinstantiation, 
// note that we get a linking error if we don't have the explicit instantiation below
#  pragma warning(disable:4660)
#  endif

// Register this class in the DataProcessor registry
// static ThresholdMinToSmallPositiveValueDataProcessor<float>::RegisterIt dummy;
// have the above variable in a separate file, which you need to pass at link time

template class ThresholdMinToSmallPositiveValueDataProcessor<DiscretisedDensity<3,float> >;

END_NAMESPACE_STIR

#ifdef STIR_DEVEL  
#include "local/stir/modelling/ParametricDiscretisedDensity.h"  
#include "local/stir/modelling/KineticParameters.h"  
namespace stir {  
  template class ThresholdMinToSmallPositiveValueDataProcessor<ParametricVoxelsOnCartesianGrid >;   
}  
#endif  



