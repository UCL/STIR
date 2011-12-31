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
  \brief Implementations for class stir::ChainedDataProcessor

  \author Kris Thielemans

  $Date$
  $Revision$
*/
#include "stir/ChainedDataProcessor.h"
#include "stir/DiscretisedDensity.h"
#include "stir/modelling/ParametricDiscretisedDensity.h"  
#include "stir/modelling/KineticParameters.h"  
#include "stir/is_null_ptr.h"
#include <memory>

#ifndef STIR_NO_NAMESPACES
using std::auto_ptr;
#endif

START_NAMESPACE_STIR

  
template <typename DataT>
Succeeded
ChainedDataProcessor<DataT>::
virtual_set_up(const DataT& data)
{
  if (!is_null_ptr(this->apply_first))
    {
      // note that we cannot really build the filter for the 2nd 
      // as we don't know what the first will do to the dimensions etc. of the data
      return this->apply_first->set_up(data);
    }
  else if (!is_null_ptr(this->apply_second))
    return this->apply_second->set_up(data);
  else
    return Succeeded::yes;  
}


template <typename DataT>
void
ChainedDataProcessor<DataT>::
virtual_apply(DataT& data) const
{  
  if (!is_null_ptr(this->apply_first))
    this->apply_first->apply(data);
  if (!is_null_ptr(this->apply_second))
    this->apply_second->apply(data);
}


template <typename DataT>
void
ChainedDataProcessor<DataT>::
virtual_apply(DataT& out_data, 
	  const DataT& in_data) const
{
  if (!is_null_ptr(this->apply_first))
    {
      if (!is_null_ptr(this->apply_second))
	{
	  // a bit complicated because we need a temporary data copy
	  auto_ptr< DataT> temp_data_ptr =
	    auto_ptr< DataT>(in_data.get_empty_copy());      
	  this->apply_first->apply(*temp_data_ptr, in_data);
	  this->apply_second->apply(out_data, *temp_data_ptr);
	}
      else
	this->apply_first->apply(out_data, in_data);
    }
  else
      if (!is_null_ptr(this->apply_second))
	this->apply_second->apply(out_data, in_data);

}

template <typename DataT>
ChainedDataProcessor<DataT>::
ChainedDataProcessor(shared_ptr<DataProcessor<DataT> > apply_first_v,
		      shared_ptr<DataProcessor<DataT> > apply_second_v)
  : apply_first(apply_first_v),
    apply_second(apply_second_v)
{
  this->set_defaults();
}

template <typename DataT>
void
ChainedDataProcessor<DataT>::
set_defaults()
{
  base_type::set_defaults();
}

template <typename DataT>
void 
ChainedDataProcessor<DataT>::
initialise_keymap()
{
  base_type::initialise_keymap();
  this->parser.add_start_key("Chained Data Processor Parameters");
  this->parser.add_parsing_key("Data Processor to apply first", &this->apply_first);
  this->parser.add_parsing_key("Data Processor to apply second", &this->apply_second);
  this->parser.add_stop_key("END Chained Data Processor Parameters");
}



template <class DataT>
const char * const 
ChainedDataProcessor<DataT>::registered_name =
  "Chained Data Processor";


#  ifdef _MSC_VER
// prevent warning message on reinstantiation, 
// note that we get a linking error if we don't have the explicit instantiation below
#  pragma warning(disable:4660)
#  endif

// Register this class in the DataProcessor registry
// static ChainedDataProcessor<float>::RegisterIt dummy;
// have the above variable in a separate file, which you need to pass at link time

template class ChainedDataProcessor<DiscretisedDensity<3,float> >;
template class ChainedDataProcessor<ParametricVoxelsOnCartesianGrid >;   
END_NAMESPACE_STIR





