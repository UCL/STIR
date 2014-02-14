//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2007, Hammersmith Imanet Ltd
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
  \ingroup projdata
  \brief Implementations for non-inline functions of class stir::Segment

  \author Kris Thielemans
  \author PARAPET project


*/

#include "stir/Segment.h"
#include "stir/SegmentByView.h"
#include "stir/IndexRange2D.h"
#include "stir/IndexRange3D.h"
#include "boost/format.hpp"

START_NAMESPACE_STIR


template<typename elemT>
bool
Segment<elemT>::
has_same_characteristics(self_type const& other,
			 string& explanation) const
{
  using boost::format;
  using boost::str;

  if (typeid(*this) != typeid(other))
    {
      explanation =
	str(format("Differing data types:%1% vs %2%")
	    % typeid(*this).name()
	    % typeid(other).name()
	    );
      return false;
    }
  if (*this->get_proj_data_info_ptr() !=
      *other.get_proj_data_info_ptr())
    {
      explanation = 
	str(format("Differing projection data info:\n%1%\n-------- vs-------\n %2%")
	    % this->get_proj_data_info_ptr()->parameter_info()
	    % other.get_proj_data_info_ptr()->parameter_info()
	    );
      return false;
    }
  if (this->get_segment_num() !=
      other.get_segment_num())
    {
      explanation = 
	str(format("Differing segment number: %1% vs %2%")
	    % this->get_segment_num()
	    % other.get_segment_num()
	    );
      return false;
    }
  return true;
}

template<typename elemT>
bool
Segment<elemT>::
has_same_characteristics(self_type const& other) const
{
  std::string explanation;
  return this->has_same_characteristics(other, explanation);
}

template<typename elemT>
bool 
Segment<elemT>::
operator !=(const self_type& that) const
{
  return !((*this) == that);
}

/*************************************
 instantiations
 *************************************/

template class Segment<float>;

END_NAMESPACE_STIR
