//
// $Id$
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000 - 2007-10-08, Hammersmith Imanet Ltd
    Copyright (C) 2011-07-01 - $Date$, Kris Thielemans
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

  \brief Implementations for non-inline functions of class stir::Viewgram

  \author Kris Thielemans
  \author PARAPET project

  $Date$
  $Revision$
*/

#include "stir/Viewgram.h"
#include "boost/format.hpp"

#ifdef _MSC_VER
// disable warning that not all functions have been implemented when instantiating
#pragma warning(disable: 4661)
#endif // _MSC_VER
START_NAMESPACE_STIR

template<typename elemT>
bool
Viewgram<elemT>::
has_same_characteristics(self_type const& other,
			 string& explanation) const
{
  using boost::format;
  using boost::str;

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
  if (this->get_view_num() !=
      other.get_view_num())
    {
      explanation = 
	str(format("Differing view number: %1% vs %2%")
	    % this->get_view_num()
	    % other.get_view_num()
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
Viewgram<elemT>::
has_same_characteristics(self_type const& other) const
{
  std::string explanation;
  return this->has_same_characteristics(other, explanation);
}

template<typename elemT>
bool 
Viewgram<elemT>::
operator ==(const self_type& that) const
{
  return
    this->has_same_characteristics(that) &&
    base_type::operator==(that);
}
  
template<typename elemT>
bool 
Viewgram<elemT>::
operator !=(const self_type& that) const
{
  return !((*this) == that);
}

/*!
  This makes sure that the new Array dimensions are the same as those in the
  ProjDataInfo member.
*/
template <typename elemT>
void 
Viewgram<elemT>::
resize(const IndexRange<2>& range)
{   
  if (range == this->get_index_range())
    return;

  assert(range.is_regular()==true);

  const int ax_min = range.get_min_index();
  const int ax_max = range.get_max_index();
  
  shared_ptr<ProjDataInfo> pdi_ptr(proj_data_info_ptr->clone());

  pdi_ptr->set_min_axial_pos_num(ax_min, get_segment_num());
  pdi_ptr->set_max_axial_pos_num(ax_max, get_segment_num());
  pdi_ptr->set_min_tangential_pos_num(range[ax_min].get_min_index());
  pdi_ptr->set_max_tangential_pos_num(range[ax_min].get_max_index());

  proj_data_info_ptr = pdi_ptr;

  Array<2,elemT>::resize(range);
	
}


/*!
  This makes sure that the new Array dimensions are the same as those in the
  ProjDataInfo member.
*/
template <typename elemT>
void 
Viewgram<elemT>::
grow(const IndexRange<2>& range)
{
  resize(range);
}


/******************************
 instantiations
 ****************************/

template class Viewgram<float>;

END_NAMESPACE_STIR
