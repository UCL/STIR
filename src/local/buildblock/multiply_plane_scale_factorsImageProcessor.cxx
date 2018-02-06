//
//
/*!

  \file
  \ingroup ImageProcessor
  \brief Implementations for class multiply_plane_scale_factorsImageProcessor

  \author Kris Thielemans

*/
/*
    Copyright (C) 2003- 2007, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/
#include "local/stir/multiply_plane_scale_factorsImageProcessor.h"
#include "stir/DiscretisedDensity.h"
#include "stir/VectorWithOffset.h"
#include "stir/Succeeded.h"
#include <algorithm>

#ifndef STIR_NO_NAMESPACES
using std::copy;
#endif

START_NAMESPACE_STIR

template <>
const char * const 
multiply_plane_scale_factorsImageProcessor<float>::registered_name =
  "multiply_plane_scale_factors";


template <typename elemT>
void
multiply_plane_scale_factorsImageProcessor<elemT>::
set_defaults()
{
  base_type::set_defaults();
  plane_scale_factors.resize(0);
}

template <typename elemT>
void 
multiply_plane_scale_factorsImageProcessor<elemT>::
initialise_keymap()
{
  base_type::initialise_keymap();
  this->parser.add_start_key("multiply_plane_scale_factors Parameters");
  this->parser.add_key("plane_scale_factors",&plane_scale_factors);
  this->parser.add_stop_key("END multiply_plane_scale_factors Parameters");
}

template <typename elemT>
multiply_plane_scale_factorsImageProcessor<elemT>::
multiply_plane_scale_factorsImageProcessor()
{
  set_defaults();
}

template <typename elemT>
multiply_plane_scale_factorsImageProcessor<elemT>::
multiply_plane_scale_factorsImageProcessor(const vector<double>& plane_scale_factors)
 : plane_scale_factors(plane_scale_factors)
{
}

template <typename elemT>
multiply_plane_scale_factorsImageProcessor<elemT>::
multiply_plane_scale_factorsImageProcessor(const VectorWithOffset<double>& plane_scale_factors_v)
{
  plane_scale_factors.resize(plane_scale_factors_v.get_length());
  copy(plane_scale_factors_v.begin(), plane_scale_factors_v.end(), plane_scale_factors.begin());
}
  
template <typename elemT>
Succeeded
multiply_plane_scale_factorsImageProcessor<elemT>::
virtual_set_up(const DiscretisedDensity<3,elemT>& density)

{
  if (density.get_length()!=static_cast<int>(plane_scale_factors.size()))
  {
    warning("multiply_plane_scale_factors: number of planes (%d) should be equal to number of scale factors (%d).\n",
        density.get_length(),plane_scale_factors.size());
    return Succeeded::no;
  }
  else
    return Succeeded::yes;  
}


template <typename elemT>
void
multiply_plane_scale_factorsImageProcessor<elemT>::
virtual_apply(DiscretisedDensity<3,elemT>& density) const

{     
  if (density.get_length()!=static_cast<int>(plane_scale_factors.size()))
    {
      error("Exiting\n");
    }
  for (int z=density.get_min_index(); 
       z<=density.get_max_index();
       ++z)
         density[z] *= 
	   static_cast<float>
	   (
	    plane_scale_factors[static_cast<std::vector<double>::size_type>
				(z-density.get_min_index())]
	    );  
}


template <typename elemT>
void
multiply_plane_scale_factorsImageProcessor<elemT>::
virtual_apply(DiscretisedDensity<3,elemT>& out_density, 
	  const DiscretisedDensity<3,elemT>& in_density) const
{
  out_density = in_density;
  virtual_apply(out_density);
}



#  ifdef _MSC_VER
// prevent warning message on reinstantiation, 
// note that we get a linking error if we don't have the explicit instantiation below
#  pragma warning(disable:4660)
#  endif

template class multiply_plane_scale_factorsImageProcessor<float>;

END_NAMESPACE_STIR



