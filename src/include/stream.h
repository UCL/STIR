//
// $Id$: $Date$
//
/*!

  \file
  \ingroup buildblock

  \brief Output of basic types to an ostream

  \author Kris Thielemans
  \author PARAPET project

  \date $Date$

  \version $Revision$
*/
#include <iostream>

#ifndef TOMO_NO_NAMESPACES
using  std::ostream;
#endif

#include "VectorWithOffset.h"
#include "BasicCoordinate.h"

START_NAMESPACE_TOMO

/*!
  \brief Outputs a VectorWithOffset to a stream.

  Output is of the form 
  \verbatim
  {1, 2, 3}
  \endverbatim
  with an endl at the end. 
  
  This can be used for higher dimensional arrays as well, where each 1D subobject 
  will be on its own line.
*/

template <typename elemT>
ostream& 
operator<<(ostream& str, const VectorWithOffset<elemT>& v)
{
      str << '{';
      for (int i=v.get_min_index(); i<v.get_max_index(); i++)
	str << v[i] << ", ";
      str << v[v.get_max_index()] << '}' << endl;
      return str;
}

/*!
  \brief Outputs a BasicCoordinate to a stream.

  Output is of the form 
  \verbatim
  {1, 2, 3}
  \endverbatim
  with no endl at the end. 
  */
template <int num_dimensions, typename coordT>
ostream& 
operator<<(ostream& str, const BasicCoordinate<num_dimensions, coordT>& v)
{
      str << '{';
      for (int i=1; i<num_dimensions; i++)
	str << v[i] << ", ";
      str << v[num_dimensions] << '}';
      return str;
}
END_NAMESPACE_TOMO
