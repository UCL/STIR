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
#include <vector>
#ifndef TOMO_NO_NAMESPACES
using  std::ostream;
using  std::vector;
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
      // KT 8/12/1000 corrected case for 0 length
      if (v.get_length()>0)
	str << v[v.get_max_index()];
      str << '}' << endl;
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
      // KT 8/12/1000 corrected case for 0 length
      if (num_dimensions>0)
	str << v[num_dimensions];
      str << '}';
      return str;
}


/*!
  \brief Outputs a vector to a stream.

  Output is of the form 
  \verbatim
  {1, 2, 3}
  \endverbatim
  with an endl at the end. 
  
  For each element of the vector ostream::operator<<() will be called.
*/
template <typename elemT>
ostream& 
operator<<(ostream& str, const vector<elemT>& v)
{
      str << '{';
      // slightly different from above because vector::size() is unsigned
      // so 0-1 == 0xFFFFFFFF (and not -1)
      if (v.size()>0)
      {
        for (unsigned int i=0; i<v.size()-1; i++)
	  str << v[i] << ", ";      
	str << v[v.size()-1];
      }
      str << '}' << endl;
      return str;
}

END_NAMESPACE_TOMO
