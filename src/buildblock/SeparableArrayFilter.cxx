//
// $Id$
//
/*!

  \file

  \brief Implementations for class SeparableArrayFilter

  \author Kris Thielemans
  \author Sanida Mustafovic

  \date $Date$
  \version $Revision$
*/

#include "tomo/SeparableArrayFilter.h"
#include "ArrayFunction.h"

START_NAMESPACE_TOMO

template <int num_dim, typename elemT>
SeparableArrayFilter<num_dim, elemT>::
SeparableArrayFilter()
: all_1d_array_filters(VectorWithOffset< shared_ptr<ArrayFilter<1,elemT> > >(num_dim))
{}

template <int num_dim, typename elemT>
SeparableArrayFilter<num_dim, elemT>::
SeparableArrayFilter(const VectorWithOffset< shared_ptr<ArrayFilter<1,elemT> > >& array_filters)
: all_1d_array_filters(array_filters)
{
  assert(all_1d_array_filters.get_length() == num_dim);
}


template <int num_dim, typename elemT>
void 
SeparableArrayFilter<num_dim, elemT>::
operator() (Array<num_dim,elemT>& array) const
{
   
  if (!is_trivial())
   in_place_apply_array_functions_on_each_index(array, 
                                             all_1d_array_filters.begin(), 
                                             all_1d_array_filters.end());

}

template <int num_dim, typename elemT>
bool 
SeparableArrayFilter<num_dim, elemT>::
is_trivial() const
{
  for ( std::vector< shared_ptr<ArrayFilter<1,elemT> > >::const_iterator iter=all_1d_array_filters.begin();
        iter!=all_1d_array_filters.end();++iter)
   {
     if (!(*iter)->is_trivial())
       return false;
   }
   return true;
}


// instantiation
template SeparableArrayFilter<3, float>;

END_NAMESPACE_TOMO



