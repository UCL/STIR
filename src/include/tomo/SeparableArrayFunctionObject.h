//
// $Id$
//
/*!
  \file
  \ingroup buildblock
  \brief Declaration of class SeparableArrayFunctionObject

  \author Kris Thielemans
  \author Sanida Mustafovic

  \date $Date$
  \version $Revision$
*/


#ifndef __Tomo_SeparableArrayFunctionObject_H__
#define __Tomo_SeparableArrayFunctionObject_H__

#include "tomo/ArrayFunctionObject_1ArgumentImplementation.h"
#include "shared_ptr.h"
#include <vector>
#include "VectorWithOffset.h"

#ifndef TOMO_NO_NAMESPACES
using std::vector;
#endif


START_NAMESPACE_TOMO



/*!
  \ingroup buildblock
  \brief This class implements an \c n -dimensional ArrayFunctionObject whose operation
  is separable.

  'Separable' means that its operation consists of \c n 1D operations, one on each
  index of the \c n -dimensional array. 
  \see in_place_apply_array_functions_on_each_index()
  
 */
template <int num_dimensions, typename elemT>
class SeparableArrayFunctionObject : 
   public ArrayFunctionObject_1ArgumentImplementation<num_dimensions,elemT>
{
public:
  SeparableArrayFunctionObject ();
  SeparableArrayFunctionObject (const VectorWithOffset< shared_ptr<ArrayFunctionObject<1,elemT> > >&); 
  bool is_trivial() const;

protected:
 
  VectorWithOffset< shared_ptr<ArrayFunctionObject<1,elemT> > > all_1d_array_filters;
  virtual void do_it(Array<num_dimensions,elemT>& array) const;

};


END_NAMESPACE_TOMO


#endif //SeparableArrayFunctionObject

