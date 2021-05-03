//
//
/*!
  \file
  \ingroup Array
  \brief Declaration of class stir::SeparableArrayFunctionObject

  \author Kris Thielemans
  \author Sanida Mustafovic

*/
/*
    Copyright (C) 2000- 2009, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/


#ifndef __stir_SeparableArrayFunctionObject_H__
#define __stir_SeparableArrayFunctionObject_H__

#include "stir/ArrayFunctionObject_1ArgumentImplementation.h"
#include "stir/shared_ptr.h"
#include "stir/VectorWithOffset.h"

START_NAMESPACE_STIR



/*!
  \ingroup Array
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
  //! Default constructor, results in a trivial ArrayFunctionObject
  SeparableArrayFunctionObject ();
  //! Constructor taking 1D ArrayFunctionObjects
  /*!
    The 1d functino objects are passed in a VectorWithOffset which needs to 
    have num_dimensions elements. (The starting index is irrelevant).

    The shared_ptr's have to be either all null (a trivial object) or all non-null.
   */
  SeparableArrayFunctionObject (const VectorWithOffset< shared_ptr<ArrayFunctionObject<1,elemT> > >&); 

  bool is_trivial() const;

protected:
 
  VectorWithOffset< shared_ptr<ArrayFunctionObject<1,elemT> > > all_1d_array_filters;
  virtual void do_it(Array<num_dimensions,elemT>& array) const;

};


END_NAMESPACE_STIR


#endif //SeparableArrayFunctionObject

