//
// $Id$
//
/*!

  \file

  \brief 

  \author Kris Thielemans
  \author Sanida Mustafovic

  \date $Date$
  \version $Revision$
*/

#ifndef __Tomo_ArrayFilterUsingSeparableConv_H__
#define __Tomo_ArrayFilterUsingSeparableConv_H__


#include "tomo/SeparableArrayFunctionObject.h"

START_NAMESPACE_TOMO
template <typename elemT>
class ArrayFilter1DUsingConvolution : public ArrayFunctionObject<1,elemT>
{
public:
  ArrayFilter1DUsingConvolution(const VectorWithOffset< elemT>&);
  /*!
  \warning Currently requires that out_array and in_array have the same index range
  */
  void operator() (Array<1,elemT>& out_array, const Array<1,elemT>& in_array) const;
  bool is_trivial() const;

private:
  VectorWithOffset< elemT> filter_coefficients;
  //static void cir_shift_to_right(VectorWithOffset<elemT>&output,const VectorWithOffset<elemT>&input);

};



END_NAMESPACE_TOMO


#endif //ArrayFilter1DUsingConvolution


