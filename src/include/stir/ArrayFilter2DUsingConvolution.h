//
//
/*!

  \file
  \ingroup buildblock
  \brief Declaration of class ArrayFilter2DUsingConvolution

  \author Kris Thielemans
  \author Sanida Mustafovic

*/
/*
    Copyright (C) 2000- 2002, IRSL
    See STIR/LICENSE.txt for details
*/

#ifndef __stir_ArrayFilter2DUsingConvolution_H__
#define __stir_ArrayFilter2DUsingConvolution_H__


#include "stir/ArrayFunctionObject_2ArgumentImplementation.h"
//#include "stir/VectorWithOffset.h"

START_NAMESPACE_STIR

template <typename elemT> class VectorWithOffset;

template <typename elemT>
class ArrayFilter2DUsingConvolution : 
  public ArrayFunctionObject_2ArgumentImplementation<2,elemT>
{
public:

  //! Construct the filter given the kernel coefficients
  /*! 
    All kernel coefficients have to be passed. 
  */
  ArrayFilter2DUsingConvolution();

  ArrayFilter2DUsingConvolution(const Array <2, float>& filter_kernel);
  
  bool is_trivial() const;

 virtual Succeeded 
    get_influencing_indices(IndexRange<1>& influencing_indices, 
                            const IndexRange<1>& output_indices) const;

  virtual Succeeded 
    get_influenced_indices(IndexRange<1>& influenced_indices, 
                           const IndexRange<1>& input_indices) const;

private:
  Array <2, float>  filter_coefficients;
  void do_it(Array<2,elemT>& out_array, const Array<2,elemT>& in_array) const;

};



END_NAMESPACE_STIR


#endif //ArrayFilter2DUsingConvolution


