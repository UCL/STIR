//
//
/*!

  \file
  \ingroup buildblock
  \brief Declaration of class ArrayFilter3DUsingConvolution

  \author Kris Thielemans
  \author Sanida Mustafovic

*/
/*
    Copyright (C) 2000- 2002, IRSL
    See STIR/LICENSE.txt for details
*/

#ifndef __stir_ArrayFilter3DUsingConvolution_H__
#define __stir_ArrayFilter3DUsingConvolution_H__


#include "stir/ArrayFunctionObject_2ArgumentImplementation.h"
//#include "stir/VectorWithOffset.h"

START_NAMESPACE_STIR

template <typename elemT> class VectorWithOffset;

template <typename elemT>
class ArrayFilter3DUsingConvolution : 
  public ArrayFunctionObject_2ArgumentImplementation<3,elemT>
{
public:

  //! Construct the filter given the kernel coefficients
  /*! 
    All kernel coefficients have to be passed. 
  */
  ArrayFilter3DUsingConvolution();

  ArrayFilter3DUsingConvolution(const Array <3, float>& filter_kernel);
  
  bool is_trivial() const;

 virtual Succeeded 
    get_influencing_indices(IndexRange<1>& influencing_indices, 
                            const IndexRange<1>& output_indices) const;

  virtual Succeeded 
    get_influenced_indices(IndexRange<1>& influenced_indices, 
                           const IndexRange<1>& input_indices) const;

private:
  Array <3, float>  filter_coefficients;
  void do_it(Array<3,elemT>& out_array, const Array<3,elemT>& in_array) const;
  void do_it_2d(Array<2,elemT>& out_array, const Array<2,elemT>& in_array) const;

};



END_NAMESPACE_STIR


#endif //ArrayFilter3DUsingConvolution


