//
// $Id$: $Date$
//

#ifndef __NumericVectorWithOffset_H__
#define __NumericVectorWithOffset_H__
/*!
  \file 
 
  \brief defines the NumericVectorWithOffset class

  \author Kris Thielemans
  \author PARAPET project

  \date    $Date$

  \version $Revision$

*/


#include "VectorWithOffset.h"

START_NAMESPACE_TOMO
/*! 
  \ingroup buildblock
  \brief like VectorWithOffset, but with various numeric operations defined

  Extra operations compared to VectorWithOffset are all the numeric
  operators +,-,*,/ with NumericVectorWithOffset objects of the
  same type, but also with objects of type \c elemT.
 */
 
template <class T, class elemT>
class NumericVectorWithOffset : public VectorWithOffset<T>
{
private:
  typedef VectorWithOffset<T> base_type;

public:
  //! Construct an empty NumericVectorWithOffset
  inline NumericVectorWithOffset();

  //! Construct a NumericVectorWithOffset of given length
  inline explicit NumericVectorWithOffset(const int hsz);
    
  //! Construct a NumericVectorWithOffset of elements with offset \c min_index
  inline NumericVectorWithOffset(const int min_index, const int max_index);


  // arithmetic operations with a vector, combining element by element

  //! adding vectors, element by element
  inline NumericVectorWithOffset operator+ (const NumericVectorWithOffset &v) const;

  //! subtracting vectors, element by element
  inline NumericVectorWithOffset operator- (const NumericVectorWithOffset &v) const;

  //! multiplying vectors, element by element
  inline NumericVectorWithOffset operator* (const NumericVectorWithOffset &v) const;

  //! dividing vectors, element by element
  inline NumericVectorWithOffset operator/ (const NumericVectorWithOffset &v) const;


  // arithmetic operations with a elemT
  // TODO??? use member templates 

  //! return a new vector with elements equal to the sum of the elements in the original and the \c elemT 
  inline NumericVectorWithOffset operator+ (const elemT &v) const;

  //! return a new vector with elements equal to the difference of the elements in the original and the \c elemT 
  inline NumericVectorWithOffset operator- (const elemT &v) const;

  //! return a new vector with elements equal to the multiplication of the elements in the original and the \c elemT 
  inline NumericVectorWithOffset operator* (const elemT &v) const;
	       
  //! return a new vector with elements equal to the division of the elements in the original and the \c elemT 
  inline NumericVectorWithOffset operator/ (const elemT &v) const;


  // corresponding assignment operators

  //! adding elements of \c v to the current vector
  inline NumericVectorWithOffset & operator+= (const NumericVectorWithOffset &v);

  //! subtracting elements of \c v from the current vector
  inline NumericVectorWithOffset & operator-= (const NumericVectorWithOffset &v);

  //! multiplying elements of the current vector with elements of \c v 
  inline NumericVectorWithOffset & operator*= (const NumericVectorWithOffset &v);

  //! dividing all elements of the current vector by elements of \c v
  inline NumericVectorWithOffset & operator/= (const NumericVectorWithOffset &v);

  //! adding an \c elemT to the elements of the current vector
  inline NumericVectorWithOffset & operator+= (const elemT &v);

  //! subtracting an \c elemT from the elements of the current vector
  inline NumericVectorWithOffset & operator-= (const elemT &v);

  //! multiplying the elements of the current vector with an \c elemT 
  inline NumericVectorWithOffset & operator*= (const elemT &v);

  //! dividing the elements of the current vector by an \c elemT 
  inline NumericVectorWithOffset & operator/= (const elemT &v);

};

END_NAMESPACE_TOMO

#include "NumericVectorWithOffset.inl"

#endif // __NumericVectorWithOffset_H__
