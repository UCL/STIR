// NumericVectorWithOffset.h 
// TODO
// by KT, based on Tensor1D.h by DH

// $Id$: $Date$

#ifndef NumericVectorWithOffset_H
#define NumericVectorWithOffset_H
//KT include for min,max definitions
#include <algorithm>

#include "pet_common.h"
#include "VectorWithOffset.h"

 
template <class T, class NUMBER>
class NumericVectorWithOffset : public VectorWithOffset<T>
{
private:
	typedef VectorWithOffset<T> base_type;
public:
  // Construct an empty NumericVectorWithOffset
  NumericVectorWithOffset()
    : base_type()
    {}

  // Construct a NumericVectorWithOffset of given length
  NumericVectorWithOffset(const Int hsz)
    : base_type(hsz)
    {}
    
  // Construct a NumericVectorWithOffset of elements with offsets hfirst
  NumericVectorWithOffset(const Int hfirst, const Int hlast)   
    : base_type(hfirst, hlast)
    {}


  // member templates ??? TODO
  // matrix addition
#ifdef TEMPLATE_ARG
  NumericVectorWithOffset operator+ (const NumericVectorWithOffset<T, NUMBER> &iv) const 	
#else
  NumericVectorWithOffset operator+ (const NumericVectorWithOffset &iv) const 
#endif
  {
    check_state();
    NumericVectorWithOffset retval(*this);
    return retval += iv; };

  // matrix subtraction
#ifdef TEMPLATE_ARG
  NumericVectorWithOffset operator- (const NumericVectorWithOffset<T, NUMBER> &iv) const 
#else
  NumericVectorWithOffset operator- (const NumericVectorWithOffset &iv) const 
#endif
  {
    check_state();
    NumericVectorWithOffset retval(*this);
    return retval -= iv; }

  // elem by elem multiplication
  //KT 16/03/98 changed % to *
#ifdef TEMPLATE_ARG
  NumericVectorWithOffset operator* (const NumericVectorWithOffset<T, NUMBER> &iv) const 
#else
  NumericVectorWithOffset operator* (const NumericVectorWithOffset &iv) const
#endif
  {
    check_state();
    NumericVectorWithOffset retval(*this);
    return retval *= iv; }

  // elem by elem division
#ifdef TEMPLATE_ARG
  NumericVectorWithOffset operator/ (const NumericVectorWithOffset<T, NUMBER> &iv) const 
#else
  NumericVectorWithOffset operator/ (const NumericVectorWithOffset &iv) const
#endif
  {
    check_state();
    NumericVectorWithOffset retval(*this);
    return retval /= iv; }

  // TODO, use member templates, or implement as a function (not a member), 
  // this would get rid of the NUMBER part of the NumericVectorWithOffset template

  // Add a constant to every element
  NumericVectorWithOffset operator+ (const NUMBER &iv) const 
  {
    check_state();
    NumericVectorWithOffset retval(*this);
    retval += iv;
    return retval;
  }

  // Subtract a constant from every element
  NumericVectorWithOffset operator- (const NUMBER &iv) const 
  {
    check_state();
    NumericVectorWithOffset retval(*this);
    retval -= iv; 
    return retval;
  }

  // Multiply every element by a constant
  NumericVectorWithOffset operator* (const NUMBER &iv) const 
  {
    check_state();
    NumericVectorWithOffset retval(*this);
    retval *= iv;
    return retval;
  }
	       
  // Divide every element by a constant
  NumericVectorWithOffset operator/ (const NUMBER &iv) const 
  {
    check_state();
    NumericVectorWithOffset retval(*this);
    retval /= iv;
    return retval;
  }


#ifdef TEMPLATE_ARG
  NumericVectorWithOffset &operator+= (const NumericVectorWithOffset<T, NUMBER> &iv) 
#else
  NumericVectorWithOffset &operator+= (const NumericVectorWithOffset &iv) 
#endif
  {
    check_state();
    grow (min(start,iv.start), max(start+length-1,iv.start+iv.length-1));
    for (Int i=iv.start; i<iv.start + iv.length; i++)
      num[i] += iv.num[i];
    check_state();
    return *this; }

#ifdef TEMPLATE_ARG
  NumericVectorWithOffset &operator-= (const NumericVectorWithOffset<T, NUMBER> &iv)
#else
  NumericVectorWithOffset &operator-= (const NumericVectorWithOffset &iv)
#endif
  {
    check_state();
    grow (min(start,iv.start), max(start+length-1,iv.start+iv.length-1));
    for (Int i=iv.start; i<iv.start + iv.length; i++)
      num[i] -= iv.num[i];
    check_state();
    return *this; }

  // KT 16/03/98 changed from % to *
#ifdef TEMPLATE_ARG
  NumericVectorWithOffset &operator*= (const NumericVectorWithOffset<T, NUMBER> &iv)
#else
  NumericVectorWithOffset &operator*= (const NumericVectorWithOffset &iv)
#endif
  {
    check_state();
    grow (min(start,iv.start), max(start+length-1,iv.start+iv.length-1));
    for (Int i=iv.start; i<iv.start + iv.length; i++)
      num[i] *= iv.num[i];
    check_state();
    return *this; }

#ifdef TEMPLATE_ARG
  NumericVectorWithOffset &operator/= (const NumericVectorWithOffset<T, NUMBER> &iv)
#else
  NumericVectorWithOffset &operator/= (const NumericVectorWithOffset &iv)
#endif
  {
    check_state();
    grow (min(start,iv.start), max(start+length-1,iv.start+iv.length-1));
    for (Int i=iv.start; i<iv.start + iv.length; i++)
      num[i] /= iv.num[i];
    check_state();
    return *this; }

  NumericVectorWithOffset &operator+= (const NUMBER &iv) {
    check_state();
    for (Int i=start; i<start + length; i++)
      num[i] += iv;
    check_state();
    return *this; }

  NumericVectorWithOffset &operator-= (const NUMBER &iv) {
    check_state();
    for (Int i=start; i<start + length; i++)
      num[i] -= iv;
    check_state();
    return *this; }

  NumericVectorWithOffset &operator*= (const NUMBER &iv) {
    check_state();
    for (Int i=start; i<start + length; i++)
      num[i] *= iv;
    check_state();
    return *this; }

  NumericVectorWithOffset &operator/= (const NUMBER &iv) {
    check_state();
    for (Int i=start; i<start + length; i++)
      num[i] /= iv;
    check_state();
    return *this; }


};

#endif // NumericVectorWithOffset_H
