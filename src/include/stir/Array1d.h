
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2007, Hammersmith Imanet Ltd
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    See STIR/LICENSE.txt for details
*/
/*!
  \file 
  \ingroup Array 
  \brief defines the 1D specialisation of the Array class for broken compilers

  \author Kris Thielemans (with help from Alexey Zverovich)
  \author PARAPET project


*/

#if !defined(__stir_Array_H__) || !defined(BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION) || !defined(elemT)
#error This file should only be included in Array.h for half-broken compilers
#endif

/* Lines here should really be identical to what you find as 1D specialisation 
   in Array.h and Array.inl, except that template statements are dropped.
   */
template<>
class Array<1, elemT> : public NumericVectorWithOffset<elemT, elemT>
#ifdef STIR_USE_BOOST
                         ,
			 boost::operators<Array<1, elemT>, NumericVectorWithOffset<elemT, elemT> >,
			 boost::operators<Array<1, elemT> >,
			 boost::operators<Array<1, elemT>, elemT>
#endif
{
protected: 
  typedef NumericVectorWithOffset<elemT,elemT> base_type;
  typedef Array<1, elemT> self;

  public:  
  //! for 1D arrays, full iterators are equal to normal iterators
  typedef iterator full_iterator;

  //! for 1D arrays, full iterators are equal to normal iterators
  typedef const_iterator const_full_iterator;

  
public:
  
 
  //! default constructor: array of length 0
  inline Array();
  
  //! constructor given an IndexRange<1>
  inline explicit Array(const IndexRange<1>& range);
  		
  //! constructor from basetype
  inline Array(const base_type &il);
  
  //! constructor given first and last indices
  inline Array(const int min_index, const int max_index);

  //! virtual destructor
  inline virtual ~Array();

  /*! @name functions returning full_iterators*/
  //@{
  //! start value for iterating through all elements in the array, see full_iterator
  inline full_iterator begin_all();
  //! start value for iterating through all elements in the (const) array, see full_iterator
  inline const_full_iterator begin_all() const;
  //! start value for iterating through all elements in the array, see full_iterator
  inline const_full_iterator begin_all_const() const;
  //! end value for iterating through all elements in the array, see full_iterator
  inline full_iterator end_all();
  //! end value for iterating through all elements in the (const) array, see full_iterator
  inline const_full_iterator end_all() const;
  //! end value for iterating through all elements in the array, see full_iterator
  inline const_full_iterator end_all_const() const;
  //@}

  //! return the range of indices used
  inline IndexRange<1> get_index_range() const;

  //! return the total number of elements in this array
  inline size_t size_all() const;	

  //! Array::grow initialises new elements to 0
  inline virtual void grow(const IndexRange<1>& range);
  
  // Array::grow initialises new elements to 0
  inline virtual void grow(const int min_index, const int max_index);
  
  //! Array::resize initialises new elements to 0
  inline virtual void resize(const IndexRange<1>& range);
  
  // Array::resize initialises new elements to 0
  inline virtual void resize(const int min_index, const int max_index);
  
  //! return sum of all elements
  inline elemT sum() const;
  
  //! add up all positive elemTs in the vector
  inline elemT sum_positive() const;
		
  //! return maximum value of all elements
  inline elemT find_max() const;
  
  //! return minimum value of all elements
  inline elemT find_min() const;
    
  //! checks if the index range is 'regular' (always \c true as this is the 1D case)
  inline bool is_regular() const;
  
  //! find regular range, returns \c false if the range is not regular
  inline bool get_regular_range(
     BasicCoordinate<1, int>& min,
     BasicCoordinate<1, int>& max) const;

#ifndef STIR_USE_BOOST
  
  /* KT 31/01/2000 I had to add these functions here, although they are 
  in NumericVectorWithOffset already.
  Reason: we allow addition (and similar operations) of tensors of 
  different sizes. This implies that operator+= can call a 'grow'
  on retval. For this to work, retval should be a Array, not 
  its base_type (which happens if these function are not repeated
  in this class).
  Complicated...
  */
  //! elem by elem addition
  inline self operator+ (const base_type &iv) const;
  
  //! elem by elem subtraction
  inline self operator- (const base_type &iv) const;
  
  //! elem by elem multiplication
  inline self operator* (const base_type &iv) const;
  
  //! elem by elem division
  inline self operator/ (const base_type &iv) const;
  
  //! addition with an 'elemT'
  inline self operator+ (const elemT a) const;
  
  //! subtraction with an 'elemT'
  inline self operator- (const elemT a) const;
  
  //! multiplication with an 'elemT'
  inline self operator* (const elemT a) const;
  
  //! division with an 'elemT'
  inline self operator/ (const elemT a) const;

#endif // boost
  
    //! allow array-style access, read/write
  inline elemT&	operator[] (int i);

  //! array access, read-only
  inline const elemT&	operator[] (int i) const;
    
  //! allow array-style access giving its BasicCoordinate, read/write  
  inline const elemT& operator[](const BasicCoordinate<1,int>& c) const;

  //! array access giving its BasicCoordinate, read-only
  inline elemT& operator[](const BasicCoordinate<1,int>& c) ;    
  
 
};



void
Array<1, elemT>::resize(const int min_index, const int max_index) 
{  
  this->check_state();
  const int oldstart = this->get_min_index();
  const int oldlength = this->size();
  
  base_type::resize(min_index, max_index);
  if (oldlength == 0)
  {
    for (int i=this->get_min_index(); i<=this->get_max_index(); i++)
      this->num[i] = elemT(0);
  }
  else
  {
    for (int i=this->get_min_index(); i<oldstart && i<=this->get_max_index(); ++i)
      this->num[i] = elemT(0);
    for (int i=std::max(oldstart + oldlength, this->get_min_index()); i<=this->get_max_index(); ++i)
      this->num[i] = elemT(0);
  }
  this->check_state();  
}

void
Array<1, elemT>::resize(const IndexRange<1>& range) 
{ 
  resize(range.get_min_index(), range.get_max_index());
}

void
Array<1, elemT>::grow(const int min_index, const int max_index) 
{
  resize(min_index, max_index);
}

void
Array<1, elemT>::grow(const IndexRange<1>& range) 
{ 
  grow(range.get_min_index(), range.get_max_index());
}

Array<1, elemT>::Array()
: base_type()
{ }


Array<1, elemT>::Array(const IndexRange<1>& range)
: base_type()
{
  grow(range);
}

Array<1, elemT>::Array(const int min_index, const int max_index)
: base_type()
{
  grow(min_index, max_index);
}

Array<1, elemT>::Array(const base_type &il)
: base_type(il)
{}

Array<1, elemT>::~Array()
{}

Array<1, elemT>::full_iterator 
Array<1, elemT>::begin_all()
{
  return this->begin();
}
  
Array<1, elemT>::const_full_iterator 
Array<1, elemT>::begin_all() const
{
  return this->begin();
}

Array<1, elemT>::full_iterator 
Array<1, elemT>::end_all()
{
  return this->end();
}


Array<1, elemT>::const_full_iterator 
Array<1, elemT>::end_all() const
{
   return this->end();
}

Array<1, elemT>::const_full_iterator 
Array<1, elemT>::begin_all_const() const
{
  return this->begin();
}

Array<1, elemT>::const_full_iterator 
Array<1, elemT>::end_all_const() const
{
   return this->end();
}

IndexRange<1>
Array<1, elemT>::get_index_range() const
{
  return IndexRange<1>(this->get_min_index(), this->get_max_index());
}

size_t
Array<1, elemT>::size_all() const 
{
  return size();
}

elemT
Array<1, elemT>::sum() const 
{
  this->check_state();
  elemT acc = 0;
  for(int i=this->get_min_index(); i<=this->get_max_index(); acc+=this->num[i++])
  {}
  return acc; 
};

#ifndef __stir_Array1d_no_comparisons__
elemT
Array<1, elemT>::sum_positive() const 
{	
  this->check_state();
  elemT acc=0;
  for(int i=this->get_min_index(); i<=this->get_max_index(); i++)
  {
    if (this->num[i] > 0)
      acc += this->num[i];
  }
  return acc; 
};

                   
elemT
Array<1, elemT>::find_max() const 
{		
  this->check_state();
  if (this->size() > 0)
  {
#ifndef STIR_NO_NAMESPACES
    return *std::max_element(this->begin(), this->end());	
#else
    return *max_element(this->begin(), this->end());
#endif
  }
  else 
  { 
    // TODO return elemT::minimum or so
    return 0; 
  } 
  this->check_state();
};

elemT
Array<1, elemT>::find_min() const 
{	
  this->check_state();
  if (this->size() > 0)
  {
#ifndef STIR_NO_NAMESPACES
    return *std::min_element(this->begin(), this->end());
#else
    return *min_element(this->begin(), this->end());
#endif
  } 
  else 
  {
    // TODO return elemT::maximum or so
    return 0; 
  } 
  this->check_state();
};  

#endif // end of __stir_Array1d_no_comparisons__

bool
Array<1, elemT>::is_regular() const
{
  return true;
}

bool
Array<1, elemT>::get_regular_range(
     BasicCoordinate<1, int>& min,
     BasicCoordinate<1, int>& max) const
{
  const IndexRange<1> range = get_index_range();
  return range.get_regular_range(min,max);
}

#ifndef STIR_USE_BOOST

/* KT 31/01/2000 I had to add these functions here, although they are 
in NumericVectorWithOffset already.
Reason: we allow addition (and similar operations) of tensors of 
different sizes. This implies that operator+= can call a 'grow'
on retval. For this to work, retval should be a Array, not 
its base_type (which happens if these function are not repeated
in this class).
Complicated...
*/
 
Array<1, elemT>
Array<1, elemT>::operator+ (const base_type &iv) const 
{
  this->check_state();
  Array<1, elemT> retval(*this);
  return retval += iv;
};

Array<1, elemT>
Array<1, elemT>::operator- (const base_type &iv) const 
{
  this->check_state();
  Array<1, elemT> retval(*this);
  return retval -= iv;      
}

Array<1, elemT>
Array<1, elemT>::operator* (const base_type &iv) const
{
  this->check_state();
  Array<1, elemT> retval(*this);
  return retval *= iv;      
}


Array<1, elemT>
Array<1, elemT>::operator/ (const base_type &iv) const
{
  this->check_state();
  Array<1, elemT> retval(*this);
  return retval /= iv;      
}


Array<1, elemT>
Array<1, elemT>::operator+ (const elemT a) const 
{
  this->check_state();
  Array<1, elemT> retval(*this);
  return (retval += a);
};


Array<1, elemT>
Array<1, elemT>::operator- (const elemT a) const 
{
  this->check_state();
  Array<1, elemT> retval(*this);
  return (retval -= a);
};


Array<1, elemT>
Array<1, elemT>::operator* (const elemT a) const 
{
  this->check_state();
  Array<1, elemT> retval(*this);
  return (retval *= a);
};


Array<1, elemT>
Array<1, elemT>::operator/ (const elemT a) const 
{
  this->check_state();
  Array<1, elemT> retval(*this);
  return (retval /= a);
};


#endif // boost

const elemT& Array<1,elemT>:: operator[] (int i) const
{
   return base_type::operator[](i);
};

elemT& Array<1,elemT>:: operator[] (int i)
{
   return base_type::operator[](i);
};
  
const elemT& Array<1,elemT>:: operator[] (const BasicCoordinate<1,int>& c) const
{
  return (*this)[c[1]];
}; 
                             
elemT& Array<1,elemT>::operator[] (const BasicCoordinate<1,int>& c) 
{
  return (*this)[c[1]];
};   
 
#undef elemT

