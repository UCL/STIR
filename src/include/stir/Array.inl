//
// $Id$
//

/*!
  \file 
  \ingroup Array 
  \brief inline implementations for the Array class 

  \author Kris Thielemans (with help from Alexey Zverovich)
  \author PARAPET project

  $Date$

  $Revision$

  For compilers that do not support partial template specialisation,
  the 1D implementations are rather tedious: full specialisations
  for a few common types. Result: lots of code repetition.
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/
// include for min,max definitions
#include <algorithm>
#ifndef STIR_NO_NAMESPACES
using std::max;
using std::min;
#endif

START_NAMESPACE_STIR

/**********************************************
 inlines for Array<num_dimensions, elemT>
 **********************************************/

template <int num_dimensions, typename elemT>
void 
Array<num_dimensions, elemT>::grow(const IndexRange<num_dimensions>& range)
{
  base_type::grow(range.get_min_index(), range.get_max_index());
  typename base_type::iterator iter = base_type::begin();
  typename IndexRange<num_dimensions>::const_iterator range_iter = range.begin();
  for (;
       iter != base_type::end(); 
       iter++, range_iter++)
    (*iter).grow(*range_iter);

  is_regular_range = range.is_regular();
}

template <int num_dimensions, typename elemT>
Array<num_dimensions, elemT>::Array()
: base_type()
{}

template <int num_dimensions, typename elemT>
Array<num_dimensions, elemT>::Array(const IndexRange<num_dimensions>& range)
: base_type()
{
  grow(range);
}

template <int num_dimensions, typename elemT>
Array<num_dimensions, elemT>::Array(const base_type& t)
:  base_type(t)
{}

template <int num_dimensions, typename elemT>
Array<num_dimensions, elemT>::~Array()
{}

#ifdef ARRAY_FULL 
template <int num_dimensions, typename elemT>
typename Array<num_dimensions, elemT>::full_iterator 
Array<num_dimensions, elemT>::begin_all()
{
  if (this->begin() == this->end())
  {
    // empty array
    return full_iterator(this->begin(), this->end(), Array<num_dimensions-1, elemT>::full_iterator());
  }
  else
    return full_iterator(this->begin(), this->end(), this->begin()->begin_all());
}
  
#ifdef ARRAY_CONST_IT
template <int num_dimensions, typename elemT>
typename Array<num_dimensions, elemT>::const_full_iterator 
Array<num_dimensions, elemT>::begin_all() const
{
  if (this->begin() == this->end())
  {
    // empty array
    return const_full_iterator(this->begin(), this->end(), Array<num_dimensions-1, elemT>::const_full_iterator());
  }
  else
    return const_full_iterator(this->begin(), this->end(), this->begin()->begin_all());
}
#endif

template <int num_dimensions, typename elemT>
typename Array<num_dimensions, elemT>::full_iterator 
Array<num_dimensions, elemT>::end_all()
{
  if (this->begin() == this->end())
  {
    // empty array
    return full_iterator(this->begin(), this->end(), Array<num_dimensions-1, elemT>::full_iterator());
  }
  else
    return full_iterator(this->end()-1, this->end(), (*(this->end()-1)).end_all());
}

#ifdef ARRAY_CONST_IT
template <int num_dimensions, typename elemT>
typename Array<num_dimensions, elemT>::const_full_iterator 
Array<num_dimensions, elemT>::end_all() const
{
  if (this->begin() == this->end())
  {
    // empty array
    return const_full_iterator(this->begin(), this->end(), Array<num_dimensions-1, elemT>::const_full_iterator());
  }
  else
  { // TODO
  const_iterator last = ((this->end()-1));
  const_iterator really_the_end = this->end();
  return const_full_iterator(this->end()-1, really_the_end/*this->end()*/, /*(*(this->end()-1))*/last->end_all());
  }
}
#endif

#endif // ARRAY_FULL

template <int num_dimensions, class elemT>
IndexRange<num_dimensions>
Array<num_dimensions, elemT>::get_index_range() const
{
  VectorWithOffset<IndexRange<num_dimensions-1> > 
    range(this->get_min_index(), this->get_max_index());

  typename VectorWithOffset<IndexRange<num_dimensions-1> >::iterator range_iter =
    range.begin();
  const_iterator array_iter = this->begin();

  for (;
       range_iter != range.end();
       range_iter++, array_iter++)
  {
     *range_iter = (*array_iter).get_index_range();
  }
  return IndexRange<num_dimensions>(range);
}


template <int num_dimensions, typename elemT>
elemT 
Array<num_dimensions, elemT>::sum() const 
{
  this->check_state();
  elemT acc=0;
  for(int i=this->get_min_index(); i<=this->get_max_index(); i++)
    acc += this->num[i].sum();
  return acc; 
}

template <int num_dimensions, typename elemT>
elemT 
Array<num_dimensions, elemT>::sum_positive() const 
{
  this->check_state();
  elemT acc=0;
  for(int i=this->get_min_index(); i<=this->get_max_index(); i++)
    acc += this->num[i].sum_positive();
  return acc; 
}

template <int num_dimensions, typename elemT>
elemT 
Array<num_dimensions, elemT>::find_max() const
{
  this->check_state();
  if (length > 0)
  {
    elemT maxval= this->num[this->get_min_index()].find_max();
    for(int i=this->get_min_index()+1; i<=this->get_max_index(); i++)
    {
#ifndef STIR_NO_NAMESPACES
      maxval = std::max(this->num[i].find_max(), maxval);
#else
      maxval = max(this->num[i].find_max(), maxval);
#endif
    }
    return maxval;
  } 
  else 
  { 
    //TODO we should return elemT::minimum or something
    return 0; 
  }
}

template <int num_dimensions, typename elemT>
elemT 
Array<num_dimensions, elemT>::find_min() const
{
  this->check_state();
  if (length > 0)
  {
    elemT minval= this->num[this->get_min_index()].find_min();
    for(int i=this->get_min_index()+1; i<=this->get_max_index(); i++)
    {
#ifndef STIR_NO_NAMESPACES
      minval = std::min(this->num[i].find_min(), minval);
#else
      minval = min(this->num[i].find_min(), minval);
#endif
    }
    return minval;
  } 
  else 
  { 
    //TODO we should return elemT::maximum or something
    return 0; 
  }
}

template <int num_dimensions, typename elemT>
void 
Array<num_dimensions, elemT>::fill(const elemT &n) 
{
  this->check_state();
  for(int i=this->get_min_index(); i<=this->get_max_index();  i++)
    this->num[i].fill(n);
  this->check_state();
}

template <int num_dimensions, typename elemT>
bool
Array<num_dimensions, elemT>::is_regular() const
{
  return is_regular_range;
}

//TODO terribly inefficient at the moment
template <int num_dimensions, typename elemT>
bool
Array<num_dimensions, elemT>::get_regular_range(
     BasicCoordinate<num_dimensions, int>& min,
     BasicCoordinate<num_dimensions, int>& max) const
{
  IndexRange<num_dimensions> range = get_index_range();
  return range.get_regular_range(min,max);
}

/*! This member function reads binary data from the stream.
  \warning The stream has to be opened with ios::binary.
  \warning read_data only works properly if elemT is a 'simple' type whose objects
  can be read using \c fread.
*/
template <int num_dimensions, typename elemT>
void 
Array<num_dimensions, elemT>::read_data(istream& s, const ByteOrder byte_order)
{
  this->check_state();
  for(int i=this->get_min_index(); i<=this->get_max_index(); i++)
    this->num[i].read_data(s, byte_order);
  this->check_state();
}

/*! This member function writes binary data to the stream.
  \warning The stream has to be opened with ios::binary.
  \warning write_data only works properly if elemT is a 'simple' type whose objects
  can be read using \c fwrite.
*/
template <int num_dimensions, typename elemT>
void 
Array<num_dimensions, elemT>::write_data(ostream& s, const ByteOrder byte_order) const
{
  this->check_state();
  for(int i=this->get_min_index(); i<=this->get_max_index(); i++)
    this->num[i].write_data(s, byte_order);
  this->check_state();
}		

template <int num_dimension, typename elemT>
Array<num_dimension-1,elemT>& 
Array<num_dimension,elemT>::operator[](int i)
{
  return base_type::operator[](i);                                   
}                                                  

template <int num_dimension, typename elemT>
const Array<num_dimension-1,elemT>& 
Array<num_dimension,elemT>::operator[](int i) const 
{ 
  return base_type::operator[](i);
}      
template <int num_dimensions, typename elemT>
elemT&
Array<num_dimensions,elemT>::operator[](const BasicCoordinate<num_dimensions,int> &c) 
{
  return (*this)[c[1]][cut_first_dimension(c)]; 
}			
template <int num_dimensions, typename elemT>
const elemT&
Array<num_dimensions,elemT>::operator[](const BasicCoordinate<num_dimensions,int> &c) const
{ 
  return (*this)[c[1]][cut_first_dimension(c)] ; 
}				    

/**********************************************
 inlines for Array<1, elemT>
 **********************************************/

#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION

template <class elemT>
void
Array<1, elemT>::grow(const int min_index, const int max_index) 
{   
  this->check_state();
  const int oldstart = this->get_min_index();
  const int oldlength = get_length();
  
  base_type::grow(min_index, max_index);
  if (oldlength == 0)
  {
    for (int i=this->get_min_index(); i<=this->get_max_index(); i++)
      this->num[i] = elemT(0);
  }
  else
  {
    {
      for (int i=this->get_min_index(); i<oldstart; i++)
	this->num[i] = elemT(0);
    }
    {
      for (int i=oldstart + oldlength; i<=this->get_max_index(); i++)
	this->num[i] = elemT(0);
    }
  }
  this->check_state();  
}

template <class elemT>
void
Array<1, elemT>::grow(const IndexRange<1>& range) 
{ 
  grow(range.get_min_index(), range.get_max_index());
}


template <class elemT>
Array<1, elemT>::Array()
: base_type()
{ }

template <class elemT>
Array<1, elemT>::Array(const IndexRange<1>& range)
: base_type()
{
  grow(range);
}

template <class elemT>
Array<1, elemT>::Array(const int min_index, const int max_index)
: base_type()
{
  grow(min_index, max_index);
}


template <class elemT>
Array<1, elemT>::Array(const base_type &il)
: base_type(il)
{}

template <typename elemT>
Array<1, elemT>::~Array()
{}

template <typename elemT>
typename Array<1, elemT>::full_iterator 
Array<1, elemT>::begin_all()
{
  return this->begin();
}
  
template <typename elemT>
typename Array<1, elemT>::const_full_iterator 
Array<1, elemT>::begin_all() const
{
  return this->begin();
}

template <typename elemT>
typename Array<1, elemT>::full_iterator 
Array<1, elemT>::end_all()
{
  return this->end();
}


template <typename elemT>
typename Array<1, elemT>::const_full_iterator 
Array<1, elemT>::end_all() const
{
   return this->end();
}
  
template <typename elemT>
IndexRange<1> 
Array<1, elemT>::get_index_range() const
{
  return IndexRange<1>(this->get_min_index(), this->get_max_index());
}

template <class elemT>
elemT
Array<1, elemT>::sum() const 
{
  this->check_state();
  elemT acc = 0;
  for(int i=this->get_min_index(); i<=this->get_max_index(); acc+=this->num[i++])
  {}
  return acc; 
};


template <class elemT>
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


template <class elemT>
elemT
Array<1, elemT>::find_max() const 
{		
  this->check_state();
  if (length > 0)
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


template <class elemT>
elemT
Array<1, elemT>::find_min() const 
{	
  this->check_state();
  if (length > 0)
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

template <typename elemT>
bool
Array<1, elemT>::is_regular() const
{
  return true;
}

template <typename elemT>
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
template <class elemT> 
Array<1, elemT>
Array<1, elemT>::operator+ (const base_type &iv) const 
{
  this->check_state();
  Array<1, elemT> retval(*this);
  return retval += iv;
};

template <class elemT>
Array<1, elemT>
Array<1, elemT>::operator- (const base_type &iv) const 
{
  this->check_state();
  Array<1, elemT> retval(*this);
  return retval -= iv;      
}
template <class elemT>
Array<1, elemT>
Array<1, elemT>::operator* (const base_type &iv) const
{
  this->check_state();
  Array<1, elemT> retval(*this);
  return retval *= iv;      
}

template <class elemT>
Array<1, elemT>
Array<1, elemT>::operator/ (const base_type &iv) const
{
  this->check_state();
  Array<1, elemT> retval(*this);
  return retval /= iv;      
}

template <class elemT>
Array<1, elemT>
Array<1, elemT>::operator+ (const elemT a) const 
{
  this->check_state();
  Array<1, elemT> retval(*this);
  return (retval += a);
};

template <class elemT>
Array<1, elemT>
Array<1, elemT>::operator- (const elemT a) const 
{
  this->check_state();
  Array<1, elemT> retval(*this);
  return (retval -= a);
};

template <class elemT>
Array<1, elemT>
Array<1, elemT>::operator* (const elemT a) const 
{
  this->check_state();
  Array<1, elemT> retval(*this);
  return (retval *= a);
};

template <class elemT>
Array<1, elemT>
Array<1, elemT>::operator/ (const elemT a) const 
{
  this->check_state();
  Array<1, elemT> retval(*this);
  return (retval /= a);
};  

template <typename elemT>    
const elemT& Array<1,elemT>:: operator[] (int i) const
{
   return base_type::operator[](i);
};

template <typename elemT>    
elemT& Array<1,elemT>:: operator[] (int i)
{
   return base_type::operator[](i);
};

template <typename elemT>    
const elemT& Array<1,elemT>:: operator[] (const BasicCoordinate<1,int>& c) const
{
  return (*this)[c[1]];
}; 
                             
template <typename elemT>    
elemT& Array<1,elemT>::operator[] (const BasicCoordinate<1,int>& c) 
{
  return (*this)[c[1]];
};   
               

#endif // boost   

#else // BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION

/********************** float ***************************/

#define elemT float

void
Array<1, float>::grow(const int min_index, const int max_index) 
{  
  this->check_state();
  const int oldstart = this->get_min_index();
  const int oldlength = get_length();
  
  base_type::grow(min_index, max_index);
  if (oldlength == 0)
  {
    for (int i=this->get_min_index(); i<=this->get_max_index(); i++)
      this->num[i] = float(0);
  }
  else
  {
    {
      for (int i=this->get_min_index(); i<oldstart; i++)
	this->num[i] = float(0);
    }
    {
      for (int i=oldstart + oldlength; i<=this->get_max_index(); i++)
	this->num[i] = float(0);
    }
  }
  this->check_state();  
}

void
Array<1, float>::grow(const IndexRange<1>& range) 
{ 
  grow(range.get_min_index(), range.get_max_index());
}

Array<1, float>::Array()
: base_type()
{ }


Array<1, float>::Array(const IndexRange<1>& range)
: base_type()
{
  grow(range);
}

Array<1, float>::Array(const int min_index, const int max_index)
: base_type()
{
  grow(min_index, max_index);
}

Array<1, float>::Array(const base_type &il)
: base_type(il)
{}

Array<1, float>::~Array()
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

IndexRange<1>
Array<1, float>::get_index_range() const
{
  return IndexRange<1>(this->get_min_index(), this->get_max_index());
}

float
Array<1, float>::sum() const 
{
  this->check_state();
  float acc = 0;
  for(int i=this->get_min_index(); i<=this->get_max_index(); acc+=this->num[i++])
  {}
  return acc; 
};

float
Array<1, float>::sum_positive() const 
{	
  this->check_state();
  float acc=0;
  for(int i=this->get_min_index(); i<=this->get_max_index(); i++)
  {
    if (this->num[i] > 0)
      acc += this->num[i];
  }
  return acc; 
};


float
Array<1, float>::find_max() const 
{		
  this->check_state();
  if (length > 0)
  {
#ifndef STIR_NO_NAMESPACES
    return *std::max_element(this->begin(), this->end());	
#else
    return *max_element(this->begin(), this->end());
#endif
  }
  else 
  { 
    // TODO return float::minimum or so
    return 0; 
  } 
  this->check_state();
};

float
Array<1, float>::find_min() const 
{	
  this->check_state();
  if (length > 0)
  {
#ifndef STIR_NO_NAMESPACES
    return *std::min_element(this->begin(), this->end());
#else
    return *min_element(this->begin(), this->end());
#endif
  } 
  else 
  {
    // TODO return float::maximum or so
    return 0; 
  } 
  this->check_state();
};  

#ifndef STIR_USE_BOOST

/* KT 31/01/2000 I had to add these functions here, although they are 
in NumericVectorWithOffset already.
Reason: we allow addition (and similar operations) of tensors of 
different sizes. This implies that operator+= can call a 'grow'
on retval. For this to work, retval should be an Array, not 
its base_type (which happens if these function are not repeated
in this class).
Complicated...
*/
 
Array<1, float>
Array<1, float>::operator+ (const base_type &iv) const 
{
  this->check_state();
  Array<1, float> retval(*this);
  return retval += iv;
};

Array<1, float>
Array<1, float>::operator- (const base_type &iv) const 
{
  this->check_state();
  Array<1, float> retval(*this);
  return retval -= iv;      
}

Array<1, float>
Array<1, float>::operator* (const base_type &iv) const
{
  this->check_state();
  Array<1, float> retval(*this);
  return retval *= iv;      
}


Array<1, float>
Array<1, float>::operator/ (const base_type &iv) const
{
  this->check_state();
  Array<1, float> retval(*this);
  return retval /= iv;      
}


Array<1, float>
Array<1, float>::operator+ (const float a) const 
{
  this->check_state();
  Array<1, float> retval(*this);
  return (retval += a);
};


Array<1, float>
Array<1, float>::operator- (const float a) const 
{
  this->check_state();
  Array<1, float> retval(*this);
  return (retval -= a);
};


Array<1, float>
Array<1, float>::operator* (const float a) const 
{
  this->check_state();
  Array<1, float> retval(*this);
  return (retval *= a);
};


Array<1, float>
Array<1, float>::operator/ (const float a) const 
{
  this->check_state();
  Array<1, float> retval(*this);
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

/************************** int ************************/

#define elemT int

void
Array<1, int>::grow(const int min_index, const int max_index) 
{   
  this->check_state();
  const int oldstart = this->get_min_index();
  const int oldlength = get_length();
  
  base_type::grow(min_index, max_index);
  if (oldlength == 0)
  {
    for (int i=this->get_min_index(); i<=this->get_max_index(); i++)
      this->num[i] = int(0);
  }
  else
  {
    {
      for (int i=this->get_min_index(); i<oldstart; i++)
	this->num[i] = int(0);
    }
    {
      for (int i=oldstart + oldlength; i<=this->get_max_index(); i++)
	this->num[i] = int(0);
    }
  }
  this->check_state();  
}

void
Array<1, int>::grow(const IndexRange<1>& range) 
{ 
  grow(range.get_min_index(), range.get_max_index());
}

Array<1, int>::Array()
: base_type()
{}

Array<1, int>::Array(const IndexRange<1>& range)
: base_type()
{
  grow(range);
}

Array<1, int>::Array(const int min_index, const int max_index)
: base_type()
{
  grow(min_index, max_index);
}

Array<1, int>::Array(const base_type &il)
: base_type(il)
{}

Array<1, int>::~Array()
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

IndexRange<1>
Array<1, int>::get_index_range() const
{
  return IndexRange<1>(this->get_min_index(), this->get_max_index());
}

int
Array<1, int>::sum() const 
{
  this->check_state();
  int acc = 0;
  for(int i=this->get_min_index(); i<=this->get_max_index(); acc+=this->num[i++])
  {}
  return acc; 
};



int
Array<1, int>::sum_positive() const 
{	
  this->check_state();
  int acc=0;
  for(int i=this->get_min_index(); i<=this->get_max_index(); i++)
  {
    if (this->num[i] > 0)
      acc += this->num[i];
  }
  return acc; 
};



int
Array<1, int>::find_max() const 
{		
  this->check_state();
  if (length > 0)
  {
#ifndef STIR_NO_NAMESPACES
    return *std::max_element(this->begin(), this->end());	
#else
    return *max_element(this->begin(), this->end());
#endif
  }
  else 
  { 
    // TODO return int::minimum or so
    return 0; 
  } 
  this->check_state();
};


int
Array<1, int>::find_min() const 
{	
  this->check_state();
  if (length > 0)
  {
#ifndef STIR_NO_NAMESPACES
    return *std::min_element(this->begin(), this->end());
#else
    return *min_element(this->begin(), this->end());
#endif
  } 
  else 
  {
    // TODO return int::maximum or so
    return 0; 
  } 
  this->check_state();
};  

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
 
Array<1, int>
Array<1, int>::operator+ (const base_type &iv) const 
{
  this->check_state();
  Array<1, int> retval(*this);
  return retval += iv;
};


Array<1, int>
Array<1, int>::operator- (const base_type &iv) const 
{
  this->check_state();
  Array<1, int> retval(*this);
  return retval -= iv;      
}

Array<1, int>
Array<1, int>::operator* (const base_type &iv) const
{
  this->check_state();
  Array<1, int> retval(*this);
  return retval *= iv;      
}


Array<1, int>
Array<1, int>::operator/ (const base_type &iv) const
{
  this->check_state();
  Array<1, int> retval(*this);
  return retval /= iv;      
}


Array<1, int>
Array<1, int>::operator+ (const int a) const 
{
  this->check_state();
  Array<1, int> retval(*this);
  return (retval += a);
};


Array<1, int>
Array<1, int>::operator- (const int a) const 
{
  this->check_state();
  Array<1, int> retval(*this);
  return (retval -= a);
};


Array<1, int>
Array<1, int>::operator* (const int a) const 
{
  this->check_state();
  Array<1, int> retval(*this);
  return (retval *= a);
};


Array<1, int>
Array<1, int>::operator/ (const int a) const 
{
  this->check_state();
  Array<1, int> retval(*this);
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

/********************** unsigned short ***************************/

#define elemT unsigned short

void
Array<1, unsigned short>::grow(const int min_index, const int max_index) 
{  
  this->check_state();
  const int oldstart = this->get_min_index();
  const int oldlength = get_length();
  
  base_type::grow(min_index, max_index);
  if (oldlength == 0)
  {
    for (int i=this->get_min_index(); i<=this->get_max_index(); i++)
      this->num[i] = unsigned short(0);
  }
  else
  {
    {
      for (int i=this->get_min_index(); i<oldstart; i++)
	this->num[i] = unsigned short(0);
    }
    {
      for (int i=oldstart + oldlength; i<=this->get_max_index(); i++)
	this->num[i] = unsigned short(0);
    }
  }
  this->check_state();  
}

void
Array<1, unsigned short>::grow(const IndexRange<1>& range) 
{ 
  grow(range.get_min_index(), range.get_max_index());
}

Array<1, unsigned short>::Array()
: base_type()
{ }


Array<1, unsigned short>::Array(const IndexRange<1>& range)
: base_type()
{
  grow(range);
}

Array<1, unsigned short>::Array(const int min_index, const int max_index)
: base_type()
{
  grow(min_index, max_index);
}

Array<1, unsigned short>::Array(const base_type &il)
: base_type(il)
{}

Array<1, unsigned short>::~Array()
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

IndexRange<1>
Array<1, unsigned short>::get_index_range() const
{
  return IndexRange<1>(this->get_min_index(), this->get_max_index());
}

unsigned short
Array<1, unsigned short>::sum() const 
{
  this->check_state();
  unsigned short acc = 0;
  for(int i=this->get_min_index(); i<=this->get_max_index(); acc+=this->num[i++])
  {}
  return acc; 
};

unsigned short
Array<1, unsigned short>::sum_positive() const 
{	
  this->check_state();
  unsigned short acc=0;
  for(int i=this->get_min_index(); i<=this->get_max_index(); i++)
  {
    if (this->num[i] > 0)
      acc += this->num[i];
  }
  return acc; 
};


unsigned short
Array<1, unsigned short>::find_max() const 
{		
  this->check_state();
  if (length > 0)
  {
#ifndef STIR_NO_NAMESPACES
    return *std::max_element(this->begin(), this->end());	
#else
    return *max_element(this->begin(), this->end());
#endif
  }
  else 
  { 
    // TODO return unsigned short::minimum or so
    return 0; 
  } 
  this->check_state();
};

unsigned short
Array<1, unsigned short>::find_min() const 
{	
  this->check_state();
  if (length > 0)
  {
#ifndef STIR_NO_NAMESPACES
    return *std::min_element(this->begin(), this->end());
#else
    return *min_element(this->begin(), this->end());
#endif
  } 
  else 
  {
    // TODO return unsigned short::maximum or so
    return 0; 
  } 
  this->check_state();
};  

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
 
Array<1, unsigned short>
Array<1, unsigned short>::operator+ (const base_type &iv) const 
{
  this->check_state();
  Array<1, unsigned short> retval(*this);
  return retval += iv;
};

Array<1, unsigned short>
Array<1, unsigned short>::operator- (const base_type &iv) const 
{
  this->check_state();
  Array<1, unsigned short> retval(*this);
  return retval -= iv;      
}

Array<1, unsigned short>
Array<1, unsigned short>::operator* (const base_type &iv) const
{
  this->check_state();
  Array<1, unsigned short> retval(*this);
  return retval *= iv;      
}


Array<1, unsigned short>
Array<1, unsigned short>::operator/ (const base_type &iv) const
{
  this->check_state();
  Array<1, unsigned short> retval(*this);
  return retval /= iv;      
}


Array<1, unsigned short>
Array<1, unsigned short>::operator+ (const unsigned short a) const 
{
  this->check_state();
  Array<1, unsigned short> retval(*this);
  return (retval += a);
};


Array<1, unsigned short>
Array<1, unsigned short>::operator- (const unsigned short a) const 
{
  this->check_state();
  Array<1, unsigned short> retval(*this);
  return (retval -= a);
};


Array<1, unsigned short>
Array<1, unsigned short>::operator* (const unsigned short a) const 
{
  this->check_state();
  Array<1, unsigned short> retval(*this);
  return (retval *= a);
};


Array<1, unsigned short>
Array<1, unsigned short>::operator/ (const unsigned short a) const 
{
  this->check_state();
  Array<1, unsigned short> retval(*this);
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

/********************** short ***************************/

#define elemT short

void
Array<1, short>::grow(const int min_index, const int max_index) 
{  
  this->check_state();
  const int oldstart = this->get_min_index();
  const int oldlength = get_length();
  
  base_type::grow(min_index, max_index);
  if (oldlength == 0)
  {
    for (int i=this->get_min_index(); i<=this->get_max_index(); i++)
      this->num[i] = short(0);
  }
  else
  {
    {
      for (int i=this->get_min_index(); i<oldstart; i++)
	this->num[i] = short(0);
    }
    {
      for (int i=oldstart + oldlength; i<=this->get_max_index(); i++)
	this->num[i] = short(0);
    }
  }
  this->check_state();  
}

void
Array<1, short>::grow(const IndexRange<1>& range) 
{ 
  grow(range.get_min_index(), range.get_max_index());
}

Array<1, short>::Array()
: base_type()
{ }


Array<1, short>::Array(const IndexRange<1>& range)
: base_type()
{
  grow(range);
}

Array<1, short>::Array(const int min_index, const int max_index)
: base_type()
{
  grow(min_index, max_index);
}

Array<1, short>::Array(const base_type &il)
: base_type(il)
{}

Array<1, short>::~Array()
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

IndexRange<1>
Array<1, short>::get_index_range() const
{
  return IndexRange<1>(this->get_min_index(), this->get_max_index());
}

short
Array<1, short>::sum() const 
{
  this->check_state();
  short acc = 0;
  for(int i=this->get_min_index(); i<=this->get_max_index(); acc+=this->num[i++])
  {}
  return acc; 
};

short
Array<1, short>::sum_positive() const 
{	
  this->check_state();
  short acc=0;
  for(int i=this->get_min_index(); i<=this->get_max_index(); i++)
  {
    if (this->num[i] > 0)
      acc += this->num[i];
  }
  return acc; 
};

                   
short
Array<1, short>::find_max() const 
{		
  this->check_state();
  if (length > 0)
  {
#ifndef STIR_NO_NAMESPACES
    return *std::max_element(this->begin(), this->end());	
#else
    return *max_element(this->begin(), this->end());
#endif
  }
  else 
  { 
    // TODO return short::minimum or so
    return 0; 
  } 
  this->check_state();
};

short
Array<1, short>::find_min() const 
{	
  this->check_state();
  if (length > 0)
  {
#ifndef STIR_NO_NAMESPACES
    return *std::min_element(this->begin(), this->end());
#else
    return *min_element(this->begin(), this->end());
#endif
  } 
  else 
  {
    // TODO return short::maximum or so
    return 0; 
  } 
  this->check_state();
};  

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
 
Array<1, short>
Array<1, short>::operator+ (const base_type &iv) const 
{
  this->check_state();
  Array<1, short> retval(*this);
  return retval += iv;
};

Array<1, short>
Array<1, short>::operator- (const base_type &iv) const 
{
  this->check_state();
  Array<1, short> retval(*this);
  return retval -= iv;      
}

Array<1, short>
Array<1, short>::operator* (const base_type &iv) const
{
  this->check_state();
  Array<1, short> retval(*this);
  return retval *= iv;      
}


Array<1, short>
Array<1, short>::operator/ (const base_type &iv) const
{
  this->check_state();
  Array<1, short> retval(*this);
  return retval /= iv;      
}


Array<1, short>
Array<1, short>::operator+ (const short a) const 
{
  this->check_state();
  Array<1, short> retval(*this);
  return (retval += a);
};


Array<1, short>
Array<1, short>::operator- (const short a) const 
{
  this->check_state();
  Array<1, short> retval(*this);
  return (retval -= a);
};


Array<1, short>
Array<1, short>::operator* (const short a) const 
{
  this->check_state();
  Array<1, short> retval(*this);
  return (retval *= a);
};


Array<1, short>
Array<1, short>::operator/ (const short a) const 
{
  this->check_state();
  Array<1, short> retval(*this);
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

#endif // BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION

END_NAMESPACE_STIR
