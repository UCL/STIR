//
// $Id$: $Date$
//

/*!
  \file 
  \ingroup buildblock 
  \brief inline implementations for the Array class 

  \author Kris Thielemans (with help from Alexey Zverovich)
  \author PARAPET project

  \date    $Date$

  \version $Revision$

  For compilers that do not support partial template specialisation,
  the 1D implementations are rather tedious: full specialisations
  for a few common types. Result: lots of code repetition.
*/
// include for min,max definitions
#include <algorithm>
#ifndef TOMO_NO_NAMESPACES
using std::max;
using std::min;
#endif

START_NAMESPACE_TOMO

/**********************************************
 inlines for Array<num_dimensions, elemT>
 **********************************************/
#ifndef ARRAY4
template <int num_dimensions, typename elemT>
void 
Array<num_dimensions, elemT>::grow(const IndexRange<num_dimensions>& range)
{
  base_type::grow(range.get_min_index(), range.get_max_index());
  base_type::iterator iter = begin();
  IndexRange<num_dimensions>::const_iterator range_iter = range.begin();
  for (;
  iter != end(); 
  iter++, range_iter++)
    (*iter).grow(*range_iter);

  is_regular_range = range.is_regular();
}
#endif

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
Array<num_dimensions, elemT>::full_iterator 
Array<num_dimensions, elemT>::begin_all()
{
  if (begin() == end())
  {
    // empty array
    return full_iterator(begin(), end(), Array<num_dimensions-1, elemT>::full_iterator());
  }
  else
    return full_iterator(begin(), end(), begin()->begin_all());
}
  
#ifdef ARRAY_CONST_IT
template <int num_dimensions, typename elemT>
Array<num_dimensions, elemT>::const_full_iterator 
Array<num_dimensions, elemT>::begin_all() const
{
  if (begin() == end())
  {
    // empty array
    return const_full_iterator(begin(), end(), Array<num_dimensions-1, elemT>::const_full_iterator());
  }
  else
    return const_full_iterator(begin(), end(), begin()->begin_all());
}
#endif

template <int num_dimensions, typename elemT>
Array<num_dimensions, elemT>::full_iterator 
Array<num_dimensions, elemT>::end_all()
{
  if (begin() == end())
  {
    // empty array
    return full_iterator(begin(), end(), Array<num_dimensions-1, elemT>::full_iterator());
  }
  else
    return full_iterator(end()-1, end(), (*(end()-1)).end_all());
}

#ifdef ARRAY_CONST_IT
template <int num_dimensions, typename elemT>
Array<num_dimensions, elemT>::const_full_iterator 
Array<num_dimensions, elemT>::end_all() const
{
  if (begin() == end())
  {
    // empty array
    return const_full_iterator(begin(), end(), Array<num_dimensions-1, elemT>::const_full_iterator());
  }
  else
  { // TODO
  const_iterator last = ((end()-1));
  const_iterator really_the_end = end();
  return const_full_iterator(end()-1, really_the_end/*end()*/, /*(*(end()-1))*/last->end_all());
  }
}
#endif

#endif // ARRAY_FULL

template <int num_dimensions, class elemT>
IndexRange<num_dimensions>
Array<num_dimensions, elemT>::get_index_range() const
{
  VectorWithOffset<IndexRange<num_dimensions-1> > 
    range(get_min_index(), get_max_index());

  VectorWithOffset<IndexRange<num_dimensions-1> >::iterator range_iter =
    range.begin();
  const_iterator array_iter = begin();

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
  check_state();
  elemT acc=0;
  for(int i=get_min_index(); i<=get_max_index(); i++)
    acc += num[i].sum();
  return acc; 
}

template <int num_dimensions, typename elemT>
elemT 
Array<num_dimensions, elemT>::sum_positive() const 
{
  check_state();
  elemT acc=0;
  for(int i=get_min_index(); i<=get_max_index(); i++)
    acc += num[i].sum_positive();
  return acc; 
}

template <int num_dimensions, typename elemT>
elemT 
Array<num_dimensions, elemT>::find_max() const
{
  check_state();
  if (length > 0)
  {
    elemT maxval= num[get_min_index()].find_max();
    for(int i=get_min_index()+1; i<=get_max_index(); i++)
    {
#ifndef TOMO_NO_NAMESPACES
      maxval = std::max(num[i].find_max(), maxval);
#else
      maxval = max(num[i].find_max(), maxval);
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
  check_state();
  if (length > 0)
  {
    elemT minval= num[get_min_index()].find_min();
    for(int i=get_min_index()+1; i<=get_max_index(); i++)
    {
#ifndef TOMO_NO_NAMESPACES
      minval = std::min(num[i].find_min(), minval);
#else
      minval = min(num[i].find_min(), minval);
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
  check_state();
  for(int i=get_min_index(); i<=get_max_index();  i++)
    num[i].fill(n);
  check_state();
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
  check_state();
  for(int i=get_min_index(); i<=get_max_index(); i++)
    num[i].read_data(s, byte_order);
  check_state();
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
  check_state();
  for(int i=get_min_index(); i<=get_max_index(); i++)
    num[i].write_data(s, byte_order);
  check_state();
}			


/**********************************************
 inlines for Array<1, elemT>
 **********************************************/

#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION

template <class elemT>
void
Array<1, elemT>::grow(const int min_index, const int max_index) 
{   
  check_state();
  const int oldstart = get_min_index();
  const int oldlength = get_length();
  
  base_type::grow(min_index, max_index);
  if (oldlength == 0)
  {
    for (int i=get_min_index(); i<=get_max_index(); i++)
      num[i] = elemT(0);
  }
  else
  {
    {
      for (int i=get_min_index(); i<oldstart; i++)
	num[i] = elemT(0);
    }
    {
      for (int i=oldstart + oldlength; i<=get_max_index(); i++)
	num[i] = elemT(0);
    }
  }
  check_state();  
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
Array<1, elemT>::full_iterator 
Array<1, elemT>::begin_all()
{
  return begin();
}
  
template <typename elemT>
Array<1, elemT>::const_full_iterator 
Array<1, elemT>::begin_all() const
{
  return begin();
}

template <typename elemT>
Array<1, elemT>::full_iterator 
Array<1, elemT>::end_all()
{
  return end();
}


template <typename elemT>
Array<1, elemT>::const_full_iterator 
Array<1, elemT>::end_all() const
{
   return end();
}
  
template <typename elemT>
IndexRange<1> 
Array<1, elemT>::get_index_range() const
{
  return IndexRange<1>(get_min_index(), get_max_index());
}

template <class elemT>
elemT
Array<1, elemT>::sum() const 
{
  check_state();
  elemT acc = 0;
  for(int i=get_min_index(); i<=get_max_index(); acc+=num[i++])
  {}
  return acc; 
};


template <class elemT>
elemT
Array<1, elemT>::sum_positive() const 
{	
  check_state();
  elemT acc=0;
  for(int i=get_min_index(); i<=get_max_index(); i++)
  {
    if (num[i] > 0)
      acc += num[i];
  }
  return acc; 
};


template <class elemT>
elemT
Array<1, elemT>::find_max() const 
{		
  check_state();
  if (length > 0)
  {
#ifndef TOMO_NO_NAMESPACES
    return *std::max_element(begin(), end());	
#else
    return *max_element(begin(), end());
#endif
  }
  else 
  { 
    // TODO return elemT::minimum or so
    return 0; 
  } 
  check_state();
};


template <class elemT>
elemT
Array<1, elemT>::find_min() const 
{	
  check_state();
  if (length > 0)
  {
#ifndef TOMO_NO_NAMESPACES
    return *std::min_element(begin(), end());
#else
    return *min_element(begin(), end());
#endif
  } 
  else 
  {
    // TODO return elemT::maximum or so
    return 0; 
  } 
  check_state();
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
  IndexRange<1> range = get_index_range();
  return range.get_regular_range(min,max);
}

#ifndef TOMO_USE_BOOST

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
  check_state();
  Array<1, elemT> retval(*this);
  return retval += iv;
};

template <class elemT>
Array<1, elemT>
Array<1, elemT>::operator- (const base_type &iv) const 
{
  check_state();
  Array<1, elemT> retval(*this);
  return retval -= iv;      
}
template <class elemT>
Array<1, elemT>
Array<1, elemT>::operator* (const base_type &iv) const
{
  check_state();
  Array<1, elemT> retval(*this);
  return retval *= iv;      
}

template <class elemT>
Array<1, elemT>
Array<1, elemT>::operator/ (const base_type &iv) const
{
  check_state();
  Array<1, elemT> retval(*this);
  return retval /= iv;      
}

template <class elemT>
Array<1, elemT>
Array<1, elemT>::operator+ (const elemT a) const 
{
  check_state();
  Array<1, elemT> retval(*this);
  return (retval += a);
};

template <class elemT>
Array<1, elemT>
Array<1, elemT>::operator- (const elemT a) const 
{
  check_state();
  Array<1, elemT> retval(*this);
  return (retval -= a);
};

template <class elemT>
Array<1, elemT>
Array<1, elemT>::operator* (const elemT a) const 
{
  check_state();
  Array<1, elemT> retval(*this);
  return (retval *= a);
};

template <class elemT>
Array<1, elemT>
Array<1, elemT>::operator/ (const elemT a) const 
{
  check_state();
  Array<1, elemT> retval(*this);
  return (retval /= a);
};

#endif // boost


#else // BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION

/********************** float ***************************/

#define elemT float

void
Array<1, float>::grow(const int min_index, const int max_index) 
{  
  check_state();
  const int oldstart = get_min_index();
  const int oldlength = get_length();
  
  base_type::grow(min_index, max_index);
  if (oldlength == 0)
  {
    for (int i=get_min_index(); i<=get_max_index(); i++)
      num[i] = float(0);
  }
  else
  {
    {
      for (int i=get_min_index(); i<oldstart; i++)
	num[i] = float(0);
    }
    {
      for (int i=oldstart + oldlength; i<=get_max_index(); i++)
	num[i] = float(0);
    }
  }
  check_state();  
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
  return begin();
}
  
Array<1, elemT>::const_full_iterator 
Array<1, elemT>::begin_all() const
{
  return begin();
}

Array<1, elemT>::full_iterator 
Array<1, elemT>::end_all()
{
  return end();
}


Array<1, elemT>::const_full_iterator 
Array<1, elemT>::end_all() const
{
   return end();
}

IndexRange<1>
Array<1, float>::get_index_range() const
{
  return IndexRange<1>(get_min_index(), get_max_index());
}

float
Array<1, float>::sum() const 
{
  check_state();
  float acc = 0;
  for(int i=get_min_index(); i<=get_max_index(); acc+=num[i++])
  {}
  return acc; 
};

float
Array<1, float>::sum_positive() const 
{	
  check_state();
  float acc=0;
  for(int i=get_min_index(); i<=get_max_index(); i++)
  {
    if (num[i] > 0)
      acc += num[i];
  }
  return acc; 
};


float
Array<1, float>::find_max() const 
{		
  check_state();
  if (length > 0)
  {
#ifndef TOMO_NO_NAMESPACES
    return *std::max_element(begin(), end());	
#else
    return *max_element(begin(), end());
#endif
  }
  else 
  { 
    // TODO return float::minimum or so
    return 0; 
  } 
  check_state();
};

float
Array<1, float>::find_min() const 
{	
  check_state();
  if (length > 0)
  {
#ifndef TOMO_NO_NAMESPACES
    return *std::min_element(begin(), end());
#else
    return *min_element(begin(), end());
#endif
  } 
  else 
  {
    // TODO return float::maximum or so
    return 0; 
  } 
  check_state();
};  

#ifndef TOMO_USE_BOOST

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
  check_state();
  Array<1, float> retval(*this);
  return retval += iv;
};

Array<1, float>
Array<1, float>::operator- (const base_type &iv) const 
{
  check_state();
  Array<1, float> retval(*this);
  return retval -= iv;      
}

Array<1, float>
Array<1, float>::operator* (const base_type &iv) const
{
  check_state();
  Array<1, float> retval(*this);
  return retval *= iv;      
}


Array<1, float>
Array<1, float>::operator/ (const base_type &iv) const
{
  check_state();
  Array<1, float> retval(*this);
  return retval /= iv;      
}


Array<1, float>
Array<1, float>::operator+ (const float a) const 
{
  check_state();
  Array<1, float> retval(*this);
  return (retval += a);
};


Array<1, float>
Array<1, float>::operator- (const float a) const 
{
  check_state();
  Array<1, float> retval(*this);
  return (retval -= a);
};


Array<1, float>
Array<1, float>::operator* (const float a) const 
{
  check_state();
  Array<1, float> retval(*this);
  return (retval *= a);
};


Array<1, float>
Array<1, float>::operator/ (const float a) const 
{
  check_state();
  Array<1, float> retval(*this);
  return (retval /= a);
};

#endif // boost

#undef elemT

/************************** int ************************/

#define elemT int

void
Array<1, int>::grow(const int min_index, const int max_index) 
{   
  check_state();
  const int oldstart = get_min_index();
  const int oldlength = get_length();
  
  base_type::grow(min_index, max_index);
  if (oldlength == 0)
  {
    for (int i=get_min_index(); i<=get_max_index(); i++)
      num[i] = int(0);
  }
  else
  {
    {
      for (int i=get_min_index(); i<oldstart; i++)
	num[i] = int(0);
    }
    {
      for (int i=oldstart + oldlength; i<=get_max_index(); i++)
	num[i] = int(0);
    }
  }
  check_state();  
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
  return begin();
}
  
Array<1, elemT>::const_full_iterator 
Array<1, elemT>::begin_all() const
{
  return begin();
}

Array<1, elemT>::full_iterator 
Array<1, elemT>::end_all()
{
  return end();
}


Array<1, elemT>::const_full_iterator 
Array<1, elemT>::end_all() const
{
   return end();
}

IndexRange<1>
Array<1, int>::get_index_range() const
{
  return IndexRange<1>(get_min_index(), get_max_index());
}

int
Array<1, int>::sum() const 
{
  check_state();
  int acc = 0;
  for(int i=get_min_index(); i<=get_max_index(); acc+=num[i++])
  {}
  return acc; 
};



int
Array<1, int>::sum_positive() const 
{	
  check_state();
  int acc=0;
  for(int i=get_min_index(); i<=get_max_index(); i++)
  {
    if (num[i] > 0)
      acc += num[i];
  }
  return acc; 
};



int
Array<1, int>::find_max() const 
{		
  check_state();
  if (length > 0)
  {
#ifndef TOMO_NO_NAMESPACES
    return *std::max_element(begin(), end());	
#else
    return *max_element(begin(), end());
#endif
  }
  else 
  { 
    // TODO return int::minimum or so
    return 0; 
  } 
  check_state();
};


int
Array<1, int>::find_min() const 
{	
  check_state();
  if (length > 0)
  {
#ifndef TOMO_NO_NAMESPACES
    return *std::min_element(begin(), end());
#else
    return *min_element(begin(), end());
#endif
  } 
  else 
  {
    // TODO return int::maximum or so
    return 0; 
  } 
  check_state();
};  

#ifndef TOMO_USE_BOOST

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
  check_state();
  Array<1, int> retval(*this);
  return retval += iv;
};


Array<1, int>
Array<1, int>::operator- (const base_type &iv) const 
{
  check_state();
  Array<1, int> retval(*this);
  return retval -= iv;      
}

Array<1, int>
Array<1, int>::operator* (const base_type &iv) const
{
  check_state();
  Array<1, int> retval(*this);
  return retval *= iv;      
}


Array<1, int>
Array<1, int>::operator/ (const base_type &iv) const
{
  check_state();
  Array<1, int> retval(*this);
  return retval /= iv;      
}


Array<1, int>
Array<1, int>::operator+ (const int a) const 
{
  check_state();
  Array<1, int> retval(*this);
  return (retval += a);
};


Array<1, int>
Array<1, int>::operator- (const int a) const 
{
  check_state();
  Array<1, int> retval(*this);
  return (retval -= a);
};


Array<1, int>
Array<1, int>::operator* (const int a) const 
{
  check_state();
  Array<1, int> retval(*this);
  return (retval *= a);
};


Array<1, int>
Array<1, int>::operator/ (const int a) const 
{
  check_state();
  Array<1, int> retval(*this);
  return (retval /= a);
};

#endif // boost

#undef elemT

/********************** unsigned short ***************************/

#define elemT unsigned short

void
Array<1, unsigned short>::grow(const int min_index, const int max_index) 
{  
  check_state();
  const int oldstart = get_min_index();
  const int oldlength = get_length();
  
  base_type::grow(min_index, max_index);
  if (oldlength == 0)
  {
    for (int i=get_min_index(); i<=get_max_index(); i++)
      num[i] = unsigned short(0);
  }
  else
  {
    {
      for (int i=get_min_index(); i<oldstart; i++)
	num[i] = unsigned short(0);
    }
    {
      for (int i=oldstart + oldlength; i<=get_max_index(); i++)
	num[i] = unsigned short(0);
    }
  }
  check_state();  
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
  return begin();
}
  
Array<1, elemT>::const_full_iterator 
Array<1, elemT>::begin_all() const
{
  return begin();
}

Array<1, elemT>::full_iterator 
Array<1, elemT>::end_all()
{
  return end();
}


Array<1, elemT>::const_full_iterator 
Array<1, elemT>::end_all() const
{
   return end();
}

IndexRange<1>
Array<1, unsigned short>::get_index_range() const
{
  return IndexRange<1>(get_min_index(), get_max_index());
}

unsigned short
Array<1, unsigned short>::sum() const 
{
  check_state();
  unsigned short acc = 0;
  for(int i=get_min_index(); i<=get_max_index(); acc+=num[i++])
  {}
  return acc; 
};

unsigned short
Array<1, unsigned short>::sum_positive() const 
{	
  check_state();
  unsigned short acc=0;
  for(int i=get_min_index(); i<=get_max_index(); i++)
  {
    if (num[i] > 0)
      acc += num[i];
  }
  return acc; 
};


unsigned short
Array<1, unsigned short>::find_max() const 
{		
  check_state();
  if (length > 0)
  {
#ifndef TOMO_NO_NAMESPACES
    return *std::max_element(begin(), end());	
#else
    return *max_element(begin(), end());
#endif
  }
  else 
  { 
    // TODO return unsigned short::minimum or so
    return 0; 
  } 
  check_state();
};

unsigned short
Array<1, unsigned short>::find_min() const 
{	
  check_state();
  if (length > 0)
  {
#ifndef TOMO_NO_NAMESPACES
    return *std::min_element(begin(), end());
#else
    return *min_element(begin(), end());
#endif
  } 
  else 
  {
    // TODO return unsigned short::maximum or so
    return 0; 
  } 
  check_state();
};  

#ifndef TOMO_USE_BOOST

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
  check_state();
  Array<1, unsigned short> retval(*this);
  return retval += iv;
};

Array<1, unsigned short>
Array<1, unsigned short>::operator- (const base_type &iv) const 
{
  check_state();
  Array<1, unsigned short> retval(*this);
  return retval -= iv;      
}

Array<1, unsigned short>
Array<1, unsigned short>::operator* (const base_type &iv) const
{
  check_state();
  Array<1, unsigned short> retval(*this);
  return retval *= iv;      
}


Array<1, unsigned short>
Array<1, unsigned short>::operator/ (const base_type &iv) const
{
  check_state();
  Array<1, unsigned short> retval(*this);
  return retval /= iv;      
}


Array<1, unsigned short>
Array<1, unsigned short>::operator+ (const unsigned short a) const 
{
  check_state();
  Array<1, unsigned short> retval(*this);
  return (retval += a);
};


Array<1, unsigned short>
Array<1, unsigned short>::operator- (const unsigned short a) const 
{
  check_state();
  Array<1, unsigned short> retval(*this);
  return (retval -= a);
};


Array<1, unsigned short>
Array<1, unsigned short>::operator* (const unsigned short a) const 
{
  check_state();
  Array<1, unsigned short> retval(*this);
  return (retval *= a);
};


Array<1, unsigned short>
Array<1, unsigned short>::operator/ (const unsigned short a) const 
{
  check_state();
  Array<1, unsigned short> retval(*this);
  return (retval /= a);
};


#endif // boost

#undef elemT

/********************** short ***************************/

#define elemT short

void
Array<1, short>::grow(const int min_index, const int max_index) 
{  
  check_state();
  const int oldstart = get_min_index();
  const int oldlength = get_length();
  
  base_type::grow(min_index, max_index);
  if (oldlength == 0)
  {
    for (int i=get_min_index(); i<=get_max_index(); i++)
      num[i] = short(0);
  }
  else
  {
    {
      for (int i=get_min_index(); i<oldstart; i++)
	num[i] = short(0);
    }
    {
      for (int i=oldstart + oldlength; i<=get_max_index(); i++)
	num[i] = short(0);
    }
  }
  check_state();  
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
  return begin();
}
  
Array<1, elemT>::const_full_iterator 
Array<1, elemT>::begin_all() const
{
  return begin();
}

Array<1, elemT>::full_iterator 
Array<1, elemT>::end_all()
{
  return end();
}


Array<1, elemT>::const_full_iterator 
Array<1, elemT>::end_all() const
{
   return end();
}

IndexRange<1>
Array<1, short>::get_index_range() const
{
  return IndexRange<1>(get_min_index(), get_max_index());
}

short
Array<1, short>::sum() const 
{
  check_state();
  short acc = 0;
  for(int i=get_min_index(); i<=get_max_index(); acc+=num[i++])
  {}
  return acc; 
};

short
Array<1, short>::sum_positive() const 
{	
  check_state();
  short acc=0;
  for(int i=get_min_index(); i<=get_max_index(); i++)
  {
    if (num[i] > 0)
      acc += num[i];
  }
  return acc; 
};


short
Array<1, short>::find_max() const 
{		
  check_state();
  if (length > 0)
  {
#ifndef TOMO_NO_NAMESPACES
    return *std::max_element(begin(), end());	
#else
    return *max_element(begin(), end());
#endif
  }
  else 
  { 
    // TODO return short::minimum or so
    return 0; 
  } 
  check_state();
};

short
Array<1, short>::find_min() const 
{	
  check_state();
  if (length > 0)
  {
#ifndef TOMO_NO_NAMESPACES
    return *std::min_element(begin(), end());
#else
    return *min_element(begin(), end());
#endif
  } 
  else 
  {
    // TODO return short::maximum or so
    return 0; 
  } 
  check_state();
};  

#ifndef TOMO_USE_BOOST

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
  check_state();
  Array<1, short> retval(*this);
  return retval += iv;
};

Array<1, short>
Array<1, short>::operator- (const base_type &iv) const 
{
  check_state();
  Array<1, short> retval(*this);
  return retval -= iv;      
}

Array<1, short>
Array<1, short>::operator* (const base_type &iv) const
{
  check_state();
  Array<1, short> retval(*this);
  return retval *= iv;      
}


Array<1, short>
Array<1, short>::operator/ (const base_type &iv) const
{
  check_state();
  Array<1, short> retval(*this);
  return retval /= iv;      
}


Array<1, short>
Array<1, short>::operator+ (const short a) const 
{
  check_state();
  Array<1, short> retval(*this);
  return (retval += a);
};


Array<1, short>
Array<1, short>::operator- (const short a) const 
{
  check_state();
  Array<1, short> retval(*this);
  return (retval -= a);
};


Array<1, short>
Array<1, short>::operator* (const short a) const 
{
  check_state();
  Array<1, short> retval(*this);
  return (retval *= a);
};


Array<1, short>
Array<1, short>::operator/ (const short a) const 
{
  check_state();
  Array<1, short> retval(*this);
  return (retval /= a);
};

#endif // boost

#undef elemT

#endif // BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION

END_NAMESPACE_TOMO
