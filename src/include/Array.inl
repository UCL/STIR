//
// $Id$: $Date$
//

// inline implementations for Array

// include for min,max definitions
#include <algorithm>

START_NAMESPACE_TOMO

/**********************************************
 inlines for Array<num_dimensions, elemT>
 **********************************************/

template <int num_dimensions, typename elemT>
void 
Array<num_dimensions, elemT>::grow(const IndexRange<num_dimensions>& range)
{
  base_type::grow(range.get_min_index(), range.get_max_index());
  // TODO
   int i = range.get_min_index();
  for (base_type::iterator iter = begin()
    //IndexRange<num_dimensions>::iterator range_iter = range.begin(); 
   ;
  iter != end(); 
  iter++,i++)// range_iter++)
    (*iter).grow(range[i]);//*range_iter);
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
      maxval = max(num[i].find_max(), maxval);
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
      minval = min(num[i].find_min(), minval);
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
void 
Array<num_dimensions, elemT>::read_data(istream& s, const ByteOrder byte_order)
{
  check_state();
  for(int i=get_min_index(); i<=get_max_index(); i++)
    num[i].read_data(s, byte_order);
  check_state();
}

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
    return *max_element(begin(), end());	
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
    return *min_element(begin(), end());
  } 
  else 
  {
    // TODO return elemT::maximum or so
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
    return *max_element(begin(), end());	
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
    return *min_element(begin(), end());
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
on retval. For this to work, retval should be a Array, not 
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


/************************** int ************************/


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
    return *max_element(begin(), end());	
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
    return *min_element(begin(), end());
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

/********************** unsigned short ***************************/


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
    return *max_element(begin(), end());	
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
    return *min_element(begin(), end());
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

/********************** short ***************************/


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
    return *max_element(begin(), end());	
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
    return *min_element(begin(), end());
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

#endif // boost

#endif // BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION

END_NAMESPACE_TOMO
