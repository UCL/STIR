//
// $Id$: $Date$
//

#include "VectorWithOffset.h"
#include "BasicCoordinate.h"

START_NAMESPACE_TOMO

template <int num_dimensions>
class IndexRange : public VectorWithOffset<IndexRange<num_dimensions-1> >
{
protected:
  typedef VectorWithOffset<IndexRange<num_dimensions-1> > base_type;

public:
  IndexRange<num_dimensions>()
  : base_type()
  {}
  IndexRange<num_dimensions>(const base_type& range_v)
  : base_type(range_v)
  {}

  inline IndexRange<num_dimensions>(
       const BasicCoordinate<num_dimensions, int>& min_v,
       const BasicCoordinate<num_dimensions, int>& max_v);
  
  //these are derived from VectorWithOffset
  /*
  const IndexRange<num_dimensions-1>& operator[](int i) const
  { return range[i]; }
  IndexRange<num_dimensions-1>& operator[](int i)
  { return range[i]; }
  */

};

template <int num_dimensions>
IndexRange<num_dimensions>::IndexRange(
       const BasicCoordinate<num_dimensions, int>& min_v,
       const BasicCoordinate<num_dimensions, int>& max_v)
       : base_type(min_v[1], max_v[1])
{
  BasicCoordinate<num_dimensions-1, int> new_min;
  BasicCoordinate<num_dimensions-1, int> new_max;  
  std::copy(min_v.begin()+1, min_v.end(), new_min.begin());
  std::copy(max_v.begin()+1, max_v.end(), new_max.begin());
  for(iterator iter=begin(); iter != end(); iter++)
    *iter = IndexRange<num_dimensions-1>(new_min, new_max);
}

template<>
class IndexRange<1>
{
public:
  IndexRange<1>()
    : min(0), max(0)
  {}
  IndexRange<1>(const int min_v, const int max_v)
  : min(min_v), max(max_v)
  {}

  IndexRange<1>(const BasicCoordinate<1,int>& min_v, const BasicCoordinate<1,int>& max_v)
    : min(min_v[1]), max(max_v[1])
  {}

  IndexRange<1>(const int length)
  : min(0), max(length-1)
  {}

  int get_min_index() const
  { return min;}
  int get_max_index() const
  { return max;}
  int get_length() const
  { return max-min+1; }
private:
  int min; int max;
};


template <int num_dimensions>
class RegularIndexRange
{

public:
  inline RegularIndexRange<num_dimensions>();
  // a misuse of the BasicCoordinate class: it's not a coordinate...
  inline RegularIndexRange<num_dimensions>(
       const BasicCoordinate<num_dimensions, int>& min_v,
       const BasicCoordinate<num_dimensions, int>& max_v);
  
  inline RegularIndexRange<num_dimensions-1> operator[](int i) const;
  // can't do this at the moment
  /*RegularIndexRange<num_dimensions-1>& operator[](int i);
  { }*/
  
  inline int get_min_index() const;
  inline int get_max_index() const;
  inline int get_length() const;

private:
  BasicCoordinate<num_dimensions, int> min;
  BasicCoordinate<num_dimensions, int> max;
};

template<>
class RegularIndexRange<1>
{
public:
  inline RegularIndexRange<1>();
  inline RegularIndexRange<1>(const int min, const int max);
  inline int get_min_index() const;
  inline int get_max_index() const;
  inline int get_length() const;
private:
  int min; int max;
};

//??? new syntax, but  does not work in gcc 2.95.2
//template<>
RegularIndexRange<1>::RegularIndexRange()
: min(0), max(0)
{}

//template<>
RegularIndexRange<1>::RegularIndexRange(const int min, const int max)
: min(min), max(max)
{}

//template<>
int 
RegularIndexRange<1>::get_min_index() const
{ return min;}

//template<>
int 
RegularIndexRange<1>::get_max_index() const
{ return max;}

//template<>
int 
RegularIndexRange<1>::get_length() const
{ return max-min+1; }

template <int num_dimensions>
RegularIndexRange<num_dimensions>::RegularIndexRange()
{
  std::fill(min.begin(), min.end(), 0);
  std::fill(max.begin(), max.end(), 0);
}
template <int num_dimensions>
RegularIndexRange<num_dimensions>::RegularIndexRange(
						     const BasicCoordinate<num_dimensions, int>& min_v,
						     const BasicCoordinate<num_dimensions, int>& max_v)
						     : min(min_v), max(max_v)
{}

template <int num_dimensions>
RegularIndexRange<num_dimensions-1> 
RegularIndexRange<num_dimensions>::operator[](int i) const
{ 
  assert(i>=get_min_index());
  assert(i<=get_max_index());
  
  BasicCoordinate<num_dimensions-1, int> new_min;
  BasicCoordinate<num_dimensions-1, int> new_max;  
  std::copy(min.begin()+1, min.end(), new_min.begin());
  std::copy(max.begin()+1, max.end(), new_max.begin());
  return RegularIndexRange<num_dimensions-1>(new_min, new_max);
}

template <int num_dimensions>
int
RegularIndexRange<num_dimensions>::get_min_index() const
{ return min[1];}

template <int num_dimensions>
int
RegularIndexRange<num_dimensions>::get_max_index() const
{ return max[1];}

template <int num_dimensions>
int
RegularIndexRange<num_dimensions>::get_length() const
{ return max[1]-min[1]+1; }



END_NAMESPACE_TOMO
