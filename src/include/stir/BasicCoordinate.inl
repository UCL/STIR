//
// $Id$
//
/*!
  \file 
  \ingroup Coordinate
 
  \brief (inline) implementations for BasicCoordinate

  \author Kris Thielemans
  \author Alexey Zverovich
  \author PARAPET project

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/

#include "stir/detail/test_if_1d.h"
// for std::inner_product
#include <numeric>
// for sqrt and acos
#include <cmath>
// for equal
#include <algorithm>

#ifndef STIR_NO_NAMESPACES
# ifndef BOOST_NO_STDC_NAMESPACE
using std::acos;
using std::sqrt;
# endif
#endif

START_NAMESPACE_STIR

/*
  iterators
*/
template <int num_dimensions, class coordT>
typename BasicCoordinate<num_dimensions, coordT>::iterator 
BasicCoordinate<num_dimensions, coordT>::begin() 
{
  return coords + 1; 
}

template <int num_dimensions, class coordT>
typename BasicCoordinate<num_dimensions, coordT>::const_iterator 
BasicCoordinate<num_dimensions, coordT>::begin() const 
{ 
  return coords + 1; 
}

template <int num_dimensions, class coordT>
typename BasicCoordinate<num_dimensions, coordT>::iterator 
BasicCoordinate<num_dimensions, coordT>::end() 
{
  return coords + num_dimensions + 1; 
}

template <int num_dimensions, class coordT>
typename BasicCoordinate<num_dimensions, coordT>::const_iterator 
BasicCoordinate<num_dimensions, coordT>::end() const 
{ 
  return coords + num_dimensions + 1; 
}


/*
  operator[]
*/
template <int num_dimensions, class coordT>
coordT& 
BasicCoordinate<num_dimensions, coordT>::operator[](const int d) 
{
  assert(d>0);
  assert(d<=num_dimensions);
  return coords[d]; 
}
 
template <int num_dimensions, class coordT>
coordT
BasicCoordinate<num_dimensions, coordT>::operator[](const int d) const
{
  assert(d>0);
  assert(d<=num_dimensions);
  return coords[d]; 
}

/*
  comparison
 */
template <int num_dimensions, class coordT>
bool
BasicCoordinate<num_dimensions, coordT>::operator==(const BasicCoordinate<num_dimensions, coordT>& c) const
{
  return 
#ifndef STIR_NO_NAMESPACES
    std:: // VC needs this explicitly
#endif
    equal(begin(), end(), c.begin());
}

/*
  (numerical) assignments
*/
template <int num_dimensions, class coordT>
BasicCoordinate<num_dimensions, coordT>&
BasicCoordinate<num_dimensions, coordT>::operator=(const BasicCoordinate<num_dimensions, coordT>& c)
{
  for (int i=1; i<=num_dimensions; i++)
    coords[i] = c[i];
  return *this;
}

template <int num_dimensions, class coordT>
BasicCoordinate<num_dimensions, coordT>::BasicCoordinate()
{}

#if 0
template <int num_dimensions, class coordT>
BasicCoordinate<num_dimensions, coordT>::BasicCoordinate(const BasicCoordinate<num_dimensions, coordT>& c)
{
  operator=(c);
}
#endif
template <int num_dimensions, class coordT>
BasicCoordinate<num_dimensions, coordT>& 
BasicCoordinate<num_dimensions, coordT>::
operator+=(const BasicCoordinate<num_dimensions, coordT>& c)
{
  for (int i=1; i<=num_dimensions; i++)
    coords[i] += c[i];
  return *this;
}

template <int num_dimensions, class coordT>
BasicCoordinate<num_dimensions, coordT>& 
BasicCoordinate<num_dimensions, coordT>::
operator-=(const BasicCoordinate<num_dimensions, coordT>& c)
{
  for (int i=1; i<=num_dimensions; i++)
    coords[i] -= c[i];
  return *this;
}

template <int num_dimensions, class coordT>
BasicCoordinate<num_dimensions, coordT>& 
BasicCoordinate<num_dimensions, coordT>::
operator*=(const BasicCoordinate<num_dimensions, coordT>& c)
{
  for (int i=1; i<=num_dimensions; i++)
    coords[i] *= c[i];
  return *this;
}

template <int num_dimensions, class coordT>
BasicCoordinate<num_dimensions, coordT>& 
BasicCoordinate<num_dimensions, coordT>::
operator/=(const BasicCoordinate<num_dimensions, coordT>& c)
{
  for (int i=1; i<=num_dimensions; i++)
    coords[i] /= c[i];
  return *this;
}

template <int num_dimensions, class coordT>
BasicCoordinate<num_dimensions, coordT>& 
BasicCoordinate<num_dimensions, coordT>::
operator+=(const coordT& a)
{
  for (int i=1; i<=num_dimensions; i++)
    coords[i] += a;
  return *this;
}

template <int num_dimensions, class coordT>
BasicCoordinate<num_dimensions, coordT>& 
BasicCoordinate<num_dimensions, coordT>::
operator-=(const coordT& a)
{
  for (int i=1; i<=num_dimensions; i++)
    coords[i] -= a;
  return *this;
}

template <int num_dimensions, class coordT>
BasicCoordinate<num_dimensions, coordT>& 
BasicCoordinate<num_dimensions, coordT>::
operator*=(const coordT& a)
{
  for (int i=1; i<=num_dimensions; i++)
    coords[i] *= a;
  return *this;
}

template <int num_dimensions, class coordT>
BasicCoordinate<num_dimensions, coordT>& 
BasicCoordinate<num_dimensions, coordT>::
operator/=(const coordT& a)
{
  for (int i=1; i<=num_dimensions; i++)
    coords[i] /= a;
  return *this;
}


/*
  numerical operators
*/
template <int num_dimensions, class coordT>
BasicCoordinate<num_dimensions, coordT> BasicCoordinate<num_dimensions, coordT>::operator+(const BasicCoordinate<num_dimensions, coordT>& c) const
{
  BasicCoordinate<num_dimensions, coordT> tmp(*this);
  tmp += c;
  return tmp;
}


template <int num_dimensions, class coordT>
BasicCoordinate<num_dimensions, coordT> BasicCoordinate<num_dimensions, coordT>::operator-(const BasicCoordinate<num_dimensions, coordT>& c) const
{
  BasicCoordinate<num_dimensions, coordT> tmp(*this);
  tmp -= c;
  return tmp;
}

template <int num_dimensions, class coordT>
BasicCoordinate<num_dimensions, coordT> BasicCoordinate<num_dimensions, coordT>::operator*(const BasicCoordinate<num_dimensions, coordT>& c) const
{
  BasicCoordinate<num_dimensions, coordT> tmp(*this);
  tmp *= c;
  return tmp;
}

template <int num_dimensions, class coordT>
BasicCoordinate<num_dimensions, coordT> BasicCoordinate<num_dimensions, coordT>::operator/(const BasicCoordinate<num_dimensions, coordT>& c) const
{
  BasicCoordinate<num_dimensions, coordT> tmp(*this);
  tmp /= c;
  return tmp;
}

template <int num_dimensions, class coordT>
BasicCoordinate<num_dimensions, coordT> BasicCoordinate<num_dimensions, coordT>::operator+(const coordT& a) const
{
  BasicCoordinate<num_dimensions, coordT> tmp(*this);
  tmp += a;
  return tmp;
}


template <int num_dimensions, class coordT>
BasicCoordinate<num_dimensions, coordT> BasicCoordinate<num_dimensions, coordT>::operator-(const coordT& a) const
{
  BasicCoordinate<num_dimensions, coordT> tmp(*this);
  tmp -= a;
  return tmp;
}


template <int num_dimensions, class coordT>
BasicCoordinate<num_dimensions, coordT> BasicCoordinate<num_dimensions, coordT>::operator*(const coordT& a) const
{
  BasicCoordinate<num_dimensions, coordT> tmp(*this);
  tmp *= a;
  return tmp;
}


template <int num_dimensions, class coordT>
BasicCoordinate<num_dimensions, coordT> BasicCoordinate<num_dimensions, coordT>::operator/(const coordT& a) const
{
  BasicCoordinate<num_dimensions, coordT> tmp(*this);
  tmp /= a;
  return tmp;
}



/*
   External functions
*/

template <int num_dimensions, class coordT>
coordT
inner_product (const BasicCoordinate<num_dimensions, coordT>& p1, 
	       const BasicCoordinate<num_dimensions, coordT>& p2)
{
#ifdef STIR_NO_NAMESPACES
  return inner_product(p1.begin(), p1.end(), p2.begin(), coordT(0));
#else
  return std::inner_product(p1.begin(), p1.end(), p2.begin(), coordT(0));
#endif
}
// TODO specialise for complex coordTs if you need them
template <int num_dimensions, class coordT>
double
norm_squared (const BasicCoordinate<num_dimensions, coordT>& p1)
{
#ifdef _MSC_VER
  return static_cast<double>(std::inner_product(p1.begin(), p1.end(), p1.begin(), coordT(0)));
#else
  return static_cast<double>(inner_product<num_dimensions,coordT>(p1,p1));
#endif
}
// TODO specialise for complex coordTs if you need them
template <int num_dimensions, class coordT>
double
norm (const BasicCoordinate<num_dimensions, coordT>& p1)
{
  return sqrt(norm_squared(p1));
}

template <int num_dimensions, class coordT>
double 
cos_angle (const BasicCoordinate<num_dimensions, coordT>& p1, 
           const BasicCoordinate<num_dimensions, coordT>& p2)
{
  return inner_product(p1,p2)/sqrt(norm_squared(p1)*norm_squared(p2)) ;
}

template <int num_dimensions, class coordT>
double 
angle (const BasicCoordinate<num_dimensions, coordT>& p1, 
       const BasicCoordinate<num_dimensions, coordT>& p2)
{
  return acos(cos_angle(p1,p2));
}

template <int num_dimensions, class coordT>
inline BasicCoordinate<num_dimensions+1, coordT> 
join(const coordT& a, 
     const BasicCoordinate<num_dimensions, coordT>& c)
{
  BasicCoordinate<num_dimensions+1, coordT> retval;
  
  *retval.begin() = a;
  std::copy(c.begin(), c.end(), retval.begin()+1);
  return retval;
}
 
template <int num_dimensions, class coordT>
BasicCoordinate<num_dimensions-1, coordT> 
cut_last_dimension(const BasicCoordinate<num_dimensions, coordT>& c)
{
  BasicCoordinate<num_dimensions-1, coordT> retval;  
  std::copy(c.begin(), c.end()-1, retval.begin());
  return retval;
}
template <int num_dimensions, class coordT>
inline BasicCoordinate<num_dimensions+1, coordT> 
join(const BasicCoordinate<num_dimensions, coordT>& c, const coordT& a)
{
  BasicCoordinate<num_dimensions+1, coordT> retval;
  
  retval[num_dimensions+1] = a;
  std::copy(c.begin(), c.end(), retval.begin());
  return retval;
}

template <int num_dimensions, class coordT>
BasicCoordinate<num_dimensions-1, coordT> 
cut_first_dimension(const BasicCoordinate<num_dimensions, coordT>& c)
{
  BasicCoordinate<num_dimensions-1, coordT> retval;
  
  std::copy(c.begin()+1, c.end(), retval.begin());
  return retval;
}         
template<int num_dimensions>
inline 
BasicCoordinate<num_dimensions,float> convert_int_to_float(const BasicCoordinate<num_dimensions,int>& cint)
{	  
  BasicCoordinate<num_dimensions,float> cfloat;
  for(int i=1;i<=num_dimensions;++i)
	  cfloat[i]=(float)cint[i];
  return cfloat;
}

// helper functions for operator<()
namespace detail
{

  template <class coordT>
  inline 
  bool 
  coordinate_less_than_help(is_1d,
			    const BasicCoordinate<1, coordT>& c1,
			    const BasicCoordinate<1, coordT>& c2)
  {
    return c1[1]<c2[1];
  }

  // specialisation for 2D, avoiding cut_first_dimension and hence potentially slow if it's not optimised away
  template <class coordT>
  inline 
  bool 
  coordinate_less_than_help(is_not_1d,
			    const BasicCoordinate<2, coordT>& c1,
			    const BasicCoordinate<2, coordT>& c2)
  {
    return 
      c1[1]<c2[1] || 
      (c1[1]==c2[1] && c1[2]<c2[2]);
  }

  // specialisation for 3D, avoiding cut_first_dimension and hence potentially slow if it's not optimised away
  template <class coordT>
  inline 
  bool 
  coordinate_less_than_help(is_not_1d,
			    const BasicCoordinate<3, coordT>& c1,
			    const BasicCoordinate<3, coordT>& c2)
  {
    return 
      c1[1]<c2[1] || 
      (c1[1]==c2[1] && 
       (c1[2]<c2[2] ||
	c1[2]==c2[2] && c1[3]<c2[3]));
  }

#if !defined(_MSC_VER) || _MSC_VER>1200
  // generic code
  template <int num_dimensions, class coordT>
  inline 
  bool 
  coordinate_less_than_help(is_not_1d,
			    const BasicCoordinate<num_dimensions, coordT>& c1,
			    const BasicCoordinate<num_dimensions, coordT>& c2)
  {
    return 
      c1[1]<c2[1] || 
      (c1[1]==c2[1] && cut_first_dimension(c1)<cut_first_dimension(c2));
  }
#else
  // VC 6.0 gets confused with the overloading, so we do 4 dimensions explicitly here
  template <class coordT>
  inline 
  bool 
  coordinate_less_than_help(is_not_1d,
			    const BasicCoordinate<4, coordT>& c1,
			    const BasicCoordinate<4, coordT>& c2)
  {
    return 
      c1[1]<c2[1] || 
      (c1[1]==c2[1] && 
       (c1[2]<c2[2] ||
	c1[2]==c2[2] && 
        (c1[3]<c2[3] ||
         c1[3]==c2[3] && c1[4]<c2[4])));
  }
#endif

} // end namespace detail

#if !defined(_MSC_VER) || _MSC_VER>1200

// generic definition
template <int num_dimensions, class coordT>
bool
BasicCoordinate<num_dimensions, coordT>::
operator<(const BasicCoordinate<num_dimensions, coordT>& c) const
{
  return detail::coordinate_less_than_help(detail::test_if_1d<num_dimensions>(),
					   *this, c);
}

#else

// VC 6.0 cannot compile the above. So we list them one by one (sigh!)

template <class coordT>
bool
inline operator<(const BasicCoordinate<1, coordT>& c1, const BasicCoordinate<1, coordT>& c) 
{
  return detail::coordinate_less_than_help(detail::is_1d(),
					   c1, c);
}
#define DEFINE_OPERATOR_LESS(num_dimensions) \
template <class coordT> \
inline bool \
operator<(const BasicCoordinate<num_dimensions, coordT>& c1,const BasicCoordinate<num_dimensions, coordT>& c)\
{ \
  return detail::coordinate_less_than_help(detail::is_not_1d(), \
					   c1, c); \
}

DEFINE_OPERATOR_LESS(2)
DEFINE_OPERATOR_LESS(3)
DEFINE_OPERATOR_LESS(4)
#undef DEFINE_OPERATOR_LESS

#endif

END_NAMESPACE_STIR


