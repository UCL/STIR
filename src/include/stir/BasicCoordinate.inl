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

template <int num_dimensions, class coordT>
bool
BasicCoordinate<num_dimensions, coordT>::operator!=(const BasicCoordinate<num_dimensions, coordT>& c) const
{
  return !(*this==c);
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

template <int num_dimensions, class coordT>
BasicCoordinate<num_dimensions, coordT>::BasicCoordinate(const BasicCoordinate<num_dimensions, coordT>& c)
{
  operator=(c);
}

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
norm (const BasicCoordinate<num_dimensions, coordT>& p1)
{
#ifdef _MSC_VER
  return sqrt(static_cast<double>(std::inner_product(p1.begin(), p1.end(), p1.begin(), coordT(0))));
#else
	return sqrt(static_cast<double>(inner_product<num_dimensions,coordT>(p1,p1)));
#endif
}
// TODO specialise for complex coordTs if you need them
template <int num_dimensions, class coordT>
double
normsq (const BasicCoordinate<num_dimensions, coordT>& p1)
{
#ifdef _MSC_VER
  return static_cast<double>(std::inner_product(p1.begin(), p1.end(), p1.begin(), coordT(0)));
#else
  return static_cast<double>(inner_product<num_dimensions,coordT>(p1,p1));
#endif
}

template <int num_dimensions, class coordT>
double 
cos_angle (const BasicCoordinate<num_dimensions, coordT>& p1, 
           const BasicCoordinate<num_dimensions, coordT>& p2)
{
  return (inner_product(p1,p2)/sqrt(normsq(p1)/ normsq(p2)));
}

template <int num_dimensions, class coordT>
double 
angle (const BasicCoordinate<num_dimensions, coordT>& p1, 
       const BasicCoordinate<num_dimensions, coordT>& p2)
{
  return acos(cos_angle(p1,p2));
}

#if !defined( __GNUC__) || !(__GNUC__ == 2 && __GNUC_MINOR__ < 9)
// only define when not gcc 2.8.1

template <int num_dimensions, class coordT>
inline BasicCoordinate<num_dimensions+1, coordT> 
join(const coordT& a, 
     const BasicCoordinate<num_dimensions, coordT>& c)
{
  BasicCoordinate<num_dimensions+1, coordT> retval;
  
  *retval.begin() = a;
#ifdef STIR_NO_NAMESPACES
  copy(c.begin(), c.end(), retval.begin()+1);
#else
  std::copy(c.begin(), c.end(), retval.begin()+1);
#endif
  return retval;
}
 
#endif // gcc 2.8.1

template <int num_dimensions, class coordT>
inline BasicCoordinate<num_dimensions-1, coordT> 
cut_first_dimension(const BasicCoordinate<num_dimensions, coordT>& c)
{
  BasicCoordinate<num_dimensions-1, coordT> retval;
  
#ifdef STIR_NO_NAMESPACES
  copy(c.begin()+1, c.end(), retval.begin());
#else
  std::copy(c.begin()+1, c.end(), retval.begin());
#endif
  return retval;
}         

END_NAMESPACE_STIR


