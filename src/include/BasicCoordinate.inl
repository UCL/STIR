//
// $Id$
//
/*!
  \file 
  \ingroup buildblock
 
  \brief (inline) implementations for BasicCoordinate

  \author Kris Thielemans
  \author Alexey Zverovich
  \author PARAPET project

  $Date$
  $Revision$
*/


// for std::inner_product
#include <numeric>
// for sqrt and acos
#include <cmath>
// for equal
#include <algorithm>

#ifndef TOMO_NO_NAMESPACES
# ifndef BOOST_NO_STDC_NAMESPACE
using std::acos;
using std::sqrt;
# endif
#endif

START_NAMESPACE_TOMO

/*
  iterators
*/
template <int num_dimensions, typename coordT>
BasicCoordinate<num_dimensions, coordT>::iterator BasicCoordinate<num_dimensions, coordT>::begin() 
{
  return coords + 1; 
}

template <int num_dimensions, typename coordT>
BasicCoordinate<num_dimensions, coordT>::const_iterator 
BasicCoordinate<num_dimensions, coordT>::begin() const 
{ 
  return coords + 1; 
}

template <int num_dimensions, typename coordT>
BasicCoordinate<num_dimensions, coordT>::iterator BasicCoordinate<num_dimensions, coordT>::end() 
{
  return coords + num_dimensions + 1; 
}

template <int num_dimensions, typename coordT>
BasicCoordinate<num_dimensions, coordT>::const_iterator 
BasicCoordinate<num_dimensions, coordT>::end() const 
{ 
  return coords + num_dimensions + 1; 
}


/*
  operator[]
*/
template <int num_dimensions, typename coordT>
coordT& 
BasicCoordinate<num_dimensions, coordT>::operator[](const int d) 
{
  assert(d>0);
  assert(d<=num_dimensions);
  return coords[d]; 
}
 
template <int num_dimensions, typename coordT>
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
template <int num_dimensions, typename coordT>
bool
BasicCoordinate<num_dimensions, coordT>::operator==(const BasicCoordinate<num_dimensions, coordT>& c) const
{
  return 
#ifndef TOMO_NO_NAMESPACES
    std:: // VC needs this explicitly
#endif
    equal(begin(), end(), c.begin());
}

/*
  (numerical) assignments
*/
template <int num_dimensions, typename coordT>
BasicCoordinate<num_dimensions, coordT>&
BasicCoordinate<num_dimensions, coordT>::operator=(const BasicCoordinate<num_dimensions, coordT>& c)
{
  for (int i=1; i<=num_dimensions; i++)
    coords[i] = c[i];
  return *this;
}

template <int num_dimensions, typename coordT>
BasicCoordinate<num_dimensions, coordT>::BasicCoordinate()
{}

template <int num_dimensions, typename coordT>
BasicCoordinate<num_dimensions, coordT>::BasicCoordinate(const BasicCoordinate<num_dimensions, coordT>& c)
{
  operator=(c);
}

template <int num_dimensions, typename coordT>
BasicCoordinate<num_dimensions, coordT>& 
BasicCoordinate<num_dimensions, coordT>::
operator+=(const BasicCoordinate<num_dimensions, coordT>& c)
{
  for (int i=1; i<=num_dimensions; i++)
    coords[i] += c[i];
  return *this;
}

template <int num_dimensions, typename coordT>
BasicCoordinate<num_dimensions, coordT>& 
BasicCoordinate<num_dimensions, coordT>::
operator-=(const BasicCoordinate<num_dimensions, coordT>& c)
{
  for (int i=1; i<=num_dimensions; i++)
    coords[i] -= c[i];
  return *this;
}

template <int num_dimensions, typename coordT>
BasicCoordinate<num_dimensions, coordT>& 
BasicCoordinate<num_dimensions, coordT>::
operator*=(const BasicCoordinate<num_dimensions, coordT>& c)
{
  for (int i=1; i<=num_dimensions; i++)
    coords[i] *= c[i];
  return *this;
}

template <int num_dimensions, typename coordT>
BasicCoordinate<num_dimensions, coordT>& 
BasicCoordinate<num_dimensions, coordT>::
operator/=(const BasicCoordinate<num_dimensions, coordT>& c)
{
  for (int i=1; i<=num_dimensions; i++)
    coords[i] /= c[i];
  return *this;
}

template <int num_dimensions, typename coordT>
BasicCoordinate<num_dimensions, coordT>& 
BasicCoordinate<num_dimensions, coordT>::
operator+=(const coordT& a)
{
  for (int i=1; i<=num_dimensions; i++)
    coords[i] += a;
  return *this;
}

template <int num_dimensions, typename coordT>
BasicCoordinate<num_dimensions, coordT>& 
BasicCoordinate<num_dimensions, coordT>::
operator-=(const coordT& a)
{
  for (int i=1; i<=num_dimensions; i++)
    coords[i] -= a;
  return *this;
}

template <int num_dimensions, typename coordT>
BasicCoordinate<num_dimensions, coordT>& 
BasicCoordinate<num_dimensions, coordT>::
operator*=(const coordT& a)
{
  for (int i=1; i<=num_dimensions; i++)
    coords[i] *= a;
  return *this;
}

template <int num_dimensions, typename coordT>
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
template <int num_dimensions, typename coordT>
BasicCoordinate<num_dimensions, coordT> BasicCoordinate<num_dimensions, coordT>::operator+(const BasicCoordinate<num_dimensions, coordT>& c) const
{
  BasicCoordinate<num_dimensions, coordT> tmp(*this);
  tmp += c;
  return tmp;
}


template <int num_dimensions, typename coordT>
BasicCoordinate<num_dimensions, coordT> BasicCoordinate<num_dimensions, coordT>::operator-(const BasicCoordinate<num_dimensions, coordT>& c) const
{
  BasicCoordinate<num_dimensions, coordT> tmp(*this);
  tmp -= c;
  return tmp;
}

template <int num_dimensions, typename coordT>
BasicCoordinate<num_dimensions, coordT> BasicCoordinate<num_dimensions, coordT>::operator*(const BasicCoordinate<num_dimensions, coordT>& c) const
{
  BasicCoordinate<num_dimensions, coordT> tmp(*this);
  tmp *= c;
  return tmp;
}

template <int num_dimensions, typename coordT>
BasicCoordinate<num_dimensions, coordT> BasicCoordinate<num_dimensions, coordT>::operator/(const BasicCoordinate<num_dimensions, coordT>& c) const
{
  BasicCoordinate<num_dimensions, coordT> tmp(*this);
  tmp /= c;
  return tmp;
}

template <int num_dimensions, typename coordT>
BasicCoordinate<num_dimensions, coordT> BasicCoordinate<num_dimensions, coordT>::operator+(const coordT& a) const
{
  BasicCoordinate<num_dimensions, coordT> tmp(*this);
  tmp += a;
  return tmp;
}


template <int num_dimensions, typename coordT>
BasicCoordinate<num_dimensions, coordT> BasicCoordinate<num_dimensions, coordT>::operator-(const coordT& a) const
{
  BasicCoordinate<num_dimensions, coordT> tmp(*this);
  tmp -= a;
  return tmp;
}


template <int num_dimensions, typename coordT>
BasicCoordinate<num_dimensions, coordT> BasicCoordinate<num_dimensions, coordT>::operator*(const coordT& a) const
{
  BasicCoordinate<num_dimensions, coordT> tmp(*this);
  tmp *= a;
  return tmp;
}


template <int num_dimensions, typename coordT>
BasicCoordinate<num_dimensions, coordT> BasicCoordinate<num_dimensions, coordT>::operator/(const coordT& a) const
{
  BasicCoordinate<num_dimensions, coordT> tmp(*this);
  tmp /= a;
  return tmp;
}



/*
   External functions
*/

template <int num_dimensions, typename coordT>
coordT
inner_product (const BasicCoordinate<num_dimensions, coordT>& p1, 
	       const BasicCoordinate<num_dimensions, coordT>& p2)
{
#ifdef TOMO_NO_NAMESPACES
  return inner_product(p1.begin(), p1.end(), p2.begin(), coordT(0));
#else
  return std::inner_product(p1.begin(), p1.end(), p2.begin(), coordT(0));
#endif
}

// TODO specialise for complex coordTs if you need them
template <int num_dimensions, typename coordT>
double
norm (const BasicCoordinate<num_dimensions, coordT>& p1)
{
  return sqrt(static_cast<double>(inner_product<num_dimensions,coordT>(p1,p1)));
}

template <int num_dimensions, typename coordT>
double 
angle (const BasicCoordinate<num_dimensions, coordT>& p1, 
       const BasicCoordinate<num_dimensions, coordT>& p2)
{
  return acos(inner_product(p1,p2)/norm(p1)/ norm(p2));
}

#if !defined( __GNUC__) || !(__GNUC__ == 2 && __GNUC_MINOR__ < 9)
// only define when not gcc 2.8.1

template <int num_dimensions, typename coordT>
inline BasicCoordinate<num_dimensions+1, coordT> 
join(const coordT& a, 
     const BasicCoordinate<num_dimensions, coordT>& c)
{
  BasicCoordinate<num_dimensions+1, coordT> retval;
  
  *retval.begin() = a;
#ifdef TOMO_NO_NAMESPACES
  copy(c.begin(), c.end(), retval.begin()+1);
#else
  std::copy(c.begin(), c.end(), retval.begin()+1);
#endif
  return retval;
}

#endif // gcc 2.8.1

END_NAMESPACE_TOMO
