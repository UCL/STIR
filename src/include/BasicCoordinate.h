#ifndef __BasicCoordinate_H__
#define __BasicCoordinate_H__
//
// $Id$: $Date$
//

/*
   This file declares BasicCoordinate<dim, coordT>: 
   a templated class for 'dim'-dimensional coordinates.
   It also declares some functions acting on BasicCoordinate objects.

   a BasicCoordinate<dim, coordT> is essentially a vector of size n, but 
   as the dimension is templated, it has better performance.

   Access to the individual coordinates is through operator[].
   Warning : Indices run from 1 to 'dim'

   History:
   1.0 (25/01/2000)
     Kris Thielemans and Alexey Zverovich
*/
#include "Tomography_common.h"

#ifndef TOMO_NO_NAMESPACES
	using std::size_t;
	using std::ptrdiff_t;
#endif

START_NAMESPACE_TOMO

template <int dim, typename coordT>
class BasicCoordinate
{ 

public:
  // typedefs for iterator support
  typedef coordT value_type;
  typedef value_type& reference;
  typedef const value_type& const_reference;
  typedef value_type* iterator;
  typedef const value_type* const_iterator;
  typedef ptrdiff_t difference_type;
  typedef size_t size_type;
	
  // default constructor. NO initialisation
  inline BasicCoordinate();
  // copy constructor
  inline BasicCoordinate(const BasicCoordinate& c);

  // virtual destructor, not implemented at the moment 
  // It's only needed whenever there is a chance that an object will be deleted
  // through a pointer to its base class and derived classes have extra
  // data members
  // virtual ~BasicCoordinate() {}

  // assignment
  inline BasicCoordinate & operator=(const BasicCoordinate& c);

  // access to elements
  inline coordT& operator[](const int d);
  inline coordT operator[](const int d) const;

  // arithmetic assignment operators
  inline BasicCoordinate & operator+= (const BasicCoordinate& c);
  inline BasicCoordinate & operator-= (const BasicCoordinate& c);
  inline BasicCoordinate & operator*= (const BasicCoordinate& c);
  inline BasicCoordinate & operator/= (const BasicCoordinate& c);

  inline BasicCoordinate & operator+= (const coordT& a);
  inline BasicCoordinate & operator-= (const coordT& a);
  inline BasicCoordinate & operator*= (const coordT& a);
  inline BasicCoordinate & operator/= (const coordT& a);

  // arithmetic operations with a BasicCoordinate, combining element by element
  inline BasicCoordinate operator+ (const BasicCoordinate& c) const;
  inline BasicCoordinate operator- (const BasicCoordinate& c) const;
  inline BasicCoordinate operator* (const BasicCoordinate& c) const;
  inline BasicCoordinate operator/ (const BasicCoordinate& c) const;

  // arithmetic operations with a coordT
  inline BasicCoordinate operator+ (const coordT& a) const;
  inline BasicCoordinate operator- (const coordT& a) const;
  inline BasicCoordinate operator* (const coordT& a) const;	      
  inline BasicCoordinate operator/ (const coordT& a) const;

  // basic iterator support
  inline iterator begin();
  inline const_iterator begin() const;
  inline iterator end();
  inline const_iterator end() const;

protected:
  // allocate 1 too many to leave space for coords[0] (which is never used)
  coordT coords[dim+1];

};


/*
  General functions on BasicCoordinate objects, like in TensorFunction.h
*/

// sum_i p1[i] * p2[i]
template <int dim, typename coordT>
inline coordT
inner_product (const BasicCoordinate<dim, coordT>& p1, 
	       const BasicCoordinate<dim, coordT>& p2);

// sqrt(inner_product(p1,p1)
template <int dim, typename coordT>
inline double
norm (const BasicCoordinate<dim, coordT>& p1);

// angle between 2 directions
template <int dim, typename coordT>
inline double 
angle (const BasicCoordinate<dim, coordT>& p1, 
       const BasicCoordinate<dim, coordT>& p2);

END_NAMESPACE_TOMO

#include "BasicCoordinate.inl"



#endif

