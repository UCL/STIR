//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2009-07-08, Hammersmith Imanet Ltd
    Copyright (C) 2011-07-01 - 2011, Kris Thielemans
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
#ifndef __stir_DiscretisedDensity_H__
#define __stir_DiscretisedDensity_H__

/*!
  \file 
  \ingroup densitydata 
  \brief defines the stir::DiscretisedDensity class 

  \author Sanida Mustafovic 
  \author Kris Thielemans 
  \author (help from Alexey Zverovich)
  \author PARAPET project



*/

#include "stir/CartesianCoordinate3D.h"
#include "stir/Array.h"
#include "stir/shared_ptr.h"
#include <string>

#ifndef STIR_NO_NAMESPACES
using std::string;
#endif

START_NAMESPACE_STIR

/*!
  \ingroup densitydata
  \brief This abstract class is the basis for all image representations.
  
  This class is templated with the number of dimensions (should be 1, 2 or 3) 
  and the type of the data.

  It defines functionality common to all discretised densities: the
  data structure itself (Array) and an origin.
 
  \warning The origin is always a CartesianCoordinate3D<float>, 
  independent of what coordinate system (or even dimension) this
  class represents. Similarly, the functions that translate from
  indices to physical coordinates assume that the later is 3D.

  Iterative algorithms generally assume that the activity density can
  be discretised in some way. That is, the continuous density can be
  approximated by having a linear combination of some
  basis-functions. The reconstruction problem will try to estimate the
  coefficients \f$\lambda_{ijk}\f$ of the discretised density
 
  \f[ \sum_{ijk} \lambda_{ijk} b_{ijk}({\bar x}) \f]

  The base class corresponding to this kind of data is
  DiscretisedDensity.  We assume that the set of basisfunctions can be
  characterised by 3 indices (ijk) such that i runs over a range of
  integers i1..i2, j runs over a similar range that can however depend
  on i, and k runs over a similar range that can depend on i and
  j. This concept of ranges is embodied in the IndexRange
  class. Multi-dimensional arrays which have such ranges are encoded
  by the Array class. This forms the data structure for the set of
  coefficients of the basisfunctions, hence DiscretisedDensity is
  derived from the Array class.  

  In most useful cases, the basisfunctions will be translations of a
  single function b(x) (although scaling etc could occur depending on
  ijk). This means that the discretisation has a certain grid,
  corresponding to the centre of the basisfunctions. This structure is
  the next level in the image hierarchy. Currently we have the class
  DiscretisedDensityOnCartesianGrid to implement the case where the
  grid is formed by an orthogonal set of vectors. Another case would
  be e.g. DiscretisedDensityOnCylindricalGrid, but we have not
  implemented this yet.  

  The next level in the hierarchy is then finally the specification of
  the basis functions themselves. We currently have only voxels and
  pixels, but another useful case would be to use Kaiser-Bessel
  functions (so called Blobs). This leads us to the image
  hierarchy as shown in the class diagram.

*/

template<int num_dimensions, typename elemT>
class DiscretisedDensity : public Array<num_dimensions,elemT>

{ 
#ifdef SWIG
  // work-around swig problem. It gets confused when using a private (or protected)
  // typedef in a definition of a public typedef/member
 public:
#else
 private: 
#endif  
  typedef Array<num_dimensions,elemT> base_type;
  typedef DiscretisedDensity<num_dimensions,elemT> self_type;
public:
  //! A static member to read an image from file
  static DiscretisedDensity * read_from_file(const string& filename);

  //! Construct an empty DiscretisedDensity
  inline DiscretisedDensity();
  
  //! Construct DiscretisedDensity of a given range of indices & origin
  inline DiscretisedDensity(const IndexRange<num_dimensions>& range,
    const CartesianCoordinate3D<float>& origin);	
  
  //! Return the origin 
  inline const CartesianCoordinate3D<float>& get_origin()  const;

  //! Set the origin
  inline void set_origin(const CartesianCoordinate3D<float> &origin);

  //! \name Translation between indices and physical coordinates
  /*! We distinguish between physical coordinates, relative coordinates (which are
    physical coordinates relative to the origin) and index coordinates (which run
    over the index range (but are allowed to have float values).

    This class provides 3-way conversion functions. The derived classes have to implement
    the actual conversion between relative and index coordinates.
  */
  //@{
  //! Return the coordinates of the centre of the basis-function corresponding to \c indices.
  /*! The return value is in the same coordinate system as get_origin().
      Implemented as 
      \code
      get_relative_coordinates_for_indices(indices)+get_origin()
      \endcode
  */
  inline 
    CartesianCoordinate3D<float>
    get_physical_coordinates_for_indices(const BasicCoordinate<num_dimensions,int>& indices) const;

  //! Return the coordinates of the centre of the basis-function corresponding to non-integer coordinate in 'index' coordinates.
  /*! \see get_physical_coordinates_for_indices(const BasicCoordinate<num_dimensions,int>&)
   */    
  inline 
    CartesianCoordinate3D<float>
    get_physical_coordinates_for_indices(const BasicCoordinate<num_dimensions,float>& indices) const;

  //! Return the relative coordinates of the centre of the basis-function corresponding to \c indices.
  /*! Implementation uses actual_get_relative_coordinates_for_indices
  */
  inline
    CartesianCoordinate3D<float>
    get_relative_coordinates_for_indices(const BasicCoordinate<num_dimensions,int>& indices) const;

  //! Return the relative coordinates of the centre of the basis-function corresponding to the non-integer coordinates in 'index' coordinates.
  /*! The return value is relative to the origin.

   Implementation uses actual_get_relative_coordinates_for_indices
      \see get_physical_coordinates_for_indices()
  */
  inline
    CartesianCoordinate3D<float>
    get_relative_coordinates_for_indices(const BasicCoordinate<num_dimensions,float>& indices) const;

  //! Return the indices of the basis-function closest to the given point.
  /*! The input argument should be in the same coordinate system as get_origin().
      Implemented as 
      \code
      get_indices_closest_to_relative_coordinates(coords-get_origin())
      \endcode
  */
  inline 
    BasicCoordinate<num_dimensions,int>
    get_indices_closest_to_physical_coordinates(const CartesianCoordinate3D<float>& coords) const;

  //! Return the indices of the basis-function closest to the given point.
  /*! The input argument should be in 'physical' coordinates relative to the origin.
      Implementation uses
      stir::round on the result of get_index_coordinates_for_relative_coordinates.
  */
  inline
    BasicCoordinate<num_dimensions,int>
    get_indices_closest_to_relative_coordinates(const CartesianCoordinate3D<float>& coords) const;

  //! Return the indices of the basis-function closest to the given point.
  /*! The input argument should be in 'physical' coordinates.
      Implementation uses get_index_coordinates_for_relative_coordinates.
  */
  inline
    BasicCoordinate<num_dimensions,float>
    get_index_coordinates_for_physical_coordinates(const CartesianCoordinate3D<float>& coords) const;

  //! Return the index-coordinates of the basis-function closest to the given point.
  /*! The input argument should be in 'physical' coordinates relative to the origin.
    Implementation uses actual_get_index_coordinates_for_relative_coordinates.
  */
  inline
    BasicCoordinate<num_dimensions,float>
    get_index_coordinates_for_relative_coordinates(const CartesianCoordinate3D<float>& coords) const;

  //@}

  //! Allocate a new DiscretisedDensity object with same characteristics as the current one.
  virtual DiscretisedDensity<num_dimensions, elemT>* get_empty_copy() const = 0;

  //! Allocate a new DiscretisedDensity object which is a copy of the current one.
  virtual DiscretisedDensity<num_dimensions, elemT>* clone() const = 0;

  //! Allocate a new DiscretisedDensity object with same characteristics as the current one.
  //*! \deprecated Use get_empty_copy() instead
  DiscretisedDensity<num_dimensions, elemT>* get_empty_discretised_density() const
    { return get_empty_copy(); }

  //! \name Equality
  //@{
  //! Checks if the 2 objects have the same type, index range, origin etc
  /*! If they do \c not have the same characteristics, the string \a explanation
      explains why.
  */
  inline bool
    has_same_characteristics(self_type const&,
			     string& explanation) const;

  //! Checks if the 2 objects have the same type, index range, origin etc
  /*! Use this version if you do not need to know why they do not match.
   */
  inline bool
    has_same_characteristics(self_type const&) const;

  //! check equality (data has to be identical)
  /*! Uses has_same_characteristics() and Array::operator==.
      \warning This function uses \c ==, which might not be what you 
      need to check when \c elemT has data with float or double numbers.
  */
  inline bool operator ==(const self_type&) const; 
  
  //! negation of operator==
  inline bool operator !=(const self_type&) const; 
  //@}

 protected:
  //! Implementation used by  has_same_characteristics
  /*! \warning Has to be overloaded by the derived classes to check for other
      parameters. Also, the overloaded function has to call the current one.

      \par Developer's note

      We need this function as C++ rules say that if you overload a function, you hide all 
      functions of the same name.
  */
  virtual bool
    actual_has_same_characteristics(DiscretisedDensity<num_dimensions, elemT> const&,
				    string& explanation) const;

  //! Implementation used by get_relative_coordinates_for_indices
  /*!  \par Developer's note

      We need this function as C++ rules say that if you overload a function, you hide all 
      functions of the same name.
  */
  virtual
    CartesianCoordinate3D<float>
    actual_get_relative_coordinates_for_indices(const BasicCoordinate<num_dimensions,float>& indices) const = 0;

  virtual
    BasicCoordinate<num_dimensions,float>
    actual_get_index_coordinates_for_relative_coordinates(const CartesianCoordinate3D<float>& coords) const = 0;

private:
  CartesianCoordinate3D<float> origin;
  
};

END_NAMESPACE_STIR

#include "stir/DiscretisedDensity.inl"
#endif
