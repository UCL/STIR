//
// $Id$: $Date$
//
#ifndef __DiscretisedDensity_H__
#define __DiscretisedDensity_H__

/*!
  \file 
 
  \brief defines the DiscretisedDensity class 

  \author Sanida Mustafovic 
  \author Kris Thielemans 
  \author (help from Alexey Zverovich)
  \author PARAPET project

  \date    $Date$

  \version $Revision$

*/


#include "CartesianCoordinate3D.h"
#include "Array.h"
#include "shared_ptr.h"
#include <string>

#ifndef TOMO_NO_NAMESPACES
using std::string;
#endif

START_NAMESPACE_TOMO

/*!
  \ingroup buildblock
  \brief This abstract class is the basis for all image representations.
  
  This class is templated with the number of dimensions (should be 2 or 3) 
  and the type of the data.

  It defines functionality common to all discretised densities: the
  data structure itself (Array) and an origin.
 
  \warning The origin is always a CartesianCoordinate3D<float>, 
  independent of what coordinate system (or even dimension) this
  class represents.

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
  
  //! Allocate a new DiscretisedDensity object with same characteristics as the current one.
  virtual DiscretisedDensity<num_dimensions, elemT>* get_empty_discretised_density() const=0;

  //! Allocate a new DiscretisedDensity object which is a copy of the current one.
  virtual DiscretisedDensity<num_dimensions, elemT>* clone() const = 0;
  
private:
  typedef Array<num_dimensions,elemT> base_type;
  CartesianCoordinate3D<float> origin;
  
};

END_NAMESPACE_TOMO

#include "DiscretisedDensity.inl"
#endif
