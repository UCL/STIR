//
// $Id$
//
/*!
  \file
  \ingroup Shape

  \brief Declaration of class CombinedShape3D

  \author Kris Thielemans
  \author Sanida Mustafovic
  $Date$
  $Revision$
*/
#ifndef __tomo_Shape_CombinedShape3D_h__
#define __tomo_Shape_CombinedShape3D_h_

#include "tomo/RegisteredParsingObject.h"
#include "local/tomo/Shape/Shape3D.h"
#include "shared_ptr.h"
#include <functional>

START_NAMESPACE_TOMO


template<class T>
struct logical_and_not : public std::binary_function<T, T, bool>
	{
     inline bool operator()(const T& x, const T& y) const
	 { return x && !y; }
    };


template<class T>
struct logical_and : public std::binary_function<T, T, bool>
	{
     inline bool operator()(const T& x, const T& y) const
	 { return x || y; }
    };

template<class operation=logical_and<bool> >
class CombinedShape3D : 
   public RegisteredParsingObject<CombinedShape3D, Shape3D, Shape3D>
{
public:
  // TODO cannot work yet because of template
  // Name which will be used when parsing a Shape3D object
  //static const char * const registered_name; 

  inline CombinedShape3D( shared_ptr<Shape3D> object1_v, shared_ptr<Shape3D> object2_v);
  inline bool is_inside_shape(const CartesianCoordinate3D<float>& index) const;
  inline void translate(const CartesianCoordinate3D<float>& direction);
  inline void scale(const CartesianCoordinate3D<float>& scale3D);
  inline Shape3D* clone() const;

private:
  shared_ptr<Shape3D> object1_ptr;
  shared_ptr<Shape3D> object2_ptr;

};


END_NAMESPACE_TOMO

#include "local/tomo/Shape/CombinedShape3D.inl"

#endif