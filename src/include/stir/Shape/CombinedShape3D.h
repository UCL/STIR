//
//
/*
    Copyright (C) 2000- 2007, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup Shape

  \brief Declaration of class stir::CombinedShape3D

  \author Kris Thielemans
  \author Sanida Mustafovic
*/
#ifndef __stir_Shape_CombinedShape3D_h__
#define __stir_Shape_CombinedShape3D_h__

#include "stir/RegisteredParsingObject.h"
#include "stir/Shape/Shape3D.h"
#include "stir/shared_ptr.h"

START_NAMESPACE_STIR

template <class T>
struct logical_and_not
{
  inline bool operator()(const T& x, const T& y) const { return x && !y; }
};

template <class T>
struct logical_and
{
  inline bool operator()(const T& x, const T& y) const { return x || y; }
};

/*! \ingroup Shape
    \brief A class that allows combining several shapes using logical operations
    \todo document more
    \todo Parsing cannot work yet because of template (can be solved by explicit instantiation)

*/
template <class operation = logical_and<bool>>
class CombinedShape3D : public RegisteredParsingObject<CombinedShape3D, Shape3D, Shape3D>
{
public:
  // Name which will be used when parsing a Shape3D object
  // static const char * const registered_name;

  inline CombinedShape3D(shared_ptr<Shape3D> object1_v, shared_ptr<Shape3D> object2_v);
  inline bool is_inside_shape(const CartesianCoordinate3D<float>& coord) const;
  inline void translate(const CartesianCoordinate3D<float>& direction);
  inline void scale(const CartesianCoordinate3D<float>& scale3D);
  inline Shape3D* clone() const;

private:
  shared_ptr<Shape3D> object1_ptr;
  shared_ptr<Shape3D> object2_ptr;
};

END_NAMESPACE_STIR

#include "stir/Shape/CombinedShape3D.inl"

#endif
