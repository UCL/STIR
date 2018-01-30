//
//
/*!
  \file
  \ingroup Shape

  \brief Inline implementations of class stir::CombinedShape3D

  \author Kris Thielemans
  \author Sanida Mustafovic
*/
/*
    Copyright (C) 2000- 2009, Hammersmith Imanet Ltd
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
START_NAMESPACE_STIR

template<class operation>
CombinedShape3D<operation>::CombinedShape3D( shared_ptr<Shape3D> object1_v, shared_ptr<Shape3D> object2_v)
    :object1_ptr(object1_v),
     object2_ptr(object2_v)
 {}


template<class operation>
bool CombinedShape3D<operation>::is_inside_shape(const CartesianCoordinate3D<float>& index) const
 {
   return operation()(object1_ptr->is_inside_shape(index),
                      object2_ptr->is_inside_shape(index));
}

template<class operation>
Shape3D* CombinedShape3D<operation>::clone() const
{
  // TODO alright ?
#if 0
  Shape3D* tmp = static_cast<Shape3D *>(new CombinedShape3D<operation>(*this));
  cerr << "Cloning " << this << 
    ", new " << tmp << endl;
  return tmp;
#else
  return static_cast<Shape3D *>(new CombinedShape3D<operation>(*this));
#endif
}

template<class operation>
void CombinedShape3D<operation>::translate(const CartesianCoordinate3D<float>& direction)
{
    // TODO alright ?
  shared_ptr<Shape3D> new_object1_ptr = object1_ptr->clone();
  shared_ptr<Shape3D> new_object2_ptr = object2_ptr->clone();
  object1_ptr = new_object1_ptr;
  object2_ptr = new_object2_ptr;
  object1_ptr->translate(direction);
  object2_ptr->translate(direction);
}


template<class operation>
void CombinedShape3D<operation>::scale(const CartesianCoordinate3D<float>& scale3D)
{
    // TODO alright ?
#if 0
  cerr << "scale: " << object1_ptr.ptr->data
     << ", " << object2_ptr.ptr->data << endl;
#endif
  shared_ptr<Shape3D> new_object1_ptr = object1_ptr->clone();
  shared_ptr<Shape3D> new_object2_ptr = object2_ptr->clone();
  object1_ptr = new_object1_ptr;
  object2_ptr = new_object2_ptr;
  object1_ptr->scale(scale3D);
  object2_ptr->scale(scale3D);
}

END_NAMESPACE_STIR
