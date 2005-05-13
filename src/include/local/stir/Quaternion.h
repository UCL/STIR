//
// $Id$
//
/*
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup buildblock

  \brief Declaration of class stir::Quaternion

  \author Sanida Mustafovic
  \author Kris Thielemans
  $Date$
  $Revision$ 
*/



#ifndef __stir_Quaternion_H__
#define __stir_Quaternion_H__

#include "stir/BasicCoordinate.h"

START_NAMESPACE_STIR

template <typename coordT>
class Quaternion : public BasicCoordinate<4, coordT>  // TODO kris wants private inheritance 
{
protected:
  typedef BasicCoordinate<4, coordT> base_type;

public:

   Quaternion();
   Quaternion(const coordT&, const coordT&, const coordT&, const coordT&);
   Quaternion(const base_type& q);
   
   /*coordT component_1() const;
   coordT component_2() const;
   coordT component_3() const;
   coordT component_4() const;*/
 
   // Overload multiplication 
   inline Quaternion & operator*= (const coordT& a);
   inline Quaternion & operator*= (const Quaternion& q);
   inline Quaternion operator* (const Quaternion& q) const;
   inline Quaternion operator* (const coordT& a) const;
   // Overload division
   inline Quaternion & operator/= (const coordT& a);
   inline Quaternion & operator/= (const Quaternion& q);
   inline Quaternion operator/ (const Quaternion& q) const;
   inline Quaternion operator/ (const coordT& a) const;

   inline void neg_quaternion();
   inline void conjugate();
   inline void normalise();
   inline void inverse();
   inline static coordT dot_product (const Quaternion&, const Quaternion&);

private:
  
};

template <typename coordT>
inline Quaternion<coordT> conjugate(const Quaternion<coordT>&);

template <typename coordT>
inline Quaternion<coordT> inverse(const Quaternion<coordT>&);

template <typename coordT>
inline coordT norm_squared(const Quaternion<coordT>& q);

template <typename coordT>
inline coordT norm(const Quaternion<coordT>& q);

END_NAMESPACE_STIR

#include "local/stir/Quaternion.inl"

#endif
