//
// $Id: 
//
/*!
  \file
  \ingroup local_buildblock

  \brief Declaration of class Quaternion

  \author: Sanida Mustafovic
  \author: Kris Thielemans
  $Date: 
  $Revision: 
*/

/*
    Copyright (C) 2000- $Date: , IRSL
    See STIR/LICENSE.txt for details
*/


#ifndef __stir_Quaternion_H__
#define __stir_Quaternion_H__

#include "stir/Array.h"
#include "stir/BasicCoordinate.h"
//#include "stir/RegisteredParsingObject.h"

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

END_NAMESPACE_STIR

#include "local/stir/Quaternion.inl"

#endif



