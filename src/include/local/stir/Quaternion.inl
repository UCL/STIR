
#include "local/stir/Quaternion.h"

START_NAMESPACE_STIR

template <typename coordT>
coordT  
Quaternion<coordT>::dot_product (const Quaternion<coordT>& q1, const Quaternion<coordT>& q2)
{
  return((q1[1]*q2[1])+(q1[2]*q2[2])+(q1[3]*q2[3])+(q1[4]*q2[4]));
}

template <typename coordT>
Quaternion<coordT>& 
Quaternion<coordT>::
operator*=(const Quaternion<coordT>& q)
{ 
  Quaternion<coordT> tmp (*this);
  coords[1] = tmp[1]*q[1]-tmp[2]*q[2]-tmp[3]*q[3]-tmp[4]*q[4];
  coords[2] = tmp[1]*q[2]+tmp[2]*q[1]+tmp[3]*q[4]-tmp[4]*q[3];
  coords[3] = tmp[1]*q[3]+tmp[3]*q[1]+tmp[4]*q[2]-tmp[2]*q[4];
  coords[4] = tmp[1]*q[4]+tmp[4]*q[1]+tmp[2]*q[3]-tmp[3]*q[2];

  return *this;
 
}

template <typename coordT>
Quaternion<coordT>& 
Quaternion<coordT>::operator*= (const coordT& a)
{
  for (int i=1; i<=4; i++)
    coords[i] *= a;
  return *this;
 
}

template <typename coordT>
Quaternion<coordT> 
Quaternion<coordT>::operator* (const Quaternion& q) const
{
  Quaternion<coordT> tmp(*this);
  tmp *= q;

  return tmp;
}

template <typename coordT>
Quaternion<coordT> 
Quaternion<coordT>::operator* (const coordT& a) const
{
  Quaternion<coordT> tmp(*this);
  tmp *= a;
  return tmp;
}

template <typename coordT>
Quaternion<coordT>&  
Quaternion<coordT>:: operator/= (const coordT& a)
{
  for (int i=1; i<=4; i++)
    coords[i] /= a;
  return *this;

}
template <typename coordT>
Quaternion<coordT>& 
Quaternion<coordT>::operator/= (const Quaternion& q)
{
  const Quaternion<coordT> con_q(q[1],-q[2],-q[3],-q[4]);
  const coordT norm_squared =(square(q[1])+square(q[2])+square(q[3])+square(q[4]));
  *this *= con_q;
  *this /= norm_squared;

  return *this;
}

template <typename coordT>
Quaternion<coordT> 
Quaternion<coordT>::operator/(const Quaternion& q) const
{
  Quaternion<coordT> tmp(*this);
  tmp/=q;
  return tmp;
}

template <typename coordT>
Quaternion<coordT> 
Quaternion<coordT>::operator/ (const coordT& a) const
{
  Quaternion<coordT> tmp(*this);
  tmp /= a;
  return tmp;

}

template <typename coordT>
void 
Quaternion<coordT>:: neg_quaternion ()
{
  coords[1] =-(*this)[1];
  coords[2] =-(*this)[2];
  coords[3] =-(*this)[3];
  coords[4] =-(*this)[4];
 
}

 

template <typename coordT>
void 
Quaternion<coordT>:: conjugate()
{
  coords[1] =(*this)[1];
  coords[2] =-(*this)[2];
  coords[3] =-(*this)[3];
  coords[4] =-(*this)[4];
}

template <typename coordT>
void 
Quaternion<coordT>:: normalise() 
{
  const coordT n = sqrt(square(coords[1]) + square(coords[2]) +square(coords[3])+ square(coords[4]));   
  coords[1] /=n;
  coords[2] /=n;
  coords[3] /=n;
  coords[4] /=n;
}

template <typename coordT>
void 
Quaternion<coordT>::inverse()	
{
  float dp = dot_product((*this),(*this));
  coords[1] =  (*this)[1]/dp;
  coords[2] = -(*this)[2]/dp;
  coords[3] = -(*this)[3]/dp;
  coords[4] = -(*this)[4]/dp;
}


template <typename coordT>
Quaternion<coordT> conjugate(const Quaternion<coordT>& q)
{
  Quaternion<coordT> tmp = q;
  tmp.conjugate();
  return tmp;
}

template <typename coordT>
Quaternion<coordT> inverse(const Quaternion<coordT>& q)
{
  Quaternion<coordT> tmp = q;
  tmp.inverse();
  return tmp;
}

END_NAMESPACE_STIR
