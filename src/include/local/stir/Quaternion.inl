
#include "local/stir/Quaternion.h"

START_NAMESPACE_STIR

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
  tmp[1] = coords[1]*q[1]-coords[2]*q[2]-coords[3]*q[3]-coords[4]*q[4];
  tmp[2] = coords[1]*q[2]+coords[2]*q[1]+coords[3]*q[4]-coords[4]*q[3];
  tmp[3] = coords[1]*q[3]+coords[3]*q[1]+coords[4]*q[2]-coords[2]*q[4];
  tmp[4] = coords[1]*q[4]+coords[4]*q[1]+coords[2]*q[3]-coords[3]*q[2];

  return tmp;

}

template <typename coordT>
Quaternion<coordT> 
Quaternion<coordT>::operator* (const coordT& a) const
{
  Quaternion<coordT> tmp(*this);
  for (int i=1; i<=4; i++)
    tmp[i] *= a;
  return tmp;
}

template <typename coordT>
Quaternion<coordT>&  
Quaternion<coordT>:: operator/= (const coordT& a)
{
  Quaternion<coordT> tmp(*this);
  for (int i=1; i<=4; i++)
    coords[i] /= a;
  return *this;

}
template <typename coordT>
Quaternion<coordT>& 
Quaternion<coordT>::operator/= (const Quaternion& q)
{
  Quaternion<coordT> tmp(*this);
  Quaternion<coordT> con_q(q[1],-q[2],-q[3],-q[4]);
  coordT norm =(square(q[1])+square(q[2])+square(q[3])+square(q[4]));
  for ( int i = 1; i<=4 ;i++)
  { 
    con_q[i] /=norm;
    coords[i] *= con_q[i];
  }

  return *this;
}

template <typename coordT>
Quaternion<coordT> 
Quaternion<coordT>::operator/(const Quaternion& q) const
{
  Quaternion<coordT> tmp(*this);
  Quaternion<coordT> con_q(q[1],-q[2], -q[3],-q[4]);
  coordT norm =(square(q[1])+square(q[2])+square(q[3])+square(q[4]));
  for ( int i = 1; i<=4 ;i++)
  { 
    con_q[i]/=norm;
  }
   tmp*=con_q;
   return tmp;
}

template <typename coordT>
Quaternion<coordT> 
Quaternion<coordT>::operator/ (const coordT& a) const
{
  Quaternion<coordT> tmp(*this);
  for (int i=1; i<=4; i++)
    tmp[i] /= a;
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
  double n = sqrt(square(coords[1]) + square(coords[2]) +square(coords[3])+ square(coords[4]));   
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
float 
Quaternion<coordT>::dot_product (Quaternion<coordT>& q1, Quaternion<coordT>& q2)
{
  return((q1[1]*q2[1])+(q1[2]*q2[2])+(q1[3]*q2[3])+(q1[4]*q2[4]));
}

END_NAMESPACE_STIR
