

#include "local/stir/Quaternion.h"


START_NAMESPACE_STIR

template <class coordT>
Quaternion<coordT>::Quaternion()
  : base_type()
{}

template <class coordT>
Quaternion<coordT>::Quaternion(const coordT& c1, 
			       const coordT& c2, 
			       const coordT& c3,
			       const coordT& c4)
  : base_type()
{
  coords[1] = c1;
  coords[2] = c2;
  coords[3] = c3;
  coords[4] = c4;
}

template <class coordT>
Quaternion<coordT>::Quaternion(const base_type& c)
  : base_type(c)
{}

#if 0
template <class coordT>
coordT
Quaternion<coordT>:: component_1() const
{
  return coords[1];
}

template <class coordT>
coordT
Quaternion<coordT>:: component_2() const
{
  return coords[2];
}

template <class coordT>
coordT
Quaternion<coordT>:: component_3() const
{
  return coords[3];
}

template <class coordT>
coordT
Quaternion<coordT>:: component_4() const
{
  return coords[4];
}

#endif
   

template class Quaternion<float>;
END_NAMESPACE_STIR