//
// $Id$
//
/*!
  \file
  \ingroup buildblock
  
  \brief Implementation of the round functions
    
  \author Kris Thielemans
      
  $Date$	
  $Revision$	
*/
/*
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

START_NAMESPACE_STIR

int round(const float x)
{
  if (x>=0)
    return static_cast<int>(x+0.5F);
  else
    return -static_cast<int>(-x+0.5F);
}

int round(const double x)
{
  if (x>=0)
    return static_cast<int>(x+0.5);
  else
    return -static_cast<int>(-x+0.5);
}

template <class elemT>
BasicCoordinate<3,int> 
round(const BasicCoordinate<3,elemT> x)
{
	BasicCoordinate<3,int> rnd_x;
	rnd_x[1]=round(x[1]);
	rnd_x[2]=round(x[2]);
	rnd_x[3]=round(x[3]);
	return rnd_x;  
}

END_NAMESPACE_STIR

