//
// $Id$
//
/*!
  \file
  \ingroup buildblock
  
  \brief Implementation of the round functions
    
  \author Kris Thielemans
  \author Charalampos Tsoumpas
      
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

template <int num_dimensions, class elemT>
BasicCoordinate<num_dimensions,int>
round(const BasicCoordinate<num_dimensions,elemT> x)
{
	BasicCoordinate<num_dimensions,int> rnd_x;
	for(int i=1;i<=num_dimensions;++i)
		rnd_x[i]=round(x[i]);	
	return rnd_x;  
}

END_NAMESPACE_STIR

