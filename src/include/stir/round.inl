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

END_NAMESPACE_STIR
