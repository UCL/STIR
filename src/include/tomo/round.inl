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

START_NAMESPACE_TOMO

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

END_NAMESPACE_TOMO
