//
// $Id$
//
/*!

  \file

  \brief 

  \author Claire Labbe
  \author Kris Thielemans
  \author PARAPET project

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/
#ifndef __RampFilter_H__
#define __RampFilter_H__

#include "local/stir/Filter.h"

START_NAMESPACE_STIR

class RampFilter : public Filter1D<float>
{

public:

 float fc;
 float alpha;
 float sampledist; 

 RampFilter(float sampledist_v, int length_v , float alpha_v=1, float fc_v=.5); 

 virtual string parameter_info() const;
 
};

END_NAMESPACE_STIR

#endif
