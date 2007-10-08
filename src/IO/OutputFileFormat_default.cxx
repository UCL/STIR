//
// $Id$
//
/*
    Copyright (C) 2005- $Date$, Hammersmith Imanet Ltd
    This file is part of STIR.
    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup IO
  \brief  initialisation of the stir::OutputFileFormat::_default_sptr member
  \author Kris Thielemans
      
  $Date$
  $Revision$
*/

#include "stir/IO/InterfileOutputFileFormat.h"
#include "stir/DiscretisedDensity.h"

START_NAMESPACE_STIR

template <>
shared_ptr<OutputFileFormat<DiscretisedDensity<3,float> > >
OutputFileFormat<DiscretisedDensity<3,float> >::_default_sptr =
   new InterfileOutputFileFormat;


END_NAMESPACE_STIR




