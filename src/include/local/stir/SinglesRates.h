//
// $Id$
//
/*
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
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
  \ingroup buildblock

  \brief Declaration of class SinglesRates

  \author Kris Thielemans and Sanida Mustafovic
  $Date$
  $Revision$ 
*/

#ifndef __stir_SinglesRates_H__
#define __stir_SinglesRates_H__

#include "stir/DetectionPosition.h"
#include "stir/RegisteredObject.h"
#include "stir/Scanner.h"
#include "stir/shared_ptr.h"

START_NAMESPACE_STIR

class SinglesRates : public RegisteredObject<SinglesRates>
{
public: 

  virtual ~SinglesRates () {};
  //! Virtual function that return singles rate given the detection postions and/or time or detection 
  virtual float get_singles_rate (const DetectionPosition<>& det_pos, 
				  const double start_time,
				  const double end_time) const =0;
  //! Get the scanner pointer
  inline const Scanner * get_scanner_ptr () const;

protected:
  shared_ptr<Scanner> scanner_sptr;

};


END_NAMESPACE_STIR

#include "local/stir/SinglesRates.inl"
#endif

