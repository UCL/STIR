//
// $Id: 
//
/*!
  \file
  \ingroup local_buildblock

  \brief Declaration of class SinglesRates

  \author Kris Thielemans and Sanida Mustafovic
  $Date: 
  $Revision: 
*/
/*
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
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

