//
//
/*
    Copyright (C) 2000- 2005, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup singles_buildblock

  \brief Implementation of class stir::SinglesRates

  \author Kris Thielemans and Sanida Mustafovic
*/


START_NAMESPACE_STIR


const 
Scanner* SinglesRates::get_scanner_ptr() const
{ 
  return scanner_sptr.get();
}



const Scanner *
FrameSinglesRates::
get_scanner_ptr() const {
  return _scanner_sptr.get();
}


END_NAMESPACE_STIR
