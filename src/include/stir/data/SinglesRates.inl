//
//
/*
    Copyright (C) 2000- 2005, Hammersmith Imanet Ltd
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
