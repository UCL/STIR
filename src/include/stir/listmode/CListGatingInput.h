//
//
/*!
  \file
  \ingroup listmode
  \brief Declarations of classes stir::CListGatingInput which
  is used for list mode data.
    
  \author Nikos Efthimiou
  \author Kris Thielemans
      
*/
/*
    Copyright (C) 2003- 2011, Hammersmith Imanet Ltd
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

#ifndef __stir_listmode_CListGatingInput_H__
#define __stir_listmode_CListGatingInput_H__


START_NAMESPACE_STIR

/*!
  \class
  \ingroup listmode
  \brief Declarations of classes stir::CListGatingInput which
  is used for list mode data for recording external input to the scanner (normally used for gating)

     For some scanners, the state of some external measurements can be recorded in the
   list file, such as ECG triggers etc. We currently assume that these take discrete values.

   If your scanner has more data available, you can provide it in the derived class.

  \author Kris Thielemans

*/
class CListGatingInput
{
public:
  virtual ~CListGatingInput() {}

  //! get gating-related info
  /*! Generally, gates are numbered from 0 to some maximum value.
   */
  virtual unsigned int get_gating() const = 0;

  virtual Succeeded set_gating(unsigned int) = 0;
}; 

END_NAMESPACE_STIR

#endif
