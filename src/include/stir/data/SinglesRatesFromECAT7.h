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
  \ingroup singles_buildblock

  \brief Declaration of class stir::ecat::ecat7::SinglesRatesFromECAT7
  \author  Sanida Mustafovic and Kris Thielemans
  $Date$
  $Revision$ 
*/

#ifndef __stir_data_SinglesRatesFromECAT7_H__
#define __stir_data_SinglesRatesFromECAT7_H__

#include "stir/data/SinglesRatesForTimeFrames.h"
#include "stir/RegisteredParsingObject.h"
#include "stir/IO/stir_ecat7.h" // necessary for namespace macros
START_NAMESPACE_STIR
START_NAMESPACE_ECAT
START_NAMESPACE_ECAT7

/*!
  \ingroup singles_buildblock
  \brief A class that extracts singles info from an ECAT7 sinogram file.
*/
class SinglesRatesFromECAT7 : 
public RegisteredParsingObject<SinglesRatesFromECAT7, SinglesRates,
                               AddParser<SinglesRatesForTimeFrames> >
{ 
public:

    //! Name which will be used when parsing a SinglesRatesFromECAT7 object 
    static const char * const registered_name; 
    
    //! Default constructor 
    SinglesRatesFromECAT7 ();

    
    //!  The function that reads singles from ECAT7 file
    /*! \return The number of frames found. 0 if failed.
     */
    int read_singles_from_file(const string& ECAT7_filename,
                               const std::ios::openmode open_mode = std::ios::in);
    
   
private:
  string ECAT7_filename;
  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();  
};

END_NAMESPACE_ECAT7
END_NAMESPACE_ECAT
END_NAMESPACE_STIR


#endif
