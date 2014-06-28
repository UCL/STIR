//
//
/*
    Copyright (C) 2000- 2009, Hammersmith Imanet Ltd
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
  \ingroup DataProcessor
  \brief Declaration of class stir::DataProcessor

  \author Kris Thielemans
  \author Sanida Mustafovic

*/
#ifndef __stir_DataProcessor_H__
#define __stir_DataProcessor_H__


#include "stir/RegisteredObject.h"
#include "stir/ParsingObject.h"
#include "stir/TimedObject.h"
#include "stir/Succeeded.h"

START_NAMESPACE_STIR

/*!
  \ingroup DataProcessor
  \brief 
  Base class that defines an interface for classes that do data processing.

  Classes at the end of the DataProcessor hierarchy have to be able 
  to parse parameter files etc. Moreover,
  it has to be possible to construct an DataProcessor object while 
  parsing, where the actual type of derived class is determined at 
  run-time. 
  Luckily, this is exactly the situation that RegisteredObject and 
  RegisteredParsingObject are supposed to handle. So, all this 
  functionality is achieved by deriving DataProcessor from the
  appropriate RegisteredObject class, and deriving the 'leaves'
  from Registered ParsingObject.
 */
template <typename DataT>
class DataProcessor : 
public RegisteredObject<DataProcessor<DataT> >,
public ParsingObject,
public TimedObject
{
public:
  inline DataProcessor();

  //! Initialises any internal data (if necessary) using \a data as a template for sizes, sampling distances etc.
  /*! 
     \warning DataProcessor does NOT check if the input data for apply()
     actually corresponds to the template. So, most derived classes will 
     \b not call set_up again if the input data does
     not correspond to the template, potentially resulting in erroneous output.

     The reason that DataProcessor does not perform this check is that
     it does not know what the requirements are to call the 2 densities
     'compatible'.
   */
  inline Succeeded  
    set_up(const DataT& data);

  //! Makes sure we will ignore any previous call to set-up()
  /*! If you change any internal variables of the data-processor, or are calling
      it on data if different size or so, you first have to call reset() such that
      the data-processor will call set_up() when necessary.

      A derived class could overload reset() to re-initialise any internal variables, 
      but this is not required.
  */
  inline virtual void 
    reset();

  //! Calls set_up() (if not already done before) and process \a data in-place
  /*! If set_up() returns Succeeded::false, a warning message is written, 
      and the \a data is not changed.
  */
  inline Succeeded
    apply(DataT& data);

  /*!
    \brief
    Calls set_up() (if not already done before) and process \a in_data, 
    putting the result in \a out_data.

    If set_up() returns Succeeded::false, a warning message is written, 
    and the \a out_data is not changed.

    \warning Most derived classes will assume that out_data is already 
    initialised appropriately (e.g. has correct dimensions, voxel sizes etc.).
  */
  inline Succeeded apply
    (DataT& out_data,
     const DataT& in_data);

  /*! \name parsing functions

      parse() returns false if there is some error, true otherwise.

      These call reset() first, and then ParsingObject::parse
  */
  //@{
  inline bool parse(std::istream& f);
  bool parse(const char * const filename);
  //@}

  // Check if filtering images with this dimensions, sampling_distances etc actually makes sense
  //virtual inline Succeeded consistency_check( const DataT& image ) const;  


protected:
  //! Will be called to build any internal parameters
  virtual Succeeded  virtual_set_up(const DataT&) = 0;
  //! Performs actual operation (virtual_set_up is called before this function)
  //*! \todo should return Succeeded */
  virtual void virtual_apply(DataT& data, 
			     const DataT& in_data) const = 0; 
  //! Performs actual operation (in-place)
  //*! \todo should return Succeeded */
  virtual void virtual_apply(DataT& data) const =0;
private:  
  bool is_set_up_already;  

};



END_NAMESPACE_STIR

#include "stir/DataProcessor.inl"

#endif
