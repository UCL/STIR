//
// $Id$
//
/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet Ltd
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
  \brief Declaration of class stir::OutputFileFormat

  \author Kris Thielemans

  $Date$
  $Revision$
*/

#ifndef __stir_IO_OutputFileFormat_H__
#define __stir_IO_OutputFileFormat_H__

#include "stir/RegisteredObject.h"
#include "stir/ParsingObject.h"
#include "stir/NumericType.h"
#include "stir/ByteOrder.h"
#include <string>

#ifndef STIR_NO_NAMESPACES
using std::string;
#endif

START_NAMESPACE_STIR

class Succeeded;


/*!
  \ingroup IO
  \brief 
  Base class for classes that create output files.

  \par Parsing
  The following keywords can be parsed (defaults indicated).
  \verbatim
   ; byte order defaults to native order
   byte order := littleendian
   ; type specification as in Interfile
   number format:=float
   number of bytes per pixel:=4

   scale_to_write_data:=0
  \endverbatim
  \todo Support projection data
  \todo  Provide functions that enable the user to inquire about
  capabilities. For instance, supports_multi_time_frames(),
  supports_different_xy_pixel_size() etc.
 */
template <typename DataT>
class OutputFileFormat : 
  public RegisteredObject<OutputFileFormat<DataT> >,
  public ParsingObject
{
public:
  //! A function to return a default output file format
  static 
    shared_ptr<OutputFileFormat<DataT> >
    default_sptr();

  OutputFileFormat(const NumericType& = NumericType::FLOAT, 
                   const ByteOrder& = ByteOrder::native);

  //! Write a single image to file
  /*! 
    \param filename desired output filename. If it does not have an extension,
           a default extension will/should be added by the derived class.
	   If there is an extension, the derived class should try to honour it.
	   On return, the parameter will be overwritten with the actual filename
	   used, such that the file can be read back using this string.
    \param data the data to write to file.
    \return Succeeded::yes if the file was successfully written.

    \warning In the case of file formats that use a separate header file, the \a
       filename argument at input is/should be used as a filename for the file
       with the actual data. At output however, the name of the header file 
       will/should be returned. This is all a bit messy, so it's 
       <strong>recommended</strong> to 
       <strong>not</strong> use an extension for the output filename.
  */
  Succeeded  
    write_to_file(string& filename, 
                  const DataT& data) const;
		  
  //! write a single image to file
  /*! See the version with non-const \a filename. This version does not return the 
      filename used. 
  */
  Succeeded  
    write_to_file(const string& filename, 
                  const DataT& density) const;


  //! get type used for outputting numbers 
  NumericType get_type_of_numbers() const;
  //! get byte order used for output 
  ByteOrder get_byte_order();
  //! get scale to write the data
  /*! \see set_scale_to_write_data
   */
  float get_scale_to_write_data() const;


  //! set type used for outputting numbers 
  /*! Returns type actually used. 
     Calls warning() with some text if the requested type is not supported.
  
     Default implementation accepts any type.
  */
  virtual NumericType set_type_of_numbers(const NumericType&, const bool warn = false);
  //! set byte order used for output
  /*! Returns type actually used.
    Calls warning() with some text if the requested type is not supported. 
  
     Default implementation accepts any byte order.
  */ 
  virtual ByteOrder set_byte_order(const ByteOrder&, const bool warn = false);
  //! set byte order and data type used for output
  /*! Changes parameters to the types actually used.
   Calls warning() with some text if the requested type is not supported. 

   This function is necessary in case a byte order for a particular data type is not supported.
    
   Default implementation calls set_byte_order() and set_type_of_numbers().
  */ 
  virtual void set_byte_order_and_type_of_numbers(ByteOrder&, NumericType&, const bool warn = false);

  //! set scale outputting numbers 
  /*! Returns scale actually used. 
     Calls warning() with some text if the requested scale is not supported.
  
     If \a scale_to_write_data is 0 (which is the default), the output will
     be rescaled such that the maximum range of the output type of numbers is used,
     except for floats and doubles in which case no rescaling occurs.

     Default implementation accepts any scale.
  */
  virtual float set_scale_to_write_data(const float new_scale_to_write_data, const bool warn=false);

protected:
  //! type used for outputting numbers 
  NumericType type_of_numbers;
  //! byte order used for output 
  ByteOrder file_byte_order;
  //! scale to write the data
  /*! \see set_scale_to_write_data
   */
  float scale_to_write_data;

  //! virtual function called by write_to_file()
  /*! This function has to be overloaded by the derived class.

      The reason we do not simply make write_to_file() virtual is that we have
      2 versions of write_to_file. C++ rules are such that overloading the virtual
      function in a derived class means that the other version gets hidden. Having
      the non-virtual write_to_file() call the virtual actual_write_to_file() solves
      this problem.
  */
  virtual Succeeded  
    actual_write_to_file(string& filename, 
                  const DataT& density) const = 0;

  // parsing stuff

  //! sets value for output data type
  /*! Has to be called by set_defaults() in the leaf-class */
  virtual void set_defaults();
  //! sets keys for output data type for parsing
  /*! Has to be called by initialise_keymap() in the leaf-class */
  virtual void initialise_keymap();
  //! Checks if parameters have sensible values after parsing
  /*! Has to be called by post_processing() in the leaf-class */
  virtual bool post_processing();
  //! overloaded member for ParsingObject::set_key_values()
  /*! Has to be called by set_key_values() in the leaf-class (if it redefines it) */
  virtual void set_key_values();

private:
  static shared_ptr<OutputFileFormat<DataT> > _default_sptr;
  // Lists of possible values for some keywords
  static ASCIIlist_type number_format_values;
  static ASCIIlist_type byte_order_values;

  // Corresponding variables here

  int number_format_index;
  int byte_order_index;
  int bytes_per_pixel;
};



END_NAMESPACE_STIR


#endif
