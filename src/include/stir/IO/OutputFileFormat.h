//
// $Id$
//
/*!

  \file
  \ingroup IO
  \brief Declaration of class OutputFileFormat

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000-2$Date$, IRSL
    See STIR/LICENSE.txt for details
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

template <int num_dimensions, typename elemT> class DiscretisedDensity;
class Succeeded;


/*!
  \ingroup IO
  \brief 
  Base class for classes that create output files.

 */
class OutputFileFormat : 
  public RegisteredObject<OutputFileFormat >,
  public ParsingObject
{
public:
  OutputFileFormat(const NumericType& = NumericType::SHORT, 
                   const ByteOrder& = ByteOrder::native);

  //! Write a single image to file
  /*! \todo 
      Unfortunately, C++ does not allow virtual member templates, so we'd need
      other versions of this for other data types or dimensions.
  */
  virtual
  Succeeded  
    write_to_file(const string& filename, 
                  const DiscretisedDensity<3,float>& density) = 0;


protected:
  //! type used for outputing numbers 
  NumericType           type_of_numbers;
  //! byte order used for output 
  ByteOrder file_byte_order;

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
  /*! Has to be called by set_key_values() in th eleaf-class (if it redefines it) */
  virtual void set_key_values();

private:
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
