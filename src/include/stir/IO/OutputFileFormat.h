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

  \todo Support projection data
  \todo  Provide functions that enable the user to inquire about
  capabilities. For instance, supports_multi_time_frames(),
  supports_different_xy_pixel_size() etc.
 */
class OutputFileFormat : 
  public RegisteredObject<OutputFileFormat >,
  public ParsingObject
{
public:
  OutputFileFormat(const NumericType& = NumericType::SHORT, 
                   const ByteOrder& = ByteOrder::native);

  //! Write a single image to file
  /*! 
    \param filename desired output filename. If it does not have an extension,
           a default extension will/should be added by the derived class.
	   If there is an extension, the derived class should try to honour it.
	   On return, the parameter will be overwritten with the actual filename
	   used, such that the file can be read back using this string.
    \param density the image to write to file.
    \return Succeeded::yes if the file was successfully written.

    \warning In the case of file formats that use a separate header file, the \a
       filename argument at input is/should be used as a filename for the file
       with the actual data. At output however, the name of the header file 
       will/should be returned. This is all a bit messy, so it's 
       <strong>recommended</strong> to 
       not use an extension for the output filename.

  \todo 
      Unfortunately, C++ does not allow virtual member templates, so we'd need
      other versions of this for other data types or dimensions.
  */
  virtual
  Succeeded  
    write_to_file(string& filename, 
                  const DiscretisedDensity<3,float>& density) const = 0;
		  
  //! write a single image to file
  /*! See the virtual version. This version does not return the 
      filename used.
  */
  Succeeded  
    write_to_file(const string& filename, 
                  const DiscretisedDensity<3,float>& density) const;


  //! get type used for outputing numbers 
  NumericType get_type_of_numbers() const;
  //! get byte order used for output 
  ByteOrder get_byte_order();
  //! set type used for outputing numbers 
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


protected:
  //! type used for outputing numbers 
  NumericType type_of_numbers;
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
