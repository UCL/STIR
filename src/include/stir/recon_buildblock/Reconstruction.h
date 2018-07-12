//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2007, Hammersmith Imanet Ltd
    Copyright (C) 2018, University College London
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
#ifndef __stir_recon_buildblock_Reconstruction_H__
#define __stir_recon_buildblock_Reconstruction_H__
/*!
  \file 
  \ingroup recon_buildblock
 
  \brief declares the stir::Reconstruction class

  \author Kris Thielemans
  \author Matthew Jacobson
  \author Claire Labbe
  \author PARAPET project

*/

#include "stir/TimedObject.h"
#include "stir/ParsingObject.h"
#include "stir/shared_ptr.h"
#include "stir/DataProcessor.h"
#include "stir/IO/OutputFileFormat.h"
#include "stir/RegisteredObject.h"
#include <string>

#include "stir/IO/ExamData.h"

START_NAMESPACE_STIR


class Succeeded;

/*!
  \brief base class for all Reconstructions
  \ingroup recon_buildblock
  As there is not a lot of commonality between different reconstruction algorithms,
  this base class is rather basic. It essentially takes care of constructing a target
  image, calls the virtual reconstruct() function, and writes the result to file.

  For convenience, the class is derived from TimedObject. It is the 
  responsibility of the derived class to run these timers though.

  \par Parsing parameters

  \verbatim
  ; post-processing after image reconstruction,see DataProcessor<TargetT>
  ; defaults to no processing ("None")
  post-filter type :=

  ; output file(s) will be written with the following file name
  output filename prefix := 
  ; output file(s) will use the following file format
  ; see OutputFileFormat<TargetT>
  output file format :=
  \endverbatim

*/

template <typename TargetT>
class Reconstruction :
        public RegisteredObject<Reconstruction < TargetT > >,
        public TimedObject,
        public ParsingObject
{
public:
  //! virtual destructor
  virtual ~Reconstruction() {};
  
  //! gives method information
  virtual std::string method_info() const = 0;
  
  //! executes the reconstruction
  /*!
    \return Succeeded::yes if everything was alright.
   */     
  virtual Succeeded 
    reconstruct() = 0;

  //! executes the reconstruction storing result in \c target_image_sptr
  /*!
   \param target_image_sptr The result of the reconstruction is stored in 
   \c *target_image_sptr.
    \return Succeeded::yes if everything was alright.

   \par Developer\'s note

   Because of C++ rules, overloading one of the reconstruct() functions
   in a derived class, will hide the other. So you have to overload both.
  */     
  virtual Succeeded 
    reconstruct(shared_ptr<TargetT> const& target_image_sptr) = 0;

  //! operations prior to the reconstruction
  /*! Will do various consistency checks and return Succeeded::no 
    if something is wrong.

    \todo Currently, set_up() is called by reconstruct(). This is in
    contrast with some other class hierarchies in STIR where set_up()
    has to be called before any actual processing. Maybe this should
    be made consistent.
  */
  virtual Succeeded set_up(shared_ptr <TargetT > const& target_data_sptr);

  /*! \name Functions to set parameters
    This can be used as alternative to the parsing mechanism.
   \warning Be careful with setting shared pointers. If you modify the objects in 
   one place, all objects that use the shared pointer will be affected.
  */
  //@{

  //! file name for output reconstructed images
  void set_output_filename_prefix(const std::string&); 

  //! defines the format of the output files
  void set_output_file_format_ptr(const shared_ptr<OutputFileFormat<TargetT> >&);

  //! post-filter
  void set_post_processor_sptr(const shared_ptr<DataProcessor<TargetT> > &);

  //! \brief set input data
  virtual void set_input_data(const shared_ptr<ExamData>&) = 0;
  //! get input data
  /*! Will throw an exception if it wasn't set first */
  virtual const ExamData& get_input_data() const = 0;
  //@}

  //!
  //! \brief set_disable_output
  //! \param _val
  //! \author Nikos Efthimiou
  //! \details This function is called if the user deside to mute any output images.
  //! The best way to do this is to use the "disable output" key in the par file.
  //! \warning The "output filename prefix" has to be set.
  void set_disable_output(bool _val);

  //!
  //! \brief set_enable_output
  //! \param _val
  //! \author Nikos Efthimiou
  //! \details The counterpart of set_disable_output().
  void set_enable_output(bool _val);

  //!
  //! \brief get_reconstructed_image
  //! \author Nikos Efthimiou
  //! \return
  //!
  shared_ptr<TargetT > get_target_image();

  // parameters
 protected:

  //! file name for output reconstructed images
  std::string output_filename_prefix; 

  //! defines the format of the output files
  shared_ptr<OutputFileFormat<TargetT> > output_file_format_ptr; 

  //! post-filter
  shared_ptr<DataProcessor<TargetT> >  post_filter_sptr;

protected:

  /*! 
  \brief 
  This function initialises all parameters, either via parsing, 
  or by calling ask_parameters() (when parameter_filename is the empty string).

  It should be called in the constructor of the last class in the 
  hierarchy. At that time, all Interfile keys will have been
  initialised, and ask_parameters() will be the appropriate virtual
  function, such that questions are asked for all parameters.

  \todo It currently calls error() when something goes wrong. It should
  return Succeeded (or throw an exception).
  */
  void initialise(const std::string& parameter_filename);
  
  virtual void set_defaults();
  virtual void initialise_keymap();
  //! used to check acceptable parameters after parsing
  /*!
    The function should be used to set members that have 
    are not set directly by the parsing. For example,
    parsing might set \c input_filename, and \c post_processing()
    might then read in the data and set the corresponding 
    Reconstruction parameter.

    Consistency checks mostly belong in \c set_up(). The reason for this
    is that for instance a GUI might not use the parsing mechanism and 
    set parameters by calling various \c set_ functions (such as
    \c set_post_processor_sptr() ).
  */
  virtual bool post_processing();

  //!
  //! \brief target_data_sptr
  //!
  shared_ptr<TargetT > target_data_sptr;

  //!
  //! \brief _disable_output
  //! \author Nikos Efthimiou
  //! \details This member mutes the creatation and write into an
  //! output image. You want to use it if you call for reconstruction
  //! from within some other code and want to use directly the output image.
  bool _disable_output;


};

END_NAMESPACE_STIR

#endif

