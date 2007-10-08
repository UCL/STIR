//
// $Id$
//
#ifndef __stir_IO_InputFileFormatRegistry_h__
#define __stir_IO_InputFileFormatRegistry_h__
/*
    Copyright (C) 2006- $Date$, Hammersmith Imanet Ltd
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
  \brief Declaration of class stir::InputFileFormatRegistry, stir::RegisterInputFileFormat.

  \author Kris Thielemans

  $Date$
  $Revision$
*/
#include "stir/utilities.h"
#include "stir/IO/InputFileFormat.h"
#include "stir/shared_ptr.h"
#include <map> 
#include <fstream>
#include <string>
#include <utility> // for make_pair

#include "stir/info.h"
#include "boost/format.hpp"

namespace std
{
  template <class T> class auto_ptr;
}

START_NAMESPACE_STIR

//! A class for registering (and finding) all input file formats
/*! \ingroup IO
    \preliminary

    This class stores 'factories' that take a file as argument to produce a new object.

    \see RegisterInputFileFormat on a convenient way to add InputFileFormat objects to
    the default registry.
    
    \par terminology 

    'Factory' is terminology often used in C++ for an object that can make another object.
*/
template <class DataT>
class InputFileFormatRegistry
{
 public:
  typedef DataT data_type;
  // maybe have a factory type, and let InputFileFormat be derived from it
  typedef InputFileFormat<DataT> Factory;
  typedef shared_ptr<Factory> FactorySPtr;
  typedef InputFileFormatRegistry<DataT> self_type;

  //! A function to return the default registry
  /*! This default registry will be created when this function called the first
      time.It will then be empty.

      \warning This function returns a reference to the default registry. This can be
      used to assign your own registry to make it the default for all other
      subsequent calls. This might lead to unexpected behaviour if your registry does
      not contain the expected factories.
  */
  static inline
    shared_ptr<self_type>& default_sptr();

  //! Default constructor without defaults (see find_factory())
  inline InputFileFormatRegistry() {}

  inline ~InputFileFormatRegistry() {}
			 

  /*! \brief Add a file-format to the registry with given ranking
    Ranking 0 is the 'highest', so will be found first.
  */
  inline void add_to_registry(FactorySPtr const & factory, const unsigned ranking)
    {
      this->_registry.insert(std::make_pair(ranking, factory));
    }
  
  //! Remove a pair from the registry
  inline void remove_from_registry(const Factory& factory);

  //! List all keys to an ostream, separated by newlines.
  //inline void list_keys(ostream& s) const;

  //! Find a factory corresponding that can handle a particular stream
  /*! The \c signature and \c input arguments are supposed to correspond to the same file.
      
      The function will loop through all factories in the registry, in order of decreasing
      \c ranking, and return the first factory found that can handle the data.
      
      If no matching factory is found, we call error().
  */
  inline Factory const & 
    find_factory(const FileSignature& signature,
		 std::istream& input)
    {
      const_iterator iter= this->_actual_find_factory(signature, input);
      if (this->_valid(iter))
	return *(iter->second);
      else
	{
	  std::cerr << "Available input file formats:\n";
	  this->list_registered_names(std::cerr);
	  error("no file format found that can read this data");
	}
      // we never get here, but most compilers will complain here
      // so we 'return' a bogus factory
      return (*iter->second);
    }

  //! Find a factory corresponding that can handle a particular filename
  /*! The \c signature and \c input arguments are supposed to correspond to the same file.
      
      The function will loop through all factories in the registry, in order of decreasing
      \c ranking, and return the first factory found that can handle the data.

      If no matching factory is found, we call error().
  */
  inline Factory const & 
    find_factory(const FileSignature& signature,
		 const std::string& filename)
    {
      std::ifstream input;
      open_read_binary(input, filename);
      const_iterator iter= this->_actual_find_factory(signature, input);
      if (this->_valid(iter))
	return (*iter->second);
      else
	{
	  std::cerr << "Available input file formats:\n";
	  this->list_registered_names(std::cerr);
	  error("no file format found that can read %s", filename.c_str());
	}
      // we never get here, but most compilers will complain here
      // so we 'return' a bogus factory
      return (*iter->second);
    }

  //! Find a factory corresponding that can handle a particular filename
  inline Factory const & 
    find_factory(const std::string& filename)
    {
      return this->find_factory(FileSignature(filename), filename);
    }

  //! Find a factory corresponding that can handle a particular stream
  inline Factory const & 
    find_factory(std::istream& input)
    {
      return this->find_factory(FileSignature(input), input);
    }

  //! List all possible registered names to the stream
  /*! Names are separated with newlines. */
  inline void 
    list_registered_names(std::ostream& stream) const;
		 
 private:
  typedef typename std::multimap<unsigned, FactorySPtr > _registry_type;
  typedef typename _registry_type::const_iterator const_iterator;
  typedef typename _registry_type::iterator iterator;

  _registry_type _registry;

  bool _valid(const_iterator iter)
  { return iter != this->_registry.end(); }

  // File can be either string or istream
  template <class File>
    inline 
    const_iterator 
    _actual_find_factory(const FileSignature& signature,
			 File& input) const
    {
      const_iterator iter = this->_registry.begin();
      const_iterator const end = this->_registry.end();
      while (iter != end)
	{
	  if (iter->second->can_read(signature, input))
	    return iter;
	  ++iter;
	}
      return end;
    }

};

//! A helper class to allow automatic registration to the default InputFileFormatRegistry
/*! \ingroup IO
  If you have a variable of type 
  RegisterInputFileFormat\<MyInputFileFormat\>, \c MyInputFileFormat will be added to the
  default registry, as long as that variable is not destructed.

  This is used in STIR in the IO/IO_registries.cxx file.
*/
template <class Format>
struct RegisterInputFileFormat
{
  typedef typename Format::data_type data_type;
  //! constructor adds the \c Format to the registry with the given \c ranking.
  explicit RegisterInputFileFormat(const unsigned ranking)
  {
    shared_ptr<InputFileFormat<data_type> > format_sptr(new Format);
#ifndef NDEBUG
    info(boost::format("Adding %1% to input-file-format registry") % format_sptr->get_name());
#endif
    InputFileFormatRegistry<data_type>::default_sptr()->
      add_to_registry(format_sptr, ranking);
  }
   
  /*! \brief Destructor remove the format from the registry.
  */
  ~RegisterInputFileFormat()
  {
    Format format;
#ifndef NDEBUG
    info(boost::format("Removing %1% from input-file-format registry") % format.get_name());
#endif
    InputFileFormatRegistry<data_type>::default_sptr()->
      remove_from_registry(format);
  }
};

template <class DataT, class File>
inline 
std::auto_ptr<DataT>
read_from_file(const FileSignature& signature, File file)
{
  const InputFileFormat<DataT>& factory = 
    InputFileFormatRegistry<DataT>::default_sptr()->
    find_factory(signature, file);
#ifndef NDEBUG
  info(boost::format("Reading using file format %1%") % factory.get_name());
#endif
  return factory.read_from_file(file);
}

template <class DataT, class File>
inline
std::auto_ptr<DataT>
read_from_file(File file)
{
  const FileSignature signature(file);
  return read_from_file<DataT>(signature, file);
}


END_NAMESPACE_STIR

#include "stir/IO/InputFileFormatRegistry.inl"

#endif
