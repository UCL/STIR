//
// $Id$
//
/*!

  \file
  \ingroup recon_buildblock

  \brief Declaration of class stir::RelatedBins

  \author Sanida Mustafovic
  \author Kris Thielemans
  \author PARAPET project

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
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
#ifndef __RelatedBins_H__
#define __RelatedBins_H__

#include "stir/shared_ptr.h"
#include <vector>
#include <iterator>

#ifndef STIR_NO_NAMESPACES
using std::size_t;
using std::ptrdiff_t;
using std::random_access_iterator_tag;
using std::vector;
#endif

START_NAMESPACE_STIR

class ProjData;
class Bin;
class DataSymmetriesForBins;
/*! 
  \ingroup recon_buildblock
  \brief This class contains all information about a set of bins related 
  by symmetry.
*/

class RelatedBins 
{
public:
 //! typedefs for iterator support


  typedef random_access_iterator_tag iterator_category;  
  typedef Bin value_type;
  typedef value_type& reference;
  typedef const value_type& const_reference;
  typedef ptrdiff_t difference_type;
  typedef size_t size_type;

  //! typedefs to make it partly comply with STL requirements
#ifndef STIR_NO_NAMESPACES
  typedef std::vector<Bin>::iterator iterator;
  typedef std::vector<Bin>::const_iterator const_iterator;
#else
  typedef vector<Bin>::iterator iterator;
  typedef vector<Bin>::const_iterator const_iterator;
#endif
   //!Default constructor: creates no bins, no symmetries  
  inline  RelatedBins();

  //! get the number of related bins
  inline int get_num_related_bins() const;

  //! get 'basic' bin coordinates
  inline Bin get_basic_bin() const;

  // get the pointer to a ProjDataInfo class 
  // inline const ProjDataInfo * get_proj_data_info_ptr() const;

  //! return the symmetries used
  inline const DataSymmetriesForBins* get_symmetries_ptr() const ;
 
  //! get an empty copy
  RelatedBins get_empty_copy() const;

 // basic iterator support

  //! use to initialise an iterator to the first element of the vector
   inline iterator begin();
   //! iterator 'past' the last element of the vector
   inline iterator end();
    //! use to initialise an iterator to the first element of the (const) vector
   inline const_iterator begin() const;
   //! iterator 'past' the last element of the (const) vector
   inline const_iterator end() const;



private:
   vector<Bin> related_bins;
   shared_ptr<DataSymmetriesForBins> symmetries;
     //! a private constructor which sets the members
  inline RelatedBins(const vector<Bin>& related_bins,
                     const shared_ptr<DataSymmetriesForBins>& symmetries_used);
 

};

END_NAMESPACE_STIR

#include "stir/recon_buildblock/RelatedBins.inl"

#endif //__RelatedBins_H__


