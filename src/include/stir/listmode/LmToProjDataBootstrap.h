//
//
/*!

  \file
  \ingroup listmode
  \brief Class for binning list mode files with the bootstrap method
    
  \author Kris Thielemans
      
*/
/*
    Copyright (C) 2003- 2011, Hammersmith Imanet
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

#ifndef __stir_listmode_LmToProjDataBootstrap_H__
#define __stir_listmode_LmToProjDataBootstrap_H__


#include "stir/listmode/LmToProjData.h"
#include <vector>

START_NAMESPACE_STIR

/*! \ingroup listmode
  \brief Class for binning list mode data into projection data using the 
  bootstrap procedure.

  The bootstrap method allows estimating the variance of an estimator
  based on a single data-set (magic!). This class can be used to
  generate multiple equivalent projdata, which can then be reconstructed.
  The sample variance computed on these images will be an estimate of 
  the variance on the reconstructions.

  For list mode data, bootstrapping works by selecting random events
  from the list mode file with replication (i.e. some events will occur
  more than once, other events will not occur at all).

  There are various papers on the bootstrap method. For PET data, it was
  for example applied by I. Buvat. (TODO add references)

  The pseudo-random number generator used is boost::mt19937 which should be
  good enough for most purposes. However, it can be easily replaced by any
  generator that follows the boost conventions.

  \par Parsing
  This class implements just one keyword in addition to those made
  available by its base type.
  \verbatim
  ; an unsigned int (but not 0) to seed the pseudo-random number generator
  seed := 42 ; default value
  \endverbatim
  \par Notes for developers

  This class is templated in terms of a LmToProjDataT to allow
  it to be used with different derived classes of LmToProjData. After all, the
  bootstrapping mechanism does not depend on how LmToProjData actually works.

*/
template< typename LmToProjDataT>
class LmToProjDataBootstrap : public LmToProjDataT
{

public:
     
  //! Constructor that parses from a file
  LmToProjDataBootstrap(const char * const par_filename);
  //! Constructor that parses from a file but with explicit seed
  /*! The \a seed argument will override any value found in the par file */
  LmToProjDataBootstrap(const char * const par_filename, const unsigned int seed);

protected:
  //! will be called when a new time frame starts
  /*! Initialises a vector with the number of times each event has to be replicated */
  virtual void start_new_time_frame(const unsigned int new_frame_num);

  virtual void get_bin_from_event(Bin& bin, const CListEvent&) const;


  // \name parsing variables
  //@{
  //! used to seed the pseudo-random number generator
  /*! should be non-zero */
  unsigned int seed;
  //@}

private:
  typedef LmToProjDataT base_type;
  typedef std::vector<unsigned char> replication_type;


  replication_type num_times_to_replicate;
  mutable replication_type::const_iterator num_times_to_replicate_iter;
  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();

  
};

END_NAMESPACE_STIR


#endif
