//
// $Id$
//
/*!

  \file
  \ingroup listmode
  \brief Class for rebinning listmode files with the bootstrap method
    
  \author Kris Thielemans
      
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2003- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#ifndef __stir_listmode_LmToProjDataBootstrap_H__
#define __stir_listmode_LmToProjDataBootstrap_H__


#include "local/stir/listmode/LmToProjData.h"
#include <vector>

#ifndef STIR_NO_NAMESPACES
using std::vector;
#endif

START_NAMESPACE_STIR

template< typename LmToProjData>
class LmToProjDataBootstrap : public LmToProjData
{

public:
     
  LmToProjDataBootstrap(const char * const par_filename);
  LmToProjDataBootstrap(const char * const par_filename, const unsigned int seed);

  virtual void get_bin_from_event(Bin& bin, const CListEvent&) const;


private:
  typedef LmToProjData base_type;

  unsigned int seed;

  vector<unsigned char> num_times_to_replicate;
  mutable vector<unsigned char>::const_iterator num_times_to_replicate_iter;
  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();

  
};

END_NAMESPACE_STIR


#endif
