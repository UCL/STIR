//
// $Id$
//
/*!
  \file
  \ingroup recon_buildblock
  \ingroup ECAT

  \brief Implementation for class ChainedBinNormalisation

  \author Kris Thielemans
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2003- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/


#include "stir/recon_buildblock/ChainedBinNormalisation.h"
#include "stir/is_null_ptr.h"
#include "stir/Succeeded.h"

START_NAMESPACE_STIR

const char * const 
ChainedBinNormalisation::registered_name = "Chained"; 

void 
ChainedBinNormalisation::set_defaults()
{
  apply_first=apply_second=0;
}

void 
ChainedBinNormalisation::
initialise_keymap()
{
  parser.add_start_key("Chained Bin Normalisation Parameters");
  parser.add_parsing_key("Bin Normalisation to apply first", &apply_first);
  parser.add_parsing_key("Bin Normalisation to apply second", &apply_second);
  parser.add_stop_key("END Chained Bin Normalisation Parameters");}

bool 
ChainedBinNormalisation::
post_processing()
{
  return false;
}


ChainedBinNormalisation::
ChainedBinNormalisation()
{
  set_defaults();
}

ChainedBinNormalisation::
ChainedBinNormalisation(shared_ptr<ChainedBinNormalisation> const& apply_first_v,
		        shared_ptr<ChainedBinNormalisation> const& apply_second_v)
  : apply_first(apply_first_v),
    apply_second(apply_second_v)
{
  post_processing();
}

Succeeded
ChainedBinNormalisation::
set_up(const shared_ptr<ProjDataInfo>& proj_data_info_ptr)
{
  if (!is_null_ptr(apply_first))
    if (apply_first->set_up(proj_data_info_ptr) == Succeeded::no)
      return  Succeeded::no;
  if (!is_null_ptr(apply_second))
    return apply_second->set_up(proj_data_info_ptr);
  else
    return Succeeded::yes;  
}


void 
ChainedBinNormalisation::apply(RelatedViewgrams<float>& viewgrams,const double start_time, const double end_time) const 
{
  if (!is_null_ptr(apply_first))
    apply_first->apply(viewgrams,start_time,end_time);
  if (!is_null_ptr(apply_second))
    apply_second->apply(viewgrams,start_time,end_time);
}

void 
ChainedBinNormalisation::
undo(RelatedViewgrams<float>& viewgrams,const double start_time, const double end_time) const 
{
  if (!is_null_ptr(apply_first))
    apply_first->undo(viewgrams,start_time,end_time);
  if (!is_null_ptr(apply_second))
    apply_second->undo(viewgrams,start_time,end_time);
}

float
ChainedBinNormalisation:: get_bin_efficiency(const Bin& bin,const double start_time, const double end_time) const
{
  // TODO
  return 1;

}
 
 
END_NAMESPACE_STIR

