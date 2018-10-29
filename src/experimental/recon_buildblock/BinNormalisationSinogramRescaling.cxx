//
//
/*!
  \file
  \ingroup recon_buildblock
  \ingroup 

  \brief Implementation for class BinNormalisationSinogramRescaling

  \author Sanida Mustafovic
*/
/*
    Copyright (C) 2003- 2004, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/


#include "stir_experimental/recon_buildblock/BinNormalisationSinogramRescaling.h"
#include "stir/shared_ptr.h"
#include "stir/RelatedViewgrams.h"
#include "stir/ViewSegmentNumbers.h"
#include "stir/IndexRange.h"
#include "stir/Bin.h"
#include "stir/stream.h"
#include "stir/Succeeded.h"
#include <algorithm>
#include <fstream>

#ifndef STIR_NO_NAMESPACES
using std::ifstream;
#endif

START_NAMESPACE_STIR


const char * const 
BinNormalisationSinogramRescaling::registered_name = "Sinogram Rescaling"; 

void 
BinNormalisationSinogramRescaling::set_defaults()
{
  sinogram_rescaling_factors_filename = "";
}

void 
BinNormalisationSinogramRescaling::
initialise_keymap()
{
  parser.add_start_key("Bin Normalisation Sinogram Rescaling");
  parser.add_key("sinogram_rescaling_factors_filename", &sinogram_rescaling_factors_filename);
  parser.add_stop_key("End Bin Normalisation Sinogram Rescaling");
}

bool 
BinNormalisationSinogramRescaling::
post_processing()
{
  if (sinogram_rescaling_factors_filename.size()==0)
   {
      warning("You have to specify sinogram rescaling filename\n");
      return true;
    }

  return false;
}


BinNormalisationSinogramRescaling::
BinNormalisationSinogramRescaling()
{
  set_defaults();
}

BinNormalisationSinogramRescaling::
BinNormalisationSinogramRescaling(const string& filename) 
  : sinogram_rescaling_factors_filename(filename)
{
}

Succeeded
BinNormalisationSinogramRescaling::
set_up(const shared_ptr<ProjDataInfo>& proj_data_info_sptr_v)
{
  proj_data_info_sptr =  proj_data_info_sptr_v;
 
  const int min_segment_num = proj_data_info_sptr->get_min_segment_num();
  const int max_segment_num = proj_data_info_sptr->get_max_segment_num();

  // empty data. get out quickly!
  if (max_segment_num < min_segment_num)
    return Succeeded::yes;
 
  ifstream input(sinogram_rescaling_factors_filename.c_str());
  input >> rescaling_factors;

  if (!input)
    {
      warning("Error reading rescaling factors from %s\n",
	      sinogram_rescaling_factors_filename.c_str());
      return Succeeded::no;
    }

  rescaling_factors.set_offset(min_segment_num);
  bool something_wrong = 
    rescaling_factors.get_max_index() != max_segment_num;

  for (int segment_num = min_segment_num; !something_wrong && segment_num<=max_segment_num; ++segment_num)
    {
      const int min_axial_pos_num =
	proj_data_info_sptr->get_min_axial_pos_num(segment_num);
      const int max_axial_pos_num =
	proj_data_info_sptr->get_max_axial_pos_num(segment_num);
      rescaling_factors[segment_num].set_offset(min_axial_pos_num);
      something_wrong = 
	something_wrong ||
	rescaling_factors[segment_num].get_max_index() != max_axial_pos_num;
      for (int axial_pos_num = min_axial_pos_num; !something_wrong && axial_pos_num<=max_axial_pos_num; ++axial_pos_num)
	{
	  rescaling_factors[segment_num][axial_pos_num].set_offset(proj_data_info_sptr->get_min_view_num());
	  something_wrong = 
	    something_wrong ||
	    rescaling_factors[segment_num][axial_pos_num].get_max_index() != proj_data_info_sptr->get_max_view_num();
	}
    }
  if (something_wrong)
    {
      warning("rescaling factors (%s) have wrong sizes for this projdata\n",
	    sinogram_rescaling_factors_filename.c_str());
      return Succeeded::no;
    }

  return Succeeded::yes;
}

float 
BinNormalisationSinogramRescaling::
get_bin_efficiency(const Bin& bin, const double /*start_time*/, const double /*end_time*/) const 
{
  return rescaling_factors[bin.segment_num()][bin.axial_pos_num()][bin.view_num()];

}


void 
BinNormalisationSinogramRescaling::
apply(RelatedViewgrams<float>& viewgrams,const double start_time, const double end_time) const 
{
  for (RelatedViewgrams<float>::iterator iter = viewgrams.begin(); iter != viewgrams.end(); ++iter)
  {

    Bin bin(iter->get_segment_num(),iter->get_view_num(), 0, 0);
    for (bin.axial_pos_num()= iter->get_min_axial_pos_num(); bin.axial_pos_num()<=iter->get_max_axial_pos_num(); ++bin.axial_pos_num())
      for (bin.tangential_pos_num()= iter->get_min_tangential_pos_num(); bin.tangential_pos_num()<=iter->get_max_tangential_pos_num(); ++bin.tangential_pos_num())
         (*iter)[bin.axial_pos_num()][bin.tangential_pos_num()] *= 
	   get_bin_efficiency(bin,start_time, end_time);
  }

}

void 
BinNormalisationSinogramRescaling::
undo(RelatedViewgrams<float>& viewgrams,const double start_time, const double end_time) const 
{
  for (RelatedViewgrams<float>::iterator iter = viewgrams.begin(); iter != viewgrams.end(); ++iter)

  {
    Bin bin(iter->get_segment_num(),iter->get_view_num(), 0,0);
    for (bin.axial_pos_num()= iter->get_min_axial_pos_num(); bin.axial_pos_num()<=iter->get_max_axial_pos_num(); ++bin.axial_pos_num())
      for (bin.tangential_pos_num()= iter->get_min_tangential_pos_num(); bin.tangential_pos_num()<=iter->get_max_tangential_pos_num(); ++bin.tangential_pos_num())
	 (*iter)[bin.axial_pos_num()][bin.tangential_pos_num()] /= 
	   std::max(1.E-20F,get_bin_efficiency(bin, start_time, end_time));

  }

}


END_NAMESPACE_STIR

