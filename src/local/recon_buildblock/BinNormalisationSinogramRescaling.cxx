//
// $Id: 
//
/*!
  \file
  \ingroup local recon_buildblock
  \ingroup 

  \brief Implementation for class BinNormalisationSinogramRescaling

  \author Sanida Mustafovic
  $Date:
  $Revision: 
*/
/*
    Copyright (C) 2002- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/


#include "local/stir/recon_buildblock/BinNormalisationSinogramRescaling.h"
#include "stir/DetectionPosition.h"
#include "stir/DetectionPositionPair.h"
#include "stir/IO/stir_ecat7.h"
#include "stir/shared_ptr.h"
#include "stir/RelatedViewgrams.h"
#include "stir/Sinogram.h"
#include "stir/ViewSegmentNumbers.h"
#include "stir/IndexRange2D.h"
#include "stir/IndexRange.h"
#include "stir/Bin.h"
#include "stir/display.h"
#include "stir/is_null_ptr.h"
#include "stir/stream.h"
#include <algorithm>
#include <fstream>

#ifndef STIR_NO_NAMESPACES
using std::ofstream;
using std::fstream;
#endif

START_NAMESPACE_STIR


const char * const 
BinNormalisationSinogramRescaling::registered_name = "Sinogram Rescaling"; 

void 
BinNormalisationSinogramRescaling::set_defaults()
{
  sinogram_rescaling_factors_filename = "";
  template_proj_data_filename ="";
}

void 
BinNormalisationSinogramRescaling::
initialise_keymap()
{
  parser.add_start_key("Bin Normalisation Sinogram Rescaling");
  parser.add_key("sinogram_rescaling_factors_filename", &sinogram_rescaling_factors_filename);
  parser.add_key("template_proj_data_filename", &template_proj_data_filename);
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

   if (template_proj_data_filename.size()==0)
   {
      warning("You have to specify sinogram rescaling filename\n");
      return true;
    }

  template_proj_data_sptr =
    ProjData::read_from_file(template_proj_data_filename);

  read_rescaling_factors(sinogram_rescaling_factors_filename);
  return false;
}


BinNormalisationSinogramRescaling::
BinNormalisationSinogramRescaling()
{
  set_defaults();
}

BinNormalisationSinogramRescaling::
BinNormalisationSinogramRescaling(const string& filename) 
{
 read_rescaling_factors(sinogram_rescaling_factors_filename);
}

Succeeded
BinNormalisationSinogramRescaling::
set_up(const shared_ptr<ProjDataInfo>& proj_data_info_sptr)
{
   
  if (*(template_proj_data_sptr->get_proj_data_info_ptr()) == *proj_data_info_sptr)
    return Succeeded::yes;
  else
  {
    warning("BinNormalisationSinogramRescaling: incompatible projection data\n");
    return Succeeded::no;
  }

}

void 
BinNormalisationSinogramRescaling::read_rescaling_factors (const string& filename)
{
 
  const ProjDataInfo* proj_data_info_sptr = template_proj_data_sptr->get_proj_data_info_ptr();
  const int min_seg_num = proj_data_info_sptr->get_min_segment_num();
  const int max_seg_num = proj_data_info_sptr->get_max_segment_num();
  const int max_axial_pos_in_seg_zero = proj_data_info_sptr->get_max_axial_pos_num(0);
  
  rescaling_factors =
    Array<2,float>(IndexRange2D(min_seg_num, max_seg_num, 
				0, max_axial_pos_in_seg_zero));
  open_read_binary(instream,filename.c_str());
  rescaling_factors.read_data(instream);
  cerr << rescaling_factors << endl;

#if 0
  for ( int i =0; i<=0; i++)
   for ( int j=0; j<=94; j++)
     {
       cerr << i << "   " << j << "    " << rescaling_factors[i][j]<< "      "<<endl;
     }
#endif

}

float 
BinNormalisationSinogramRescaling::
get_bin_efficiency(const Bin& bin, const double start_time, const double end_time) const 
{
  const int axial_pos_num = bin.axial_pos_num();
  const int segment_num = bin.segment_num();
 
  return rescaling_factors[segment_num][axial_pos_num];

}


void 
BinNormalisationSinogramRescaling::apply(RelatedViewgrams<float>& viewgrams,const double start_time, const double end_time) const 
{
  for (RelatedViewgrams<float>::iterator iter = viewgrams.begin(); iter != viewgrams.end(); ++iter)
  {
 //   if (iter->get_view_num()>8)
 //     continue;

    Bin bin(iter->get_segment_num(),iter->get_view_num(), 0, 0);
    for (bin.axial_pos_num()= iter->get_min_axial_pos_num(); bin.axial_pos_num()<=iter->get_max_axial_pos_num(); ++bin.axial_pos_num())
      for (bin.tangential_pos_num()= iter->get_min_tangential_pos_num(); bin.tangential_pos_num()<=iter->get_max_tangential_pos_num(); ++bin.tangential_pos_num())

	 (*iter)[bin.axial_pos_num()][bin.tangential_pos_num()] /= 
#ifndef STIR_NO_NAMESPACES
         std::
#endif
	 max(1.E-20F,get_bin_efficiency(bin, start_time, end_time));

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
         (*iter)[bin.axial_pos_num()][bin.tangential_pos_num()] *= get_bin_efficiency(bin,start_time, end_time);
  }

}


END_NAMESPACE_STIR

