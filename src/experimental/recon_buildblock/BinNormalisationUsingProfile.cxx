//
//
/*!
  \file
  \ingroup recon_buildblock

  \brief Implementations for class BinNormalisationUsingProfile

  \author Kris Thielemans
*/
/*
    Copyright (C) 2000- 2003, Hammersmith Imanet
    See STIR/LICENSE.txt for details
*/


#include "stir_experimental/recon_buildblock/BinNormalisationUsingProfile.h"
#include "stir/RelatedViewgrams.h"
#include "stir/stream.h"

START_NAMESPACE_STIR

const char * const 
  BinNormalisationUsingProfile::
  registered_name = "Using Profile"; 

void BinNormalisationUsingProfile::set_defaults()
{
  profile_filename = "";
}

void BinNormalisationUsingProfile::initialise_keymap()
{
  parser.add_start_key("Bin Normalisation Using Profile");
  parser.add_key("normalisation_profile_filename", &profile_filename);
  parser.add_stop_key("End Bin Normalisation Using Profile");
}

bool 
BinNormalisationUsingProfile::
post_processing()
{
#if 0
  profile = Array<1,float>(-114,113);
  ifstream profile_data(profile_filename.c_str());
  for (int i=profile.get_min_index(); i<=profile.get_max_index(); ++i)
    profile_data >> profile[i];
#else
  ifstream profile_data(profile_filename.c_str());
  profile_data >> profile;
  profile.set_offset(-(profile.get_length()/2));
#endif
  if (!profile_data)
    {
      warning("Error reading profile %s\n", profile_filename.c_str());
      return true;
    }
  return false;
}


BinNormalisationUsingProfile::
BinNormalisationUsingProfile()
{
  set_defaults();
}

BinNormalisationUsingProfile::
BinNormalisationUsingProfile(const string& filename)
  : profile_filename(filename)
{
  if (post_processing()==true)
    error("Exiting\n");
}

void 
BinNormalisationUsingProfile::
apply(RelatedViewgrams<float>& viewgrams,const double start_time, const double end_time) const 
{
  RelatedViewgrams<float>::iterator viewgrams_iter = 
    viewgrams.begin();
  for (; 
       viewgrams_iter != viewgrams.end();
       ++viewgrams_iter)
    {
      for (int ax_pos_num=viewgrams_iter->get_min_index();
	   ax_pos_num<=viewgrams_iter->get_max_index();
	   ++ax_pos_num)
	{
	  for (int i=std::max(profile.get_min_index(),viewgrams.get_min_tangential_pos_num());
	       i <= std::min(profile.get_max_index(),viewgrams.get_max_tangential_pos_num());
	       ++i)
	  (*viewgrams_iter)[ax_pos_num][i] *= profile[i];

	  // (*viewgrams_iter)[ax_pos_num] *= profile;
	}
    }
}

void
BinNormalisationUsingProfile::
undo(RelatedViewgrams<float>& viewgrams,const double start_time, const double end_time) const 
{
  /*
  const int old_min = profile.get_min_index();
  const int old_max = profile.get_max_index();
  profile.grow(viewgrams.get_min_tangential_pos_num(),viewgrams.get_max_tangential_pos_num());
  if (profile.get_min_index() != old_min ||
      profile.get_max_index() != old_max)
    warning("BinNormalisationUsingProfile: growing profile to (%d,%d)\n",
      profile.get_min_index(), profile.get_max_index());
  for (int i=profile.get_min_index(); i<old_min; ++i)
    profile[i] = 1.F;
  for (int i=profile.get_max_index(); i>old_max; --i)
    profile[i] = 1.F;
  */

  RelatedViewgrams<float>::iterator viewgrams_iter = 
    viewgrams.begin();
  for (; 
       viewgrams_iter != viewgrams.end();
       ++viewgrams_iter)
    {
      for (int ax_pos_num=viewgrams_iter->get_min_index();
	   ax_pos_num<=viewgrams_iter->get_max_index();
	   ++ax_pos_num)
	{
	  for (int i=std::max(profile.get_min_index(),viewgrams.get_min_tangential_pos_num());
	       i <= std::min(profile.get_max_index(),viewgrams.get_max_tangential_pos_num());
	       ++i)
	  (*viewgrams_iter)[ax_pos_num][i] /= profile[i];
	}
    }
}


END_NAMESPACE_STIR
  
