//
// $Id$
//
/*!
  \file
  \ingroup recon_buildblock

  \brief Implementations for class BinNormalisationUsingProfile

  \author Kris Thielemans
  $Date$
  $Revision$
*/


#include "local/tomo/recon_buildblock/BinNormalisationUsingProfile.h"
#include "RelatedViewgrams.h"

START_NAMESPACE_TOMO

const char * const 
  BinNormalisationUsingProfile::
  registered_name = "Using Profile"; 

void BinNormalisationUsingProfile::set_defaults()
{}//TODO

void BinNormalisationUsingProfile::initialise_keymap()
{}

BinNormalisationUsingProfile::
BinNormalisationUsingProfile()
{
  set_defaults();
}

BinNormalisationUsingProfile::
BinNormalisationUsingProfile(const string& filename)
    : profile(  Array<1,float>(-40,39) )
{
  ifstream profile_data(filename.c_str());
  for (int i=profile.get_min_index(); i<=profile.get_max_index(); ++i)
    profile_data >> profile[i];
  if (!profile_data)
    error("Error reading profile %s\n", filename.c_str());
}

void 
BinNormalisationUsingProfile::
apply(RelatedViewgrams<float>& viewgrams) const 
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
	  (*viewgrams_iter)[ax_pos_num] *= profile;
	}
    }
}

void
BinNormalisationUsingProfile::
undo(RelatedViewgrams<float>& viewgrams) const 
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
	  (*viewgrams_iter)[ax_pos_num] /= profile;
	}
    }
}


END_NAMESPACE_TOMO
  
