//
// $Id$
//
/*!
  \file
  \ingroup recon_buildblock

  \brief Declaration of class BinNormalisationUsingProfile

  \author Kris Thielemans
  $Date$
  $Revision$
*/

#ifndef __Tomo_recon_buildblock_BinNormalisationUsingProfile_H__
#define __Tomo_recon_buildblock_BinNormalisationUsingProfile_H__

#include "tomo/recon_buildblock/BinNormalisation.h"
#include "tomo/RegisteredParsingObject.h"
#include "RelatedViewgrams.h"

#include <string>

#ifndef TOMO_NO_NAMESPACE
using std::string;
#endif

START_NAMESPACE_TOMO

class BinNormalisationUsingProfile : 
  public RegisteredParsingObject<BinNormalisationUsingProfile, BinNormalisation>
{
public:
  //! Name which will be used when parsing a BinNormalisation object
  static const char * const registered_name; 

  BinNormalisationUsingProfile();

  BinNormalisationUsingProfile(const string& filename);

  virtual void apply(RelatedViewgrams<float>& viewgrams) const;

  virtual void undo(RelatedViewgrams<float>& viewgrams) const;
private:
  mutable Array<1,float> profile;
  string profile_filename;

  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();
};

END_NAMESPACE_TOMO

#endif
  
