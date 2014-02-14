//
//
/*!
  \file
  \ingroup recon_buildblock

  \brief Declaration of class BinNormalisationUsingProfile

  \author Kris Thielemans
*/
/*
    Copyright (C) 2000- 2003, IRSL
    See STIR/LICENSE.txt for details
*/

#ifndef __stir_recon_buildblock_BinNormalisationUsingProfile_H__
#define __stir_recon_buildblock_BinNormalisationUsingProfile_H__

#include "stir/recon_buildblock/BinNormalisation.h"
#include "stir/RegisteredParsingObject.h"
#include "stir/RelatedViewgrams.h"

#include <string>

#ifndef STIR_NO_NAMESPACE
using std::string;
#endif

START_NAMESPACE_STIR

class BinNormalisationUsingProfile : 
  public RegisteredParsingObject<BinNormalisationUsingProfile, BinNormalisation>
{
public:
  //! Name which will be used when parsing a BinNormalisation object
  static const char * const registered_name; 

  BinNormalisationUsingProfile();

  BinNormalisationUsingProfile(const string& filename);

  virtual void apply(RelatedViewgrams<float>& viewgrams,const double start_time, const double end_time) const;

  virtual void undo(RelatedViewgrams<float>& viewgrams,const double start_time, const double end_time) const;

  virtual float get_bin_efficiency(const Bin& bin,const double start_time, const double end_time) const { return 1;}
 
private:
  mutable Array<1,float> profile;
  string profile_filename;

  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();
};

END_NAMESPACE_STIR

#endif
  
