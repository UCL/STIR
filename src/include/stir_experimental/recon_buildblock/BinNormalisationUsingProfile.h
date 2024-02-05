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

START_NAMESPACE_STIR

class BinNormalisationUsingProfile : 
  public RegisteredParsingObject<BinNormalisationUsingProfile, BinNormalisation>
{
public:
  //! Name which will be used when parsing a BinNormalisation object
  static const char * const registered_name; 

  BinNormalisationUsingProfile();

  BinNormalisationUsingProfile(const std::string& filename);

  void apply(RelatedViewgrams<float>& viewgrams) const override;

  void undo(RelatedViewgrams<float>& viewgrams) const override;

  float get_bin_efficiency(const Bin& bin) const override { return 1;}
 
private:
  mutable Array<1,float> profile;
  std::string profile_filename;

  void set_defaults() override;
  void initialise_keymap() override;
  bool post_processing() override;
};

END_NAMESPACE_STIR

#endif
  
