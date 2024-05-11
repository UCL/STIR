/*
    Copyright (C) 2022 University College London
    Copyright (C) 2022 Positrigo
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \brief Interface file for SWIG: stir::BinNormalisation hierarchy

  \author Kris Thielemans
  \author Markus Jehl

*/

%shared_ptr(stir::BinNormalisation);
%shared_ptr(stir::RegisteredObject<stir::BinNormalisation>);
%shared_ptr(stir::RegisteredParsingObject<stir::BinNormalisationFromProjData, stir::BinNormalisation>);
%shared_ptr(stir::BinNormalisationFromProjData);
%shared_ptr(stir::RegisteredParsingObject<stir::BinNormalisationFromAttenuationImage, stir::BinNormalisation>);
%shared_ptr(stir::BinNormalisationFromAttenuationImage);
%shared_ptr(stir::RegisteredParsingObject<stir::TrivialBinNormalisation, stir::BinNormalisation>);
%shared_ptr(stir::TrivialBinNormalisation);

%include "stir/recon_buildblock/BinNormalisation.h"

%template (internalRPBinNormalisationFromProjData) stir::RegisteredParsingObject<
  stir::BinNormalisationFromProjData, stir::BinNormalisation>;
%include "stir/recon_buildblock/BinNormalisationFromProjData.h"

%template (internalRPBinNormalisationFromAttenuationImage) stir::RegisteredParsingObject<
  stir::BinNormalisationFromAttenuationImage, stir::BinNormalisation>;
%include "stir/recon_buildblock/BinNormalisationFromAttenuationImage.h"

%template (internalRPTrivialBinNormalisation) stir::RegisteredParsingObject<
  stir::TrivialBinNormalisation, stir::BinNormalisation>;
%include "stir/recon_buildblock/TrivialBinNormalisation.h"
