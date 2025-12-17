// //
// //
// /*!
//   \file
//   \ingroup normalisation

//   \brief Declaration of class stir::BinNormalisationFromPETSIRD

//   \author Nikos Efthimiou
// */
// /*
//     Copyright (C) 2025, UMCG
//     This file is part of STIR.

//     SPDX-License-Identifier: Apache-2.0

//     See STIR/LICENSE.txt for details
// */


// #ifndef __stir_recon_buildblock_BinNormalisationFromPETSIRD_H__
// #define __stir_recon_buildblock_BinNormalisationFromPETSIRD_H__

// #include "stir/recon_buildblock/BinNormalisation.h"
// #include "stir/recon_buildblock/BinNormalisationWithCalibration.h"
// #include "stir/RegisteredParsingObject.h"
// #include "stir/ProjData.h"
// #include "stir/shared_ptr.h"

// using std::string;

// START_NAMESPACE_STIR

// class BinNormalisationFromPETSIRD : public RegisteredParsingObject<BinNormalisationFromPETSIRD, BinNormalisation, BinNormalisationWithCalibration>
// {
// private:
//   using base_type = BinNormalisation; 

//   public:
//   //! Name which will be used when parsing a BinNormalisation object
//   static const char* const registered_name;

//   BinNormalisationFromPETSIRD(); 

//   BinNormalisationFromPETSIRD(const std::string& filename);

//   Succeeded set_up(const shared_ptr<const ExamInfo>& exam_info_sptr, const shared_ptr<const ProjDataInfo>&) override;


//   private: 

//   void set_defaults() override;

//   void initialise_keymap() override;

//   bool post_processing() override;

//   void read_norm_data(const string& filename);

//   string normalisation_filename;

// }; 


// END_NAMESPACE_STIR

// #endif
