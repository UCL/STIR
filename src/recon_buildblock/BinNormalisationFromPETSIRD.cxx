// //
// //
// /*
//     Copyright (C) 2025, UMCG
//     This file is part of STIR.

//     SPDX-License-Identifier: Apache-2.0

//     See STIR/LICENSE.txt for details
// */
// /*!
//   \file
//   \ingroup normalisation

//   \brief Implementation for class stir::BinNormalisationFromPETSIRD

//   \author Nikos Efthimiou
// */

// #include "stir/recon_buildblock/BinNormalisationFromPETSIRD.h"

// START_NAMESPACE_STIR

// const char* const BinNormalisationFromPETSIRD::registered_name = "From PETSIRD";

// void
// BinNormalisationFromPETSIRD::set_defaults()
// {
//   base_type::set_defaults();
//   normalisation_filename = "";
// }

// void
// BinNormalisationFromPETSIRD::initialise_keymap()
// {
//   base_type::initialise_keymap();
//   parser.add_start_key("Bin Normalisation From PETSIRD");
//   parser.add_key("normalisation_filename", &normalisation_filename);
//   parser.add_stop_key("End Bin Normalisation From PETSIRD");
// }

// bool
// BinNormalisationFromPETSIRD::post_processing()
// {
//   if (base_type::post_processing())
//     return true;
//   read_norm_data(normalisation_filename);
// }

// BinNormalisationFromPETSIRD::BinNormalisationFromPETSIRD()
// {
//   set_defaults();
// }

// BinNormalisationFromPETSIRD::BinNormalisationFromPETSIRD(const std::string& filename)
// {
//   read_norm_data(filename);
// }

// Succeeded
// BinNormalisationFromPETSIRD::set_up(const shared_ptr<const ExamInfo>& exam_info_sptr,
//                                     const shared_ptr<const ProjDataInfo>& proj_data_info_ptr_v)
// {
//   base_type::set_up(exam_info_sptr, proj_data_info_ptr_v);
// }

// void
// BinNormalisationFromPETSIRD::read_norm_data(const string& filename)
// {
//     normalisation_filename = filename;
// }

// END_NAMESPACE_STIR