/*
    Copyright (C) 2019 University of Hull
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup listmode SimSET
  \brief Implementation of class stir::CListModeDataSimSET

  \author Nikos Efthimiou
*/

#include "stir/listmode/CListModeDataSimSET.h"
#include "stir/Scanner.h"
#include "stir/Succeeded.h"
#include "stir/FilePath.h"
#include "stir/info.h"
#include "stir/warning.h"
#include "stir/error.h"

#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>

extern "C" {
#include <PhgBin.h>
}

START_NAMESPACE_STIR

//! N.E: Large parts adapted from phgbin and functions called by it;
CListModeDataSimSET::
CListModeDataSimSET(const std::string& _phg_filename)
    : phg_filename(_phg_filename)
{
    set_defaults();

    double SubObjCurTimeBinDuration = 0.0;

    // The dirty trick we did in SimSETListmodeInputFileFormat::is_SimSET_signature()
    char** argv = new char*[3];
    argv[0] = nullptr;
    argv[1] = nullptr;
    argv[2] = nullptr;

    char* pseudo_binary = new char[7];
    memset(pseudo_binary, 0, 7);
    strcpy(pseudo_binary, "phgbin\0");
    argv[0] = pseudo_binary;

    char* flag = new char[4];
    memset(flag, 0, 4);
    strcpy(flag, "-p");
    flag[3] = '\0';
    argv[1] = flag;

    char *argv_c = new char[phg_filename.size() + 1];
    memset(argv_c, 0, phg_filename.size()+1);
    phg_filename.copy(argv_c, phg_filename.size());
    argv_c[phg_filename.size()] = '\0';
    argv[2] = argv_c;

    char *knownOptions = new char[4];
    strcpy(knownOptions, "pcd");
    knownOptions[3] = '\0';

    char optArgs[PHGRDHST_NumFlags][LBEnMxArgLen];
    LbUsFourByte optArgFlags = (LBFlag0);
    LbUsFourByte phgrdhstArgIndex;

    /* Perform initialization tasks */

    /* Get our options */
    if (!LbEnGetOptions(3, argv, &knownOptions,
                        &PhgOptions, optArgs, optArgFlags, &phgrdhstArgIndex))
    {
        error("CListModeDataSimSET: Unable to get options.");
    }

    /* Make sure the didn't specify more than one history file */
    if (PHGRDHST_IsUsePHGHistory() && (PHGRDHST_IsUseColHistory() || PHGRDHST_IsUseDetHistory()))
    {
        error("CListModeDataSimSET: You can only specify one type of history file.");
    }

    /* Make sure they specified a history file */
    // N.E.: This should never be the case
    if (!PHGRDHST_IsUsePHGHistory() && !PHGRDHST_IsUseColHistory() && !PHGRDHST_IsUseDetHistory())
    {
        error("CListModeDataSimSET: You must specify the use of a PHG history file (-p)\n"
              " or a collimator history file (-c)\n"
              " or a detector history file (-d)\n");
    }

    strcpy(PhgRunTimeParams.PhgParamFilePath,argv[2]);

    /* Get our run-time parameters */
    if (!PhgGetRunTimeParams())
        error("CListModeDataSimSET: Error geting our run-time parameters.");

    /* If user requested to bin PHG history file, use the one specified in
     the param file.
     */
    if (PHGRDHST_IsUsePHGHistory() &&
            (strlen(PhgRunTimeParams.PhgPhoHFileHistoryFilePath) == 0))
    {

        error("CListModeDataSimSET: No history file supplied in run time parameters.");
    }
    else {
        strcpy(phgrdhstHistName, PhgRunTimeParams.PhgPhoHFileHistoryFilePath);
        strcpy(phgrdhstHistParamsName, PhgRunTimeParams.PhgPhoHParamsFilePath);
    }

    if (PhgRunTimeParams.PhgIsPET != 1)
        error("CListModeDataSimSET: Currently we are able to process only PET files.");

    /* Initialize the math library - NE: skipped*/
    /* Initialize the emission list manager - NE: skipped*/
    /* Initialize the sub-object manager - NE: partial*/

    /* Set the length of the current time bin */
    SubObjCurTimeBinDuration = static_cast<double>(PhgRunTimeParams.Phg_LengthOfScan);

    /* Initialize the productivity table manager - NE: skipped*/
    /* Initialize the Cylinder Positions */
    {
        /* Set object cylinder */
        if (!CylPosInitObjectCylinder())
        {
            error("CListModeDataSimSET: Error initialize the Cylinder Positions.");
        }

        /* Set the criticial zone */
        if (!CylPosInitCriticalZone(PhgRunTimeParams.Phg_AcceptanceAngle))
        {
            error("CListModeDataSimSET: Error setting the criticial zone.");
        }

        /* Set the limit cylinder */
        CylPosInitLimitCylinder();
    }

    /* Setup up the productivity information - NE: Reluctantly skipped */
    /* Initialize the photon tracking module - NE: skipped */
    // In the following part several nonPET and simulation parts, are skipped.
    /* We support only PET files so this should leed to an error always. */
    ColCurParams = 0;
    if (PHG_IsCollimateOnTheFly())
    {
        error("CListModeDataSimSET: This history files has collimator information. \n"
              "Currently we support only PET simulations.");
    }

    /* Initialize the detection module if necessary */
    if (PHG_IsDetectOnTheFly())
    {
        if (!DetInitialize(PHGRDHST_IsUsePHGHistory()))
            error("CListModeDataSimSET: Unable to initialize the detection module.");
    }

    /* Initialize the binning module if necessary */
    /* Initialize parameters */
    if (PhgBinInitParams(PhgRunTimeParams.PhgBinParamsFilePath[0],
                         &PhgBinParams[0], &PhgBinData[0], &PhgBinFields[0]) == false)
        error("CListModeDataSimSET: Unable to initialize the bining module.");


    // Try to guess scanner
    // CylPosTargetCylinder holds the goemetric information on the scanner.
    // //   double			radius;		/* Radius of cylinder */
    // //	double			zMin;		/* Minimum z coordinate of cylinder */
    // //	double			zMax;		/* Maximum z coordinate of cylinder */
    // //	double			centerX;	/* Center point on x axis */
    // //	double			centerY;	/* Center point on y axis */

    //DetRunTimeParamsTy: Timing information
    // //    double					PhotonTimeFWHM;							/* Photon time resolution (nanoseconds) */
    // //    double					EnergyResolutionPercentage;				/* Energy resolution in percentage */
    // //    double					ReferenceEnergy;						/* Energy resolution in percentage */
    // //    double					CoincidenceTimingWindowNS;	/* coincidence timing window in nanoseconds */

    // DetRunTimeParams : Cylindrical detector
    // LayerInfo: InnerRadius
    // LayerInfo: OutterRadius
    // NumLayers
    if (PhgRunTimeParams.PhgIsPET == false)
        error("CListModeDataSimSET: Only PET scanners are supported.");


    // We have already established that a cylindrical scanner will be used.
    shared_ptr<Scanner> tmpl_scanner;
    // SimplePET
    if (DetRunTimeParams[0].DetectorType == 1)
    {
        if (check_scanner_match_geometry( PhgBinParams->numTDBins,
                                          PhgBinParams->numZBins,
                                          tmpl_scanner) == Succeeded::yes)
        {
            std::string msg("\n**************************************************************\n"
                            "CListModeDataSimSET: Based on the information harvested by the PHG"
                            "params file and the Bining file we believe that the best matching"
                            "scanner is: ");
            msg.append(tmpl_scanner->get_name());
            msg.append("!\n**************************************************************\n");
            info(msg);
        }
        else
        {
            error("CListModeDataSimSET: The information harvested from the PHG file and Bining file "
                  "do not match a scanner in the Scanner list.");
        }
    }
    else if (DetRunTimeParams[0].DetectorType == 6) // CylindricalPET
    {
        uint chk_rings = 0 ;

        if (DetRunTimeParams[0].CylindricalDetector.NumRings == 1)
            chk_rings = PhgBinParams->numZBins;
        else
        {
            chk_rings =
                    static_cast<int>(static_cast<float>(DetRunTimeParams[0].CylindricalDetector.NumRings)/2.F + 0.5F);
        }

        if (check_scanner_match_geometry( PhgBinParams->numTDBins,
                                          chk_rings,
                                          tmpl_scanner,
                                          DetRunTimeParams[0].CylindricalDetector.RingInfo->LayerInfo->InnerRadius,
                                          DetRunTimeParams[0].CylindricalDetector.RingInfo->MinZ,
                                          DetRunTimeParams[0].CylindricalDetector.RingInfo->MaxZ,
                                          DetRunTimeParams[0].CylindricalDetector.RingInfo->NumLayers,
                                          DetRunTimeParams->EnergyResolutionPercentage,
                                          DetRunTimeParams->ReferenceEnergy) == Succeeded::yes)
        {
            std::string msg("\n**************************************************************\n"
                            "CListModeDataSimSET: Based on the information harvested by the PHG"
                            "params file and the Bining file we believe that the best matching"
                            "scanner is: ");
            msg.append(tmpl_scanner->get_name());
            msg.append("!\n**************************************************************\n");
            info(msg);
        }
        else
        {
            error("CListModeDataSimSET: The information harvested from the PHG file and Bining file "
                  "do not match a scanner in the Scanner list.");
        }
    }
    else
    {
        error("CListModeDataSimSET: Only cylindricalPET and simple PET scanners are supported.");
    }

    // ExamInfo initialisation
    this->exam_info_sptr.reset(new ExamInfo);

    // Only PET scanners supported
    this->exam_info_sptr->imaging_modality = ImagingModality::PT;
    this->exam_info_sptr->originating_system = this->originating_system;
    this->exam_info_sptr->set_low_energy_thres(static_cast<float>(PhgBinParams->minE));
    this->exam_info_sptr->set_high_energy_thres(static_cast<float>(PhgBinParams->maxE));


    // initialise ProjData.
    shared_ptr<ProjDataInfo> tmp( ProjDataInfo::construct_proj_data_info(tmpl_scanner,
                                                                         1,
                                                                         tmpl_scanner->get_num_rings()-1,
                                                                         tmpl_scanner->get_num_detectors_per_ring()/2,
                                                                         tmpl_scanner->get_max_num_non_arccorrected_bins(),
                                                                         /* arc_correction*/false));
    this->set_proj_data_info_sptr(tmp);

    // if the ProjData have been initialised properly create a
    // Input Stream from SimSET.
    history_file_sptr.reset(new InputStreamFromSimSET());
    std::string tmpString(phgrdhstHistName);
    history_file_sptr->set_up(tmpString,
                              &PhgBinParams[0],
            exam_info_sptr->get_low_energy_thres(),
            exam_info_sptr->get_high_energy_thres());

    // Clean up.
    delete [] argv;
    delete [] pseudo_binary;
    delete [] argv_c;
    delete [] flag;
    delete [] knownOptions;
}

std::string
CListModeDataSimSET::
get_name() const
{
    return phg_filename;
}

CListModeDataSimSET::~CListModeDataSimSET()
{
}

shared_ptr <CListRecord>
CListModeDataSimSET::
get_empty_record_sptr() const
{
    shared_ptr<CListRecord> sptr(new CListRecordSimSET(this->get_proj_data_info_sptr()->get_scanner_sptr()));
    return sptr;
}

Succeeded
CListModeDataSimSET::
open_lm_file()
{
//    info(boost::format("CListModeDataSimSET: used History file %s") %
//         this->history_file_sptr->get_total_number_of_events());
    return Succeeded::yes;
}



Succeeded
CListModeDataSimSET::
get_next_record(CListRecord& record_of_general_type) const
{
    CListRecordSimSET& record = dynamic_cast<CListRecordSimSET&>(record_of_general_type);
    return history_file_sptr->get_next_record(record);
}

Succeeded
CListModeDataSimSET::
reset()
{
    return history_file_sptr->reset();
}

unsigned long CListModeDataSimSET::get_total_number_of_events() const
{
        return history_file_sptr->get_total_number_of_events();
}

CListModeData::SavedPosition
CListModeDataSimSET::
save_get_position()
{
    return static_cast<SavedPosition>(history_file_sptr->save_get_position());
}

Succeeded
CListModeDataSimSET::
set_get_position(const CListModeDataSimSET::SavedPosition& pos)
{
    return history_file_sptr->set_get_position(pos);
}

void
CListModeDataSimSET::
set_defaults()
{
    PhgBinFields[0].NumCoincidences = 0;
    PhgBinFields[0].NumAcceptedCoincidences = 0;
    PhgBinFields[0].TotBluePhotons = 0;
    PhgBinFields[0].TotPinkPhotons = 0;
    PhgBinFields[0].AccBluePhotons = 0;
    PhgBinFields[0].AccPinkPhotons = 0;
    PhgBinFields[0].AccCoincidenceWeight = 0;
    PhgBinFields[0].AccCoincidenceSquWeight = 0;
    PhgBinFields[0].StartAccCoincidenceWeight = 0;
    PhgBinFields[0].StartAccCoincidenceSquWeight = 0;

    LbHdrStNull(&PhgBinFields[0].CountImgHdrHk);
    PhgBinFields[0].CountFile = nullptr;
    PhgBinData[0].countImage = nullptr;
    LbHdrStNull(&PhgBinFields[0].WeightSquImgHdrHk);
    PhgBinFields[0].WeightSquFile = nullptr;
    PhgBinData[0].weightSquImage = nullptr;
    LbHdrStNull(&PhgBinFields[0].WeightImgHdrHk);
    PhgBinFields[0].WeightFile = nullptr;
    PhgBinData[0].weightImage = nullptr;

    /* Clear the file name parameters */
    phgrdhstHistParamsName[0] = '\0';
    phgrdhstHistName[0] = '\0';
}

Succeeded
CListModeDataSimSET::
check_scanner_match_geometry(const unsigned int _numTDBins,
                             const unsigned int _numZbins,
                             shared_ptr<Scanner> &scanner_sptr,
                             const double _radius,
                             const double _minZ,
                             const double _maxZ,
                             const unsigned int _numLayers,
                             const double _enResolution,
                             const double _enResReference)
{

    std::string all_scanners = Scanner::list_all_names();

    std::vector<std::string> names;
    boost::split(names, all_scanners, [](char c){return c == '\n';});

    for ( int scanInt = Scanner::Type::E931;
          scanInt != Scanner::Type::User_defined_scanner; ++scanInt )
    {
        Scanner::Type cur_scanner = static_cast<Scanner::Type>(scanInt);
        scanner_sptr.reset(new Scanner(cur_scanner));

        if ((_radius > 0.0 ?
             scanner_sptr->get_inner_ring_radius() == static_cast<float>(_radius*10) : 1) &&
                scanner_sptr->get_num_detector_layers() == static_cast<int>(_numLayers) &&
                scanner_sptr->get_num_rings() == static_cast<int>(_numZbins) &&
                (scanner_sptr->get_energy_resolution() > -1.f ?
                 scanner_sptr->get_energy_resolution() == static_cast<float>(_enResolution) : 1) &&
                scanner_sptr->get_num_detectors_per_ring() == 2*static_cast<int>(_numTDBins) &&
                scanner_sptr->get_max_num_non_arccorrected_bins() == static_cast<int>(_numTDBins))
            return Succeeded::yes;
    }

    return Succeeded::no;
}

END_NAMESPACE_STIR
