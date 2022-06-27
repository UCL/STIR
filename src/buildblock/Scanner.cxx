/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000 - 2010-07-21, Hammersmith Imanet Ltd
    Copyright (C) 2011, Kris Thielemans
    Copyright (C) 2010-2013, King's College London
    Copyright 2017 ETH Zurich, Institute of Particle Physics and Astrophysics
    Copyright (C) 2013-2016,2019-2021 University College London
    Copyright (C) 2017-2018, University of Leeds
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/
/*!

  \file
  \ingroup buildblock

  \brief Implementations for class stir::Scanner

  \author Nikos Efthimiou
  \author Charalampos Tsoumpas
  \author Sanida Mustafovic
  \author Kris Thielemans
  \author Claire Labbe
  \author Palak Wadhwa
  \author Ottavia Bertolli
  \author PARAPET project
  \author Parisa Khateri
*/

#include "stir/Scanner.h"
#include "stir/utilities.h"
#include "stir/Succeeded.h"
#include "stir/interfile_keyword_functions.h"
#include "stir/info.h"
#include "stir/DetectorCoordinateMap.h"
#include "stir/GeometryBlocksOnCylindrical.h"
#include <iostream>
#include <algorithm>
#ifdef BOOST_NO_STRINGSTREAM
#include <strstream.h>
#else
#include <sstream>
#endif


#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::cout;
using std::endl;
using std::cin;
using std::string;
using std::list;
#endif

START_NAMESPACE_STIR

// local convenience functions to make a list of strings
static list<string> 
   string_list(const string&);
static list<string> 
   string_list(const string&, const string&);
static list<string> 
   string_list(const string&, const string&, const string&);
static list<string> 
   string_list(const string&, const string&, const string&, const string&);
static list<string>
   string_list(const string&, const string&, const string&, const string&, const string&);


  
Scanner::Scanner(Type scanner_type)
  : _already_setup(false)
{

  // set_params parameters:
  //
  //            Type type_v,
  //            const list<string>& list_of_names_v,
  //
  //            int num_rings_v, 
  //            int max_num_non_arccorrected_bins_v,        
  // (optional) int default_num_arccorrected_bins_v,  
  //            int num_detectors_per_ring_v,
  //
  //            float inner_ring_radius_v,
  //            float average_depth_of_interaction_v,
  //            float ring_spacing_v,
  //            float bin_size_v, 
  //            float intrinsic_tilt_v,
  //
  //            int num_axial_blocks_per_bucket_v, 
  //            int num_transaxial_blocks_per_bucket_v, 
  //            int num_axial_crystals_per_block_v, 
  //            int num_transaxial_crystals_per_block_v,
  //            int num_axial_crystals_per_singles_unit_v,
  //            int num_transaxial_crystals_per_singles_unit_v,
  //            int num_detector_layers_v
  //

  
  /* for CTI scanners (at least upto 966):

    before arc-correction, central_bin_size ~= ring_radius* pi/num_detectors 
    num_transaxial_crystals_per_singles_unit= 
       transaxial_blocks_per_bucket*transaxial_crystals_per_block

    num_axial_crystals_per_singles_unit= 
       axial_crystals_per_block * x
    where x=1 except for the 966 where x=2
  */

  
  switch ( scanner_type ) {

  case E931:

    // KT 25/01/2002 corrected ring_spacing
    set_params(E931, string_list("ECAT 931"),  
               8, 192, 2 * 256, 
               510.0F, 7.0F, 13.5F, 3.129F, 0.0F, 
               2, 4, 4, 8, 4, 8 * 4, 1,
               0.37f, 511.f);
    // 16 BUCKETS per ring in TWO rings - i.e. 32 buckets in total

    break;

  case E951:

    set_params(E951, string_list("ECAT 951"), 
               16, 192, 2 * 256, 
               510.0F, 7.0F, 6.75F, 3.12932F, 0.0F, 
               1, 4, 8, 8, 8, 8 * 4, 1);
    break;

  case E953:

    set_params(E953, string_list("ECAT 953"), 
               16, 160, 2 * 192, 
               382.5F, 7.0F, 6.75F, 3.12932F, static_cast<float>(15.*_PI/180), 
               1, 4, 8, 8, 8, 8 * 4, 1);
    break;

  case E921:

    set_params(E921, string_list("ECAT 921", "ECAT EXACT", "EXACT"), 
               24, 192, 2* 192, 
               412.5F, 7.0F, 6.75F, 3.375F, static_cast<float>(15.*_PI/180),
               1, 4, 8, 8, 8, 8 * 4, 1);
    break;

  case E925:
    
    set_params(E925, string_list("ECAT 925", "ECAT ART"), 
               24, 192, 2* 192, 
               412.5F, 7.0F, 6.75F, 3.375F, static_cast<float>(15.*_PI/180),
               3, 4, 8, 8, 8, 8 * 4, 1);
    break;

  
  case E961:

    set_params(E961,string_list("ECAT 961", "ECAT HR"), 
               24, 336, 2* 392, 
               412.0F, 7.0F, 6.25F, 1.650F, static_cast<float>(13.*_PI/180),
               1, 8, 8, 7, 8, 7 * 8, 1);
    break;  

  case E962:

    set_params(E962,string_list("ECAT 962","ECAT HR+"), 
               32, 288, 2* 288, 
               412.0F, 7.0F, 4.85F, 2.25F,  0.0F, 
               4, 3, 8, 8, 8, 8 * 3, 1);
    break;

  case E966:

    set_params(E966, string_list("ECAT EXACT 3D", "EXACT 3D", "ECAT HR++","ECAT 966"), 
               48, 288, 2* 288, 
               412.0F, 7.0F, 4.850F, 2.250F, 0.0, 
               6, 2, 8, 8, 2 * 8, 8 * 2, 1);
    break;  

  case E1080:
    // data added by Robert Barnett, Westmead Hospital, Sydney
    set_params(E1080, string_list("ECAT 1080", "Biograph 16", "1080"),
               41, 336, 2* 336,
               412.0F, 7.0F, 4.0F, 2.000F, 0.0F,
               1, 2, 13+1, 13+1, 0, 0, 1);// TODO bucket/singles info?
    // Transaxial blocks have 13 physical crystals and a gap at the  
    // 14th crystal where the counts are zero.
    // There are 39 rings with 13 axial crystals per block. Two virtual
    // rings are added, but contain counts after applying axial compression.
    break;

  case Siemens_mMR:
    // 8x8 blocks, 1 virtual "crystal", 56 blocks along the ring, 8 blocks in axial direction
    // Transaxial blocks have 8 physical crystals and a gap at the  
    // 9th crystal where the counts are zero.
    set_params(Siemens_mMR, string_list("Siemens mMR", "mMR", "2008"),
               64, 344, 2* 252,
               328.0F, 7.0F, 4.0625F, 2.08626F, 0.0F,
               2, 1, 8, 9, 16, 9, 1,
               0.145f, 511.f); // TODO bucket/singles info incorrect? 224 buckets in total, but not sure how distributed
    break;

  case Siemens_mCT:
    // 13x13 blocks, 1 virtual "crystal" along axial and transaxial direction, 48 blocks along the ring, 4 blocks in axial direction
    set_params(Siemens_mCT, string_list("Siemens mCT", "mCT", "2011", "1104" /* used in norm files */, "1094" /* used in attenuation files */),
               55, 400, (13+1)*48,
               421.0F, 7.0F, 4.054F, 2.005F, 0.0F,
               4, 1, 13+1, 13+1, 0,0, 1 ); // TODO singles info incorrect
    // energy: 435-650
    // 13 TOF bins
    break;

  case RPT:
    
    set_params(RPT, string_list("PRT-1", "RPT"), 
               16, 128, 2 * 192, 
               380.0F - 7.0F, 7.0F, 6.75F, 3.1088F, 0.0F, 
               1, 4, 8, 8, 8, 32, 1);

    // Default 7.0mm average interaction depth.
    // This 7mm taken off the inner ring radius so that the effective radius remains 380mm
    break;    

  case RATPET:
    
    set_params(RATPET, string_list("RATPET"), 
               8, 56, 2 * 56, 
               115 / 2.F,  7.0F, 6.25F, 1.65F, 0.0F, 
               1, 16, 8, 7, 8, 0, 1); // HR block, 4 buckets per ring
    
    // Default 7.0mm average interaction depth.
    // 8 x 0 crystals per singles unit because not known 
    // although likely transaxial_blocks_per_bucket*transaxial_crystals_per_block
    break;

  case PANDA:
    
    set_params(PANDA, string_list("PANDA"), 
               1 /*NumRings*/, 512 /*MaxBinsNonArcCor*/, 512 /*MaxBinsArcCor*/, 2048 /*NumDetPerRing*/, 
               /*MeanInnerRadius*/ 75.5/2.F, /*AverageDoI*/ 10.F, /*Ring Spacing*/ 3.F, /*BinSize*/ 0.1F, /*IntrinsicTilt*/ 0.F, 
               1, 1, 1, 1, 0, 0, 1);     
    break;
		  
  case nanoPET:
		  
	  set_params(nanoPET, string_list("nanoPET"), /*Modelling the gap with one fake crystal */
				 81, 39*3, /* We could also model gaps in the future as one detector so 39->39+1, while 1 (point source), 3 (mouse) or 5 (rats) */
				 39*3, /* Just put the same with NonArcCor for now*/
				 12 * 39, 174.F,  5.0F, 1.17F, 1.17F, /* Actual size is 1.12 and 0.05 is the thickness of the optical reflector */ 0.0F, /* not sure for this */ 
				 0,0,0,0,0,0, 1);
	  break; 

  case HYPERimage:
		  
	  set_params(HYPERimage, string_list("HYPERimage"), /*Modelling the gap with one fake crystal */
				 22, 239, 245,
			     490, 103.97F, 3.0F, 1.4F, 1.4F, /* Actual size is 1.3667 and assume 0.0333 is the thickness of the optical reflector */  0.F,
				 0,0,0,0,0,0,1);
	  break; 
		  
		  
  case Advance:
    
    // 283 bins (non-uniform sampling) 
    // 281 bins (uniform sampling)
    /* crystal size 4x8x30*/
    set_params(Advance, string_list("GE Advance", "Advance"), 
               18, 283, 281, 2 * 336, 
               471.875F - 8.4F, 8.4F, 8.5F, 1.970177F, 0.0F, //TODO view offset shouldn't be zero
               3, 2, 6, 6, 1, 1, 1);
    break;  

  case DiscoveryLS:
    // identical to Advance
    set_params(DiscoveryLS, string_list("GE Discovery LS", "Discovery LS"), 
               18, 283, 281, 2 * 336, 
               471.875F - 8.4F, 8.4F, 8.5F, 1.970177F, 0.0F, //TODO view offset shouldn't be zero
               3, 2, 6, 6, 1, 1, 1);
    break;
  case DiscoveryST: 

    // 249 bins (non-uniform sampling) 
    // 221 bins (uniform sampling)
    /* crystal size: 6.3 x 6.3 x 30 mm*/
    set_params(DiscoveryST, string_list("GE Discovery ST", "Discovery ST"), 
	       24, 249, 221, 2 * 210,
               886.2F/2.F, 8.4F, 6.54F, 3.195F, 
	       static_cast<float>(-4.54224*_PI/180),//sign?
	       4, 2, 6, 6, 1, 1, 1);// TODO not sure about sign of view_offset
    break;

 case DiscoverySTE: 

    set_params(DiscoverySTE, string_list("GE Discovery STE", "Discovery STE"), 
           24, 329, 293, 2 * 280,
               886.2F/2.F, 8.4F, 6.54F, 2.397F,
	       static_cast<float>(-4.5490*_PI/180),//sign?
           4, 2, 6, 8, 1, 1, 1);// TODO not sure about sign of view_offset
    break;

 case DiscoveryRX: 

    set_params(DiscoveryRX, string_list("GE Discovery RX", "Discovery RX"), 
	       24, 
	       367, 
	       331, 
	       2 * 315,
               886.2F/2.F, 
	       9.4F,  
	       6.54F, 2.13F,
	       static_cast<float>(-4.5950*_PI/180),//sign?
	       4,
	       2,
	       6, 9, 1, 1, 1);// TODO not sure about sign of view_offset    
    break;

 case Discovery600: 

    set_params(Discovery600, string_list("GE Discovery 600", "Discovery 600"), 
	       24, 
	       339, 
	       293, // TODO
	       2 * 256,
               826.70F/2.F - 8.4F, 
	       8.4F,  
	       6.54F,
	       2.3974F,
	       static_cast<float>(-4.5490*_PI/180),//sign? TODO value
	       4,
	       2,
	       6, 8, 1, 1, 1);
    break;


case PETMR_Signa:

  set_params(PETMR_Signa, string_list("GE Signa PET/MR", "PET/MR Signa", "Signa PET/MR"),
	       45, 
	       357, 
	       331, // TODO
	       2 * 224,
           311.8F,
           8.5F,
           5.56F,
           2.01565F, // TO CHECK
	       static_cast<float>(-5.23*_PI/180),//sign? TODO value
	       5,
	       4,
             9, 4, 1, 1, 1,
             0.105F, // energy resolution from Levin et al. TMI 2016
             511.F);
break;

  case Discovery690:
    // same as 710
    set_params(Discovery690, string_list("GE Discovery 690", "Discovery 690",
                                         "GE Discovery 710", "Discovery 710"),
               24,
               381,
               331, // TODO
               2 * 288,
               405.1F,
               9.4F,
               6.54F,
               2.1306F,
               static_cast<float>(-5.021*_PI/180),//sign? TODO value
               4,
               2,
               6, 9, 1, 1, 1
#ifdef STIR_TOF
               ,
			   (short int)(55),
			   (float)(89.0F),
			   (float)(550.0F)
#endif
);

    break;
  

  case DiscoveryMI3ring: // This is the 3-ring DMI
    // Hsu et al. 2017 JNM
    // crystal size 3.95 x 5.3 x 25
    set_params(DiscoveryMI3ring, string_list("GE Discovery MI 3 rings", "Discovery MI3", "Discovery MI"), // needs to include last value as used by GE in RDF files
	       27,
	       415,
	       401, // TODO should compute num_arccorrected_bins from effective_FOV/default_bin_size
	       2 * 272,
               380.5F - 9.4F,//TODO inner_ring_radius and DOI, currently set such that effective ring-radius is correct
               9.4F,//TODO DOI
               5.52296F, // ring-spacing
               2.206F,//TODO currently using the central bin size default bin size. GE might be using something else
	       static_cast<float>(-4.399*_PI/180), //TODO check sign
	       3, 4,
	       9, 4,
               1, 1,
               1,
               0.0944F, // energy resolution from Hsu et al. 2017
               511.F);
    break;

  case DiscoveryMI4ring: // This is the 4-ring DMI
    // as above, but one extra block
    // Hsu et al. 2017 JNM
    // crystal size 3.95 x 5.3 x 25
    set_params(DiscoveryMI4ring, string_list("GE Discovery MI 4 rings", "Discovery MI4", "Discovery MI"), // needs to include last value as used by GE in RDF files
	       36,
	       415,
	       401, // TODO should compute num_arccorrected_bins from effective_FOV/default_bin_size
	       2 * 272,
               380.5F - 9.4F,//TODO inner_ring_radius and DOI, currently set such that effective ring-radius is correct
               9.4F,//TODO DOI
               5.52296F, // ring-spacing
               2.206F,//TODO currently using the central bin size default bin size. GE might be using something else
	       static_cast<float>(-4.399*_PI/180), //TODO check sign
	       4, 4,
	       9, 4,
               1, 1,
               1,
               0.0944F, // energy resolution from Hsu et al. 2017
               511.F);
    break;

    case DiscoveryMI5ring: // This is the 5-ring DMI
      // as above, but one extra block
      // Hsu et al. 2017 JNM
      // crystal size 3.95 x 5.3 x 25
      set_params(DiscoveryMI5ring, string_list("GE Discovery MI 5 rings", "Discovery MI5", "Discovery MI"), // needs to include last value as used by GE in RDF files
                 45,
                 415,
                 401, // TODO should compute num_arccorrected_bins from effective_FOV/default_bin_size
                 2 * 272,
                 380.5F - 9.4F,//TODO inner_ring_radius and DOI, currently set such that effective ring-radius is correct
                 9.4F,//TODO DOI
                 5.52296F, // ring-spacing
                 2.206F,//TODO currently using the central bin size default bin size. GE might be using something else
                 static_cast<float>(-4.399*_PI/180), //TODO check sign
                 5, 4,
                 9, 4,
                 1, 1,
                 1,
                 0.0944F, // energy resolution from Hsu et al. 2017
                 511.F);
      break;
  case HZLR:

    set_params(HZLR, string_list("Positron HZL/R"), 
               32, 256, 2 * 192, 
               780.0F, 7.0F, 5.1875F, 2.F, 0.0F, 
               0, 0, 0, 0, 0,0, 1);
    // Default 7.0mm average interaction depth.
    //  crystals per singles unit etc unknown
    break;

  case HRRT:

    set_params(HRRT, string_list("HRRT"), 
               104, 288, 2 * 288, 
               234.765F, 7.0F, 2.4375F, 1.21875F, 0.0F, 
               0, 0, 0, 0, 0, 0, 2); // added by Dylan Togane
    // warning: used 7.0mm average interaction depth.
    // crystals per singles unit etc unknown
    break;

  case Allegro:

    /* 
       The following info is partially from
 
       Journal of Nuclear Medicine Vol. 45 No. 6 1040-1049
       Imaging Characteristics of a 3-Dimensional GSO Whole-Body PET Camera 
       Suleman Surti, PhD and Joel S. Karp, PhD 
       http://jnm.snmjournals.org/cgi/content/full/45/6/1040

       Other info is from Ralph Brinks (Philips Research Lab, Aachen).
 
       The Allegro scanner is comprised of 28 flat modules of a 22 x 29 array
       of 4 x 6 x 20 mm3 GSO crystals. The output sinograms however consist
       of 23 x 29 logical crystals per module. 
       This creates problems for the current version of STIR as the current
       Scanner object does not support does. At present, KT put the 
       transaxial info on crystals to 0.
       For 662keV photons the mean positron range in GSO is about 14 mm,
       so we put in 12mm for 511 keV, but we don't really know.
       Ralph Brinks things there is only one singles rate for the whole
       scanner.
    */
    set_params(Allegro,string_list("Allegro", "Philips Allegro"), 
	       29, 295, 28*23, 
	       430.05F, 12.F, 
	       6.3F, 4.3F, 0.0F, 
	       1, 0, 
	       29, 0 /* 23* or 22*/,
	       29, 0 /*  all detectors in a ring? */, 
	       1);
    break;

  case GeminiTF:
    set_params(GeminiTF,string_list("GeminiTF", "Philips GeminiTF"), 
               44, 322, 287, // Based on GATE output - Normally it is 644 detectors at each of the 44 rings
               322*2, // Actual number of crystals is 644
               450.17F, 8.F, // DOI is from XXX et al 2008 MIC
               4.F, 4.F, 0.F, 
               0, 0, 
               0, 0, // Not considering any gap, but this is per module 28 flat modules in total, while 420 PMTs 
               0, 0 /*  Not sure about these, but shouldn't be important */, 
               1);
    break;

  case HiDAC: // all of these don't make any sense for the HiDAC
    set_params(HiDAC, string_list("HiDAC"), 
               0, 0, 0, 
               0.F, 0.F, 0.F, 0.F, 0.F, 
               0, 0, 0, 0, 0, 0, 0);
 
    break;

  case SAFIRDualRingPrototype: 
  set_params(SAFIRDualRingPrototype, string_list("SAFIRDualRingPrototype"), 
             16, //num_rings_v
             150, //max_num_non_arccorrected_bins_v,
             150, //default_num_arccorrected_bins_v,
             180, //num_detectors_per_ring_v    
             64.05, //  inner_ring_radius_v
             5, //average_depth_of_interaction_v
             2.2, //ring_spacing_v
             1.1, //bin_size_v
             0, //intrinsic_tilt_v
             2, //num_axial_blocks_per_bucket_v
             1, //num_transaxial_blocks_per_bucket_v
             8, //num_axial_crystals_per_block_v
             15, //num_transaxial_crystals_per_block_v
             1, //num_axial_crystals_per_singles_unit_v
             1, //num_transaxial_crystals_per_singles_unit_v
             1, //num_detector_layers_v
             -1, //energy_resolution_v
             -1, //reference_energy_v
             "", //scanner_geometry_v
             2.2, //axial_crystal_spacing_v
             2.2, //transaxial_crystal_spacing_v
             18.1, //axial_block_spacing_v
             33.6, //transaxial_block_spacing_v
             ""//crystal_map_file_name_v
            );  
  break;

  case UPENN_5rings:
    set_params(UPENN_5rings, string_list("UPENN_5rings"),
               (40+16)*5,
               331, 331,
               576+18,
               382.0F, 7.0F,
               3.9655, 2.0F,
               static_cast<float>(0),
               7,     //            int num_axial_blocks_per_bucket_v,
               4,     //            int num_transaxial_blocks_per_bucket_v,
               8,  //            int num_axial_crystals_per_block_v,
               8, //            int num_transaxial_crystals_per_block_v,
               8*7,  //            int num_axial_crystals_per_singles_unit_v,
               8 * 4, // +1 gap     //            int num_transaxial_crystals_per_singles_unit_v,
               1,
               0.109F, 511.F
           #ifdef STIR_TOF
               ,
               (short int)(512),
               (float)(19.53125),
               (float)(272.55F)
           #endif
);
    break;

case UPENN_6rings:
    set_params(UPENN_6rings, string_list("UPENN_6rings"),
               (40+16)*6,
               331, 331,
               576+18,
               382.0F, 7.0F,
               3.9655, 2.02035F,
               static_cast<float>(0),
               7 * 6,     //            int num_axial_blocks_per_bucket_v,
               1,     //            int num_transaxial_blocks_per_bucket_v,
               8,  //            int num_axial_crystals_per_block_v,
               33, //            int num_transaxial_crystals_per_block_v,
               8,  //            int num_axial_crystals_per_singles_unit_v,
               33, //int num_transaxial_crystals_per_singles_unit_v,
               1,
               0.109F, 511.F
           #ifdef STIR_TOF
               ,
               (short int)(512),
               (float)(19.53125),
               (float)(272.55F)
           #endif
);
    break;

  case UPENN_5rings_no_gaps:
    set_params(UPENN_5rings_no_gaps, string_list("UPENN_5rings_no_gaps"),
               40*5,
               301, 301,
               576,
               382.0F, 7.0F,
               3.9655, 2.08349F,
               static_cast<float>(0),
               7 * 5,     //            int num_axial_blocks_per_bucket_v,
               4,     //            int num_transaxial_blocks_per_bucket_v,
               8,  //            int num_axial_crystals_per_block_v,
               8, //            int num_transaxial_crystals_per_block_v,
               8,  //            int num_axial_crystals_per_singles_unit_v,
               8 * 4,  //            int num_transaxial_crystals_per_singles_unit_v,
               1,
               0.109F, 511.F
           #ifdef STIR_TOF
               ,
               (short int)(512),
               (float)(19.53125),
               (float)(272.55F)
           #endif
);
    break;

  case UPENN_6rings_no_gaps:
    set_params(UPENN_6rings_no_gaps, string_list("UPENN_6rings_no_gaps"),
               40 * 6,
               321, 321,
               576,
               382.0F, 7.0F,
               3.9655, 2.08F,
               static_cast<float>(0),
               5 * 6,     //            int num_axial_blocks_per_bucket_v,
               4,     //            int num_transaxial_blocks_per_bucket_v,
               8,  //            int num_axial_crystals_per_block_v,
               8, //            int num_transaxial_crystals_per_block_v,
               8,  //            int num_axial_crystals_per_singles_unit_v,
               8,  //            int num_transaxial_crystals_per_singles_unit_v,
               1,
               0.109F, 511.F
           #ifdef STIR_TOF
               ,
               (short int)(512),
               (float)(19.53125),
               (float)(272.55F)
           #endif
);
    break;
  
  case User_defined_scanner: // zlong, 08-04-2004, Userdefined support

    set_params(User_defined_scanner, string_list("Userdefined"), 
               0, 0, 0, 
               0.F, 0.F, 0.F, 0.F, 0.F, 
               0, 0, 0, 0, 0, 0, 0);
    
    break;

  default:
    // warning("Unknown scanner type used for initialisation of Scanner\n"); 
    set_params(Unknown_scanner, string_list("Unknown"), 
               0, 0, 0, 
               0.F, 0.F, 0.F, 0.F, 0.F, 
               0, 0, 0, 0, 0, 0, 0);
    
    break;
 
  }

}


Scanner::Scanner(Type type_v, const list<string>& list_of_names_v,
                 int num_detectors_per_ring_v, int num_rings_v, 
                 int max_num_non_arccorrected_bins_v,
                 int default_num_arccorrected_bins_v,
                 float inner_ring_radius_v, float average_depth_of_interaction_v, 
                 float ring_spacing_v, float bin_size_v, float intrinsic_tilt_v,
                 int num_axial_blocks_per_bucket_v, int num_transaxial_blocks_per_bucket_v,
                 int num_axial_crystals_per_block_v, int num_transaxial_crystals_per_block_v,
                 int num_axial_crystals_per_singles_unit_v, 
                 int num_transaxial_crystals_per_singles_unit_v,
                 int num_detector_layers_v,
                 float energy_resolution_v,
                 float reference_energy_v,
                 const string& scanner_geometry_v,
                 float axial_crystal_spacing_v,
                 float transaxial_crystal_spacing_v,
                 float axial_block_spacing_v,
                 float transaxial_block_spacing_v,
                 const std::string& crystal_map_file_name_v)
: _already_setup(false)
{
  set_params(type_v, list_of_names_v, num_rings_v,
             max_num_non_arccorrected_bins_v,
             default_num_arccorrected_bins_v,
             num_detectors_per_ring_v,
             inner_ring_radius_v,
             average_depth_of_interaction_v,
             ring_spacing_v, bin_size_v, intrinsic_tilt_v,
             num_axial_blocks_per_bucket_v, num_transaxial_blocks_per_bucket_v,
             num_axial_crystals_per_block_v, num_transaxial_crystals_per_block_v,
             num_axial_crystals_per_singles_unit_v,
             num_transaxial_crystals_per_singles_unit_v,
             num_detector_layers_v,
             energy_resolution_v,
             reference_energy_v,
             scanner_geometry_v,
             axial_crystal_spacing_v,
             transaxial_crystal_spacing_v,
             axial_block_spacing_v,
             transaxial_block_spacing_v,
             crystal_map_file_name_v);
}



Scanner::Scanner(Type type_v, const string& name,
                 int num_detectors_per_ring_v, int num_rings_v, 
                 int max_num_non_arccorrected_bins_v,
                 int default_num_arccorrected_bins_v,
                 float inner_ring_radius_v, float average_depth_of_interaction_v, 
                 float ring_spacing_v, float bin_size_v, float intrinsic_tilt_v,
                 int num_axial_blocks_per_bucket_v, int num_transaxial_blocks_per_bucket_v,
                 int num_axial_crystals_per_block_v, int num_transaxial_crystals_per_block_v,
                 int num_axial_crystals_per_singles_unit_v, 
                 int num_transaxial_crystals_per_singles_unit_v,
                 int num_detector_layers_v,
                 float energy_resolution_v,
                 float reference_energy_v,
                 const string& scanner_geometry_v,
                 float axial_crystal_spacing_v,
                 float transaxial_crystal_spacing_v,
                 float axial_block_spacing_v,
                 float transaxial_block_spacing_v,
                 const std::string& crystal_map_file_name_v)
  : _already_setup(false)
{
  set_params(type_v, string_list(name), num_rings_v,
             max_num_non_arccorrected_bins_v,
             default_num_arccorrected_bins_v,
             num_detectors_per_ring_v,
             inner_ring_radius_v,
             average_depth_of_interaction_v,
             ring_spacing_v, bin_size_v, intrinsic_tilt_v,
             num_axial_blocks_per_bucket_v, num_transaxial_blocks_per_bucket_v,
             num_axial_crystals_per_block_v, num_transaxial_crystals_per_block_v,
             num_axial_crystals_per_singles_unit_v,
             num_transaxial_crystals_per_singles_unit_v,
             num_detector_layers_v,
             energy_resolution_v,
             reference_energy_v,
             scanner_geometry_v,
             axial_crystal_spacing_v,
             transaxial_crystal_spacing_v,
             axial_block_spacing_v,
             transaxial_block_spacing_v,
             crystal_map_file_name_v);
}







void
Scanner::
set_params(Type type_v,const list<string>& list_of_names_v,
           int num_rings_v, 
           int max_num_non_arccorrected_bins_v,
           int num_detectors_per_ring_v,
           float inner_ring_radius_v,
           float average_depth_of_interaction_v,
           float ring_spacing_v,
           float bin_size_v, float intrinsic_tilt_v,
           int num_axial_blocks_per_bucket_v, int num_transaxial_blocks_per_bucket_v, 
           int num_axial_crystals_per_block_v, int num_transaxial_crystals_per_block_v,
           int num_axial_crystals_per_singles_unit_v,
           int num_transaxial_crystals_per_singles_unit_v,
           int num_detector_layers_v,
           float energy_resolution_v,
           float reference_energy_v,
           const string& scanner_geometry_v,
           float axial_crystal_spacing_v,
           float transaxial_crystal_spacing_v,
           float axial_block_spacing_v,
           float transaxial_block_spacing_v,
           const std::string& crystal_map_file_name_v)
{
  set_params(type_v, list_of_names_v, num_rings_v,
             max_num_non_arccorrected_bins_v,
	     max_num_non_arccorrected_bins_v,
	     num_detectors_per_ring_v, 
	     inner_ring_radius_v, 
             average_depth_of_interaction_v,
             ring_spacing_v, bin_size_v, intrinsic_tilt_v,
	     num_axial_blocks_per_bucket_v, num_transaxial_blocks_per_bucket_v,
	     num_axial_crystals_per_block_v, num_transaxial_crystals_per_block_v,
             num_axial_crystals_per_singles_unit_v, 
             num_transaxial_crystals_per_singles_unit_v,
	     num_detector_layers_v,
             energy_resolution_v,
             reference_energy_v,
             scanner_geometry_v,
             axial_crystal_spacing_v,
             transaxial_crystal_spacing_v,
             axial_block_spacing_v,
             transaxial_block_spacing_v,
             crystal_map_file_name_v);
}


void
Scanner::
set_params(Type type_v,const list<string>& list_of_names_v, 
           int num_rings_v, 
           int max_num_non_arccorrected_bins_v,
           int default_num_arccorrected_bins_v,
           int num_detectors_per_ring_v,
           float inner_ring_radius_v,
           float average_depth_of_interaction_v,
           float ring_spacing_v,
           float bin_size_v, float intrinsic_tilt_v,
           int num_axial_blocks_per_bucket_v, int num_transaxial_blocks_per_bucket_v, 
           int num_axial_crystals_per_block_v, int num_transaxial_crystals_per_block_v,
           int num_axial_crystals_per_singles_unit_v,
           int num_transaxial_crystals_per_singles_unit_v,
           int num_detector_layers_v,
           float energy_resolution_v,
           float reference_energy_v,
           const string& scanner_geometry_v,
           float axial_crystal_spacing_v,
           float transaxial_crystal_spacing_v,
           float axial_block_spacing_v,
           float transaxial_block_spacing_v,
           const std::string& crystal_map_file_name_v)
{
  type = type_v;
  list_of_names = list_of_names_v;  
  num_rings =  num_rings_v;
  max_num_non_arccorrected_bins = max_num_non_arccorrected_bins_v;
  default_num_arccorrected_bins = default_num_arccorrected_bins_v;
  num_detectors_per_ring = num_detectors_per_ring_v;    
  inner_ring_radius =  inner_ring_radius_v;
  average_depth_of_interaction = average_depth_of_interaction_v;
  ring_spacing = ring_spacing_v;
  bin_size = bin_size_v;
  intrinsic_tilt = intrinsic_tilt_v;	
  num_transaxial_blocks_per_bucket = num_transaxial_blocks_per_bucket_v;
  num_axial_blocks_per_bucket = num_axial_blocks_per_bucket_v;
  num_axial_crystals_per_block= num_axial_crystals_per_block_v;
  num_transaxial_crystals_per_block= num_transaxial_crystals_per_block_v;
  num_axial_crystals_per_singles_unit = num_axial_crystals_per_singles_unit_v;
  num_transaxial_crystals_per_singles_unit = num_transaxial_crystals_per_singles_unit_v;
  num_detector_layers = num_detector_layers_v;

  energy_resolution = energy_resolution_v;
  if (reference_energy_v <= 0)
      reference_energy = 511.f;
  else
      reference_energy = reference_energy_v;
  
  axial_crystal_spacing = axial_crystal_spacing_v;
  transaxial_crystal_spacing = transaxial_crystal_spacing_v;
  axial_block_spacing = axial_block_spacing_v;
  transaxial_block_spacing = transaxial_block_spacing_v;
  
  crystal_map_file_name = crystal_map_file_name_v;

  if (scanner_geometry_v == "")
    set_scanner_geometry("Cylindrical");
  else
    set_scanner_geometry(scanner_geometry_v);

  set_up();
}

void Scanner::set_scanner_geometry(const std::string& new_scanner_geometry)
{
  scanner_geometry = new_scanner_geometry;
   _already_setup = false;
}

void Scanner::set_up()
{
  if (scanner_geometry == "Generic")
    {
      if (!this->detector_map_sptr){
          if (crystal_map_file_name == "")          
        error("Scanner: scanner_geometry=Generic needs a crystal map");
      
      read_detectormap_from_file(crystal_map_file_name);
      }
    }
  else
    {
      if (crystal_map_file_name != "")
        error("Scanner: use scanner_geometry=Generic when specifying a crystal map");
      if (scanner_geometry == "BlocksOnCylindrical")
        this->detector_map_sptr.reset(new GeometryBlocksOnCylindrical(*this));
      else
        {
          this->detector_map_sptr = 0;
          if (scanner_geometry != "Cylindrical")
            error("Scanner::scanner_geometry needs to be one of Cylindrical, BlocksOnCylindrical, Generic");
        }
    }
  _already_setup = true;
}

void
Scanner::
set_detector_map( const DetectorCoordinateMap::det_pos_to_coord_type& coord_map )
{
  this->detector_map_sptr.reset(new DetectorCoordinateMap(coord_map));
  if ((unsigned)num_detectors_per_ring != detector_map_sptr->get_num_tangential_coords() ||
      (unsigned)num_rings != detector_map_sptr->get_num_axial_coords() ||
      (unsigned)num_detector_layers != detector_map_sptr->get_num_radial_coords())
      error("Scanner:set_detector_map: inconsistent number of detectors");
}

void
Scanner::
read_detectormap_from_file( const std::string& filename )
{
  this->detector_map_sptr.reset(new DetectorCoordinateMap(filename));
}

/*! \todo The current list is bound to be incomplete. would be better to stick it in set_params().
 */
int
Scanner::
get_num_virtual_axial_crystals_per_block() const
{
  switch(get_type())
    {
    case E1080:
    case Siemens_mCT:
      return 1;
    default:
      return 0;
    }
}

/*! \todo The current list is bound to be incomplete. would be better to stick it in set_params().
 */
int
Scanner::
get_num_virtual_transaxial_crystals_per_block() const
{
  switch(get_type())
    {
    case E1080:
    case Siemens_mCT:
    case Siemens_mMR:
    case UPENN_5rings:
    case UPENN_6rings:
      return 1;
    default:
      return 0;
    }
}
/*! \todo Can currently only set to hard-wired values. Otherwise calls error() */
void
Scanner::
set_num_virtual_axial_crystals_per_block(int val)
{
  //num_virtual_axial_crystals_per_block = val;
  if (this->get_num_virtual_axial_crystals_per_block() != val)
    error("Scanner::set_num_virtual_axial_crystals_per_block not really implemented yet");
}

/*! \todo Can currently only set to hard-wired values. Otherwise calls error() */
void
Scanner::
set_num_virtual_transaxial_crystals_per_block(int val)
{
  //num_virtual_transaxial_crystals_per_block = val;
  if (this->get_num_virtual_transaxial_crystals_per_block() != val)
    error("Scanner::set_num_virtual_transaxial_crystals_per_block not really implemented yet");
}


Succeeded 
Scanner::
check_consistency() const
{
  if (intrinsic_tilt<-_PI || intrinsic_tilt>_PI)
    warning("Scanner %s: intrinsic_tilt is very large. maybe it's in degrees (but should be in radians)",
	    this->get_name().c_str());

  {
    if (get_num_transaxial_crystals_per_block() <= 0 ||
	get_num_transaxial_blocks() <= 0)
      warning("Scanner %s: transaxial block info is not set (probably irrelevant unless you use a projector or normalisation that needs this block info)",
	      this->get_name().c_str());
    else
      {
	const int dets_per_ring =
	  get_num_transaxial_blocks() *
	  get_num_transaxial_crystals_per_block();
    // exclusion of generic as 'get_num_transaxial_crystals_per_block()' is sometimes false for asymmetric detectors and not important for generic
	if ( dets_per_ring != get_num_detectors_per_ring() && scanner_geometry != "Generic")
	  { 
	    warning("Scanner %s: inconsistent transaxial block info",
		    this->get_name().c_str()); 
	    return Succeeded::no; 
	  }
      }
  }
  {
    if (get_num_transaxial_blocks_per_bucket() <= 0 ||
	get_num_transaxial_buckets() <=0)
      warning("Scanner %s: transaxial bucket info is not set (probably irrelevant unless you use dead-time correction that needs this info)",
	      this->get_name().c_str());
    else
      {
	const int blocks_per_ring =
	  get_num_transaxial_buckets() *
	  get_num_transaxial_blocks_per_bucket();
    // exclusion of generic as 'get_num_transaxial_blocks_per_bucket()' is sometimes false for asymmetric detectors and not important for generic
	if ( blocks_per_ring != get_num_transaxial_blocks() && scanner_geometry != "Generic")
	  { 
	    warning("Scanner %s: inconsistent transaxial block/bucket info",
		    this->get_name().c_str()); 
	    return Succeeded::no; 
	  }
      }
  }
  {
    if (get_num_axial_crystals_per_block() <= 0 ||
	get_num_axial_blocks() <=0)
      warning("Scanner %s: axial block info is not set (probably irrelevant unless you use a projector or normalisation that needs this block info)",
	      this->get_name().c_str());
    else
      {
	const int dets_axial =
	  get_num_axial_blocks() *
	  get_num_axial_crystals_per_block();

	// exclusion of generic as 'get_num_axial_crystals_per_block()' is sometimes false for asymmetric detectors and not important for generic
  if ( dets_axial != (get_num_rings() + get_num_virtual_axial_crystals_per_block())  && scanner_geometry != "Generic")
	  { 
	    warning("Scanner %s: inconsistent axial block info: %d vs %d",
		    this->get_name().c_str(),
                    dets_axial, get_num_rings() + get_num_virtual_axial_crystals_per_block()); 
	    return Succeeded::no; 
	  }
      }
  }
  {
    if (get_num_axial_blocks_per_bucket() <= 0 ||
	get_num_axial_buckets() <=0)
      warning("Scanner %s: axial bucket info is not set (probably irrelevant unless you use dead-time correction that needs this info)",
	      this->get_name().c_str());
    else
      {
	const int blocks_axial =
	  get_num_axial_buckets() *
	  get_num_axial_blocks_per_bucket();
    // exclusion of generic as 'get_num_axial_blocks_per_bucket()' is sometimes false for asymmetric detectors and not important for generic
	if ( blocks_axial != get_num_axial_blocks() && scanner_geometry != "Generic")
	  { 
	    warning("Scanner %s: inconsistent axial block/bucket info",
		    this->get_name().c_str()); 
	    return Succeeded::no; 
	  }
      }
  }
  // checks on singles units
  {
    if (get_num_transaxial_crystals_per_singles_unit() <= 0)
      warning("Scanner %s: transaxial singles_unit info is not set (probably irrelevant unless you use dead-time correction that needs this info)",
	      this->get_name().c_str());
    else
      {
	if ( get_num_detectors_per_ring() % get_num_transaxial_crystals_per_singles_unit() != 0)
	  { 
	    warning("Scanner %s: inconsistent transaxial singles unit info:\n"
		    "\tnum_detectors_per_ring %d should be a multiple of num_transaxial_crystals_per_singles_unit %d",
		    this->get_name().c_str(),
		    get_num_detectors_per_ring(), get_num_transaxial_crystals_per_singles_unit()); 
	    return Succeeded::no; 
	  }
	if ( get_num_transaxial_crystals_per_bucket() % get_num_transaxial_crystals_per_singles_unit() != 0)
	  { 
	    warning("Scanner %s: inconsistent transaxial singles unit info:\n"
		    "\tnum_transaxial_crystals_per_bucket %d should be a multiple of num_transaxial_crystals_per_singles_unit %d",
		    this->get_name().c_str(),
		    get_num_transaxial_crystals_per_bucket(), get_num_transaxial_crystals_per_singles_unit()); 
	    return Succeeded::no; 
	  }
      }
  }
  {
    if (get_num_axial_crystals_per_singles_unit() <= 0)
      warning("Scanner %s: axial singles_unit info is not set (probably irrelevant unless you use dead-time correction that needs this info)",
	      this->get_name().c_str());
    else
      {
	if ( get_num_rings() % get_num_axial_crystals_per_singles_unit() != 0)
	  { 
	    warning("Scanner %s: inconsistent axial singles unit info:\n"
		    "\tnum_rings %d should be a multiple of num_axial_crystals_per_singles_unit %d",
		    this->get_name().c_str(),
		    get_num_rings(), get_num_axial_crystals_per_singles_unit()); 
	    return Succeeded::no; 
	  }
	if ( get_num_axial_crystals_per_bucket() % get_num_axial_crystals_per_singles_unit() != 0)
	  { 
	    warning("Scanner %s: inconsistent axial singles unit info:\n"
		    "\tnum_axial_crystals_per_bucket %d should be a multiple of num_axial_crystals_per_singles_unit %d",
		    this->get_name().c_str(),
		    get_num_axial_crystals_per_bucket(), get_num_axial_crystals_per_singles_unit()); 
	    return Succeeded::no; 
	  }
      }
  }
  
  if (get_scanner_geometry() == "BlocksOnCylindrical")
  {//! Check consistency of axial and transaxial spacing for block geometry
      if (get_axial_crystal_spacing()*get_num_axial_crystals_per_block() > get_axial_block_spacing())
      {
         warning("Scanner %s: inconsistent axial spacing:\n"
              "\taxial_crystal_spacing %f muliplied by num_axial_crystals_per_block %d should fit into axial_block_spacing %f",
                 this->get_name().c_str(),
           get_axial_crystal_spacing(), get_num_axial_crystals_per_block(), get_axial_block_spacing());
         return Succeeded::no;
        }
        if (get_transaxial_crystal_spacing()*get_num_transaxial_crystals_per_block() > get_transaxial_block_spacing())
        {
          warning("Scanner %s: inconsistent transaxial spacing:\n"
                "\ttransaxial_crystal_spacing %f muliplied by num_transaxial_crystals_per_block %d should fit into transaxial_block_spacing %f",
              this->get_name().c_str(),
              get_transaxial_crystal_spacing(), get_num_transaxial_crystals_per_block(), get_transaxial_block_spacing());
          return Succeeded::no;
        }
        
        if (round(get_transaxial_block_spacing()*get_num_transaxial_blocks_per_bucket()*1000.0)/1000.0
            < round (2*inner_ring_radius*tan(_PI/2/get_num_transaxial_buckets())*1000.0)/1000.0)
      {
         warning("Scanner %s: inconsistent transaxial spacing:\n"
              "\ttransaxial_block_spacing %f muliplied by num_transaxial_blocks_per_bucket %d should fit into a polygon that encircles a cylinder with inner_ring_radius %f",
                 this->get_name().c_str(),
                 get_transaxial_block_spacing(), get_num_transaxial_blocks_per_bucket(), get_inner_ring_radius());
         return Succeeded::no;
      }
    else if (get_scanner_geometry() == "Generic")
    { //! Check if the crystal map is correct and given
      if (get_crystal_map_file_name() == "")
      {
        warning("No crystal map is provided. The scanner geometry Generic needs it! Please provide one.");
        return Succeeded::no;
      }
      else
      {
        std::ifstream crystal_map(get_crystal_map_file_name());
        if( !crystal_map)
        {
          warning("No correct crystal map provided. Please check the file name.");
          return Succeeded::no;
        }
      }
    }
  
  }

  return Succeeded::yes;
}







// TODO replace by using boost::floating_point_comparison
bool static close_enough(const double a, const double b)
{
  return fabs(a-b) <= std::min(fabs(a), fabs(b)) * 10E-4;
}

bool 
Scanner::operator ==(const Scanner& scanner) const
{
if (!close_enough(energy_resolution, scanner.energy_resolution) &&
      !close_enough(reference_energy, scanner.reference_energy))
    warning("The energy resolution of the two scanners is different. \n"
            " %f opposed to %f"
            "This only affects scatter simulation. \n", energy_resolution, scanner.energy_resolution);

  return
      (num_rings == scanner.num_rings) &&
      (max_num_non_arccorrected_bins == scanner.max_num_non_arccorrected_bins) &&
      (default_num_arccorrected_bins == scanner.default_num_arccorrected_bins) &&
      (num_detectors_per_ring == scanner.num_detectors_per_ring) &&
      close_enough(inner_ring_radius, scanner.inner_ring_radius) &&
      close_enough(average_depth_of_interaction, scanner.average_depth_of_interaction) &&
      close_enough(ring_spacing, scanner.ring_spacing) &&
      close_enough(bin_size,scanner.bin_size) &&
      close_enough(intrinsic_tilt,scanner.intrinsic_tilt) &&
      close_enough(axial_crystal_spacing, scanner.axial_crystal_spacing) &&
      close_enough(transaxial_crystal_spacing, scanner.transaxial_crystal_spacing) &&
      close_enough(axial_block_spacing, scanner.axial_block_spacing) &&
      close_enough(transaxial_block_spacing, scanner.transaxial_block_spacing) &&
      (num_transaxial_blocks_per_bucket == scanner.num_transaxial_blocks_per_bucket) &&
      (num_axial_blocks_per_bucket == scanner.num_axial_blocks_per_bucket) &&
      (num_axial_crystals_per_block == scanner.num_axial_crystals_per_block) &&
      (num_transaxial_crystals_per_block == scanner.num_transaxial_crystals_per_block) &&
      (num_detector_layers == scanner.num_detector_layers) &&
      (num_axial_crystals_per_singles_unit == scanner.num_axial_crystals_per_singles_unit) &&
      (num_transaxial_crystals_per_singles_unit == scanner.num_transaxial_crystals_per_singles_unit);

}


const list<string>& 
Scanner::get_all_names() const
{return list_of_names;}


const string&
Scanner::get_name() const
{
  
 return *(list_of_names.begin()); 
    
}

string
Scanner::parameter_info() const
{
  // warning: these should match the parsing keywords in InterfilePDFSHeader
#ifdef BOOST_NO_STRINGSTREAM
  // dangerous for out-of-range, but 'old-style' ostrstream seems to need this
  char str[10000];
  ostrstream s(str, 10000);
#else
  std::ostringstream s;
#endif
  s << "Scanner parameters:= "<<'\n';

  s << "Scanner type := " << get_name() <<'\n';     

  s << "Number of rings                          := " << num_rings << '\n';
  s << "Number of detectors per ring             := " << get_num_detectors_per_ring() << '\n';

  s << "Inner ring diameter (cm)                 := " << get_inner_ring_radius()*2./10 << '\n'
    << "Average depth of interaction (cm)        := " << get_average_depth_of_interaction() / 10 << '\n'
    << "Distance between rings (cm)              := " << get_ring_spacing()/10 << '\n'
    << "Default bin size (cm)                    := " << get_default_bin_size()/10. << '\n'
    << "View offset (degrees)                    := " << get_intrinsic_azimuthal_tilt()*180/_PI << '\n';
  s << "Maximum number of non-arc-corrected bins := "
    << get_max_num_non_arccorrected_bins() << '\n'
    << "Default number of arc-corrected bins     := "
    << get_default_num_arccorrected_bins() << '\n';
  if (get_energy_resolution() >= 0 && get_reference_energy() >= 0)
  {
    s << "Energy resolution := " << get_energy_resolution() << '\n';
    s << "Reference energy (in keV) := " << get_reference_energy() << '\n';
  }

  // block/bucket description
  s << "Number of blocks per bucket in transaxial direction         := "
    << get_num_transaxial_blocks_per_bucket() << '\n'
    << "Number of blocks per bucket in axial direction              := "
    << get_num_axial_blocks_per_bucket() << '\n'
    << "Number of crystals per block in axial direction             := "
    << get_num_axial_crystals_per_block() << '\n'
    << "Number of crystals per block in transaxial direction        := "
    << get_num_transaxial_crystals_per_block() << '\n'
    << "Number of detector layers                                   := "
    << get_num_detector_layers() << '\n'
    << "Number of crystals per singles unit in axial direction      := "
    << get_num_axial_crystals_per_singles_unit() << '\n'
    << "Number of crystals per singles unit in transaxial direction := "
    << get_num_transaxial_crystals_per_singles_unit() << '\n';
  
  //block and generic geometry description
  if (crystal_map_file_name != "")
    s << "Name of crystal map                                         := "
      << get_crystal_map_file_name() << '\n';
  if (get_scanner_geometry() != "")
  {
    s << "Scanner geometry (BlocksOnCylindrical/Cylindrical/Generic)  := "
      <<get_scanner_geometry() << '\n';
  }
  if (get_axial_crystal_spacing() >=0)
    s << "Distance between crystals in axial direction (cm)           := "
      << get_axial_crystal_spacing()/10 << '\n';
  if (get_transaxial_crystal_spacing() >=0)
    s << "Distance between crystals in transaxial direction (cm)      := "
      << get_transaxial_crystal_spacing()/10 << '\n';
  if (get_axial_block_spacing() >=0)
    s << "Distance between blocks in axial direction (cm)             := "
      << get_axial_block_spacing()/10 << '\n';
  if (get_transaxial_block_spacing() >=0)
    s << "Distance between blocks in transaxial direction (cm)        := "
      << get_transaxial_block_spacing()/10 << '\n';

  s << "end scanner parameters:=\n";

  return s.str();
}

string Scanner::list_names() const
{
#ifdef BOOST_NO_STRINGSTREAM
  // dangerous for out-of-range, but 'old-style' ostrstream seems to need this
  char str[3000];
  ostrstream s(str, 3000);
#else
  std::ostringstream s;
#endif

#ifdef _MSC_VER
  // work-around VC bug
  std::
#endif
  list<string>::const_iterator iterator = list_of_names.begin(); 
  s << *iterator;
  ++iterator;
  while(iterator!=list_of_names.end())
  {
    s << " , " << *iterator ;
    ++iterator;
  }

  return s.str();
}

/************************************************
 static members
 *************************************************/
Scanner* Scanner::ask_parameters() 
{

  cerr << list_all_names();

  const string name=ask_string("Enter the name of the scanner");

  //get the type from the name itself
  Scanner* scanner_ptr = 
    get_scanner_from_name(name);

  // N.E: New optional parameters have been added, namely
  // energy resolution and timing resolution,
  // lets give users the chance to set these parameters on
  // old scanners. This should stay here as a transitional step.
  if (scanner_ptr->type != Unknown_scanner && scanner_ptr->type != User_defined_scanner)
    {
      info("more options are available for the scanner: \n(a) Energy Resolution :=\n(b) Reference energy (in keV)\t:="
        "\n(c) Scanner geometry ( BlocksOnCylindrical / Cylindrical / Generic ) \n(d) Scanner orientation (X or Y)\t:="
        "\n\n(a) and (b) are used in Scatter Simulation. \n (c) is used to choose more precise models of the scanner. "
        "\n(d) is used in BlocksOnCylindrical Geometry to build the proper crystal map."
        "\nIn case, you need them, set them manually in your interfile header before 'end scanner parameters:='.");
      
      //This is needed for finding effective central bin size, because it is different for different geometries.
      const string ScannerGeometry =
        ask_string("Enter the scanner geometry ( BlocksOnCylindrical / Cylindrical / Generic ) :", "Cylindrical");

      if (ScannerGeometry == "Generic")
      {
        string CrystalMapFileName = ask_string("Enter the name of the crystal map: ", "");
        scanner_ptr->set_crystal_map_file_name(CrystalMapFileName);
      }
  
      // will also read detector-map from file
      scanner_ptr->set_scanner_geometry(ScannerGeometry);

      return scanner_ptr;
    }

  if (scanner_ptr->type == Unknown_scanner)
    cerr << "I didn't recognise the scanner you entered.";
  cerr << "I'll ask lots of questions\n";
  
  while (true)
    {
      int num_detectors_per_ring = 
	ask_num("Enter number of detectors per ring:",0,2000,128);
  
      int NoRings = 
        ask_num("Enter number of rings :",0,1000,16);
  
      int NoBins = 
        ask_num("Enter default number of tangential positions for this scanner: ",0,3000,128);
  
      float InnerRingRadius=
	ask_num("Enter inner ring radius (in mm): ",0.F,600.F,256.F);
  
      float AverageDepthOfInteraction = 
        ask_num("Enter average depth of interaction (in mm): ", 0.F, 100.F, 0.F);
      
      float RingSpacing= 
        ask_num("Enter ring spacing (in mm): ",0.F,30.F,6.75F);
  
      float BinSize= 
        ask_num("Enter default (tangential) bin size after arc-correction (in mm):",0.F,60.F,3.75F);
      float intrTilt=
	ask_num("Enter intrinsic_tilt (in degrees):",-180.F,360.F,0.F);
      int TransBlocksPerBucket = 
	ask_num("Enter number of transaxial blocks per bucket: ",0,10,2);
      int AxialBlocksPerBucket = 
	ask_num("Enter number of axial blocks per bucket: ",0,10,6);
      int AxialCrystalsPerBlock = 
	ask_num("Enter number of axial crystals per block: ",0,16,8);
      int TransaxialCrystalsPerBlock = 
	ask_num("Enter number of transaxial crystals per block: ",0,16,8);
      int AxialCrstalsPerSinglesUnit = 
        ask_num("Enter number of axial crystals per singles unit: ", 0, NoRings, 1);
      int TransaxialCrystalsPerSinglesUnit = 
        ask_num("Enter number of transaxial crystals per singles unit: ", 0, num_detectors_per_ring, 1);
        
     float EnergyResolution =
          ask_num("Enter the energy resolution of the scanner : ", 0.0f, 1000.0f, -1.0f);

      float ReferenceEnergy =
          ask_num("Enter the reference energy for the energy resolution (in keV):", 0.0f, 1000.0f, -1.0f);

      int num_detector_layers =
    ask_num("Enter number of detector layers per block: ",1,100,1);
           
      const string ScannerGeometry =
  ask_string("Enter the scanner geometry ( BlocksOnCylindrical / Cylindrical / Generic ) :", "Cylindrical");
      
      float AxialCrystalSpacing=      
  ask_num("Enter crystal spacing in axial direction (in mm): ",0.F,30.F,6.75F);
      float TransaxialCrystalSpacing=
  ask_num("Enter crystal spacing in transaxial direction (in mm): ",0.F,30.F,6.75F);
      float AxialBlockSpacing=
  ask_num("Enter block spacing in axial direction (in mm): ",0.F,360.F,54.F);
      float TransaxialBlockSpacing=
  ask_num("Enter block spacing in transaxial direction (in mm): ",0.F,360.F,54.F);
  
  string crystal_map_file_name = "";
  if (ScannerGeometry == "Generic") {
      crystal_map_file_name =
        ask_string("Enter the name of the crystal map: ", "");
  }
  
      Type type = User_defined_scanner;
  
      scanner_ptr =
            new Scanner(type, string_list(name),
                        num_detectors_per_ring,  NoRings,
                        NoBins, NoBins,
                        InnerRingRadius, AverageDepthOfInteraction,
                        RingSpacing, BinSize,intrTilt*float(_PI)/180,
                        AxialBlocksPerBucket,TransBlocksPerBucket,
                        AxialCrystalsPerBlock,TransaxialCrystalsPerBlock,
                        AxialCrstalsPerSinglesUnit, TransaxialCrystalsPerSinglesUnit,
                        num_detector_layers,
                        EnergyResolution,
                        ReferenceEnergy,
                        ScannerGeometry,
                        TransaxialCrystalSpacing,
                        AxialCrystalSpacing,
                        AxialBlockSpacing,
                        TransaxialBlockSpacing,
                        crystal_map_file_name);
  
      if (scanner_ptr->check_consistency()==Succeeded::yes ||
	  !ask("Ask questions again?",true))
	return scanner_ptr;
  
      delete scanner_ptr;
    } // infinite loop
}



Scanner *
Scanner::get_scanner_from_name(const string& name)
{ 
  Scanner * scanner_ptr;

  const string matching_name =
    standardise_interfile_keyword(name);
  Type type= E931; 
  while (type != Unknown_scanner)
  {
    scanner_ptr = new Scanner(type);
    const list<string>& list_of_names = scanner_ptr->get_all_names();
    for (std::list<string>::const_iterator iter =list_of_names.begin();
	 iter!=list_of_names.end();
	   ++iter)
      {
	const string matching_scanner_name =
	  standardise_interfile_keyword(*iter);
	if (matching_scanner_name==matching_name)
	  return scanner_ptr;
      }
    
    // we didn't find it yet
    delete scanner_ptr;
    // tricky business to find next type
    int int_type = type;
    ++int_type;
    type = static_cast<Type>(int_type);
  }
  // it's not in the list
  return new Scanner(Unknown_scanner);
}


string Scanner:: list_all_names()
{
#ifdef BOOST_NO_STRINGSTREAM
  // dangerous for out-of-range, but 'old-style' ostrstream seems to need this
  char str[30000];
  ostrstream s(str, 30000);
#else
  std::ostringstream s;
#endif

  Type type= E931; 
  while (type != Unknown_scanner)
  {
    Scanner scanner(type);
    // tricky business to find next type
    type = static_cast<Type>(static_cast<int>(type)+1);
    if (scanner.get_type() == User_defined_scanner)
      continue;
    s << scanner.list_names() << '\n';
  }
  
  return s.str();
}

std::list<std::string> Scanner::get_names_of_predefined_scanners()
{
  std::list<std::string> ret;
  Type type= E931;
  while (type != Unknown_scanner)
  {
    Scanner scanner(type);
    // tricky business to find next type
    type = static_cast<Type>(static_cast<int>(type)+1);
    if (scanner.get_type() == User_defined_scanner)
      continue;
    ret.push_back(scanner.get_name());
  }
  return ret;
}

static list<string> 
string_list(const string& s)
{
  list<string> l;
  l.push_back(s);
  return l;
}

static list<string> 
string_list(const string& s1, const string& s2)
{
  list<string> l;
  l.push_back(s1);
  l.push_back(s2);
  return l;
}

static list<string> 
string_list(const string& s1, const string& s2, const string& s3)
{
  list<string> l;
  l.push_back(s1);
  l.push_back(s2);
  l.push_back(s3);
  return l;
}

static list<string> 
string_list(const string& s1, const string& s2, const string& s3, const string& s4)
{
  list<string> l;
  l.push_back(s1);
  l.push_back(s2);
  l.push_back(s3);
  l.push_back(s4);
  return l;
}

static list<string>
string_list(const string& s1, const string& s2, const string& s3, const string& s4, const string& s5)
{
  list<string> l;
  l.push_back(s1);
  l.push_back(s2);
  l.push_back(s3);
  l.push_back(s4);
  l.push_back(s5);
  return l;
}

END_NAMESPACE_STIR
