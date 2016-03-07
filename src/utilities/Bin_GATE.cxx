
/*!
  \file 
  \ingroup listmode
  \ingroup executables

  \brief Preliminary Implementation of converting GATE ROOT file to STIR ProjData

  Fairly ugly hack from LmToProjData combined with reading ROOT. Based on Sadek Nehmeh's code
  available on OpenGATE.

  \author Kris Thielemans
  \author Sanida Mustafovic
*/
/*
    Copyright (C) 2015, University College London
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

// uncomment next line to use Cylindrical PET
// #define MODULES

#include "stir/ExamInfo.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"

#include "stir/Scanner.h"
#include "stir/ProjDataInterfile.h"
#include "stir/SegmentByView.h"
#include "stir/TimeFrameDefinitions.h"
#include "stir/is_null_ptr.h"

#include <fstream>
#include <iostream>
#include <vector>

#ifndef STIR_NO_NAMESPACES
using std::string;
using std::fstream;
using std::ifstream;
using std::iostream;
using std::ofstream;
using std::ios;
using std::cerr;
using std::cout;
using std::flush;
using std::endl;
using std::min;
using std::max;
using std::vector;
using std::pair;
#endif

USING_NAMESPACE_STIR
typedef float elem_type;
#  define OUTPUTNumericType NumericType::FLOAT
typedef SegmentByView<elem_type> segment_type;
/******************** Prototypes  for local routines ************************/



static void 
allocate_segments(VectorWithOffset<segment_type *>& segments,
                       const int start_segment_index, 
	               const int end_segment_index,
                       const ProjDataInfo* proj_data_info_ptr);

// In the next 2 functions, the 'output' parameter needs to be passed 
// because save_and_delete_segments needs it when we're not using SegmentByView

/* last parameter only used if USE_SegmentByView
   first parameter only used when not USE_SegmentByView
 */         
static void 
save_and_delete_segments(shared_ptr<iostream>& output,
			      VectorWithOffset<segment_type *>& segments,
			      const int start_segment_index, 
			      const int end_segment_index, 
			      ProjData& proj_data);
static
shared_ptr<ProjData>
construct_proj_data(shared_ptr<iostream>& output,
                    const string& output_filename, 
		    const ExamInfo& exam_info,
                    const shared_ptr<ProjDataInfo>& proj_data_info_ptr);


#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "TROOT.h"
#include "TSystem.h"
#include "TChain.h"
#include "TH2D.h" 
#include "TDirectory.h"
#include "TList.h"
#include "Rtypes.h"
#include "TChainElement.h"
#include "TTree.h"
#include "TFile.h"
#include "TStyle.h"
#include "TH2.h"
#include "TH2F.h"
#include "TCanvas.h"
#include "TRandom.h"

using namespace std ;

#define  N_RINGS               64 // mMR: Number of rings
#define  N_SEG                127 // mMR: Number of segments (2 N_RINGS - 1)
#define  N_DET                504 // mMR: Detectors per ring
#define  S_WIDTH              344 // mMR: Number of radial sinogram bins
#define  N_RSEC                56 // mMR: Number of resector
#define  N_MODULE               8 // mMR: Number of modules (2x3)
#define  N_MOD_xy               1 // mMR: Number of tangential modules
#define  N_MOD_z                8 // mMR: Number of axial modules
#define  N_SUBMOD              1 // mMR: Number of submodules (1x1)
#define  N_SMOD_xy              1 // mMR: Number of tangential submodules
#define  N_SMOD_z               1 // mMR: Number of axial submodules
#define  N_CRYSTAL            72 // mMR: Number of crystals (6x6)
#define  N_CRY_xy               9 // mMR: Number of tangential crystals
#define  N_CRY_z               8 // mMR: Number of axial crystals
#define  MAX_D_RING            63 // mMR: Maximum ring difference
#define  N_PLANES              4096// mMR: Total number of sinograms
#define  FOV                  300 // mMR: Width of the FOV (mm)
#define  SLICES_PER_FOV       64 // mMR: Number of slices per FOV
#define  USE_OFFSET             0 // mMR: On/Off use of offset
#define  OFFSET                 378 // mMR: Sets initial sinogram angle

unsigned short ans,ans1;
unsigned short ans_SC, ans_RC, ans_SRC, ans_S, ans_R;


int main(int argc, char** argv)
{
  //---------------------------------------------------------------------------------
  // the first  argument (argv[1]) is the sub-directory of the input file
  // the second argument (argv[2]) is the name of the output file
  //
  //---------------------------------------------------------------------------------
  if(argc<2) {
    std::cout<<" Usage: root-filename STIR-output-filename "<<std::endl ;
    return 1;
  }
  
  const string inputfilename = argv[1];

  cout << "Input file name is " << inputfilename << endl;
  TChain *Coincidences = new TChain("Coincidences") ;
  Coincidences->Add(inputfilename.c_str()) ; 
  
  const string outputfilename = argv[2] ; 
  cout << "Projection file name is = " << outputfilename << endl ;

 
//#####################################################################
//#              Loop over the .root file in the directory "PATH"     #
//#####################################################################
  Int_t   Trues = 0, Scatters = 0, Randoms = 0;
  Int_t   nbytes = 0;
  
  //####################################################################
  //#             Declaration of leaves types - TTree Coincidences     #
  //####################################################################
  
  Float_t         axialPos, rotationAngle, sinogramS, sinogramTheta;
  Char_t          comptVolName1[40], comptVolName2[40];
  Int_t           compton1, compton2;
  Int_t           runID, sourceID1, sourceID2, eventID1, eventID2; 
  Int_t           crystalID1, crystalID2;
  Int_t           comptonPhantom1, comptonPhantom2;
  Float_t         energy1, energy2;   
  Float_t         globalPosX1, globalPosX2, globalPosY1, globalPosY2, globalPosZ1, globalPosZ2;
  Float_t         sourcePosX1, sourcePosX2, sourcePosY1, sourcePosY2, sourcePosZ1, sourcePosZ2;
  Double_t        time1, time2;
  
  //######################################################################################
  //#                        Set branch addresses - TTree Coincidences                   #
  //######################################################################################
  
  Coincidences->SetBranchStatus("*",0);
  Coincidences->SetBranchAddress("axialPos",&axialPos);
  Coincidences->SetBranchAddress("comptVolName1",&comptVolName1);
  Coincidences->SetBranchAddress("comptVolName2",&comptVolName2);
  Coincidences->SetBranchAddress("comptonCrystal1",&compton1);
  Coincidences->SetBranchAddress("comptonCrystal2",&compton2);
  Coincidences->SetBranchAddress("crystalID1",&crystalID1);
  Coincidences->SetBranchAddress("crystalID2",&crystalID2);
  Coincidences->SetBranchAddress("comptonPhantom1",&comptonPhantom1);
  Coincidences->SetBranchAddress("comptonPhantom2",&comptonPhantom2);
  Coincidences->SetBranchAddress("energy1",&energy1);
  Coincidences->SetBranchAddress("energy2",&energy2);   
  Coincidences->SetBranchAddress("eventID1",&eventID1);
  Coincidences->SetBranchAddress("eventID2",&eventID2);
  Coincidences->SetBranchAddress("globalPosX1",&globalPosX1);
  Coincidences->SetBranchAddress("globalPosX2",&globalPosX2);
  Coincidences->SetBranchAddress("globalPosY1",&globalPosY1);
  Coincidences->SetBranchAddress("globalPosY2",&globalPosY2);
  Coincidences->SetBranchAddress("globalPosZ1",&globalPosZ1);
  Coincidences->SetBranchAddress("globalPosZ2",&globalPosZ2);
#ifdef MODULES
  Int_t           layerID1, layerID2;
  Int_t           submoduleID1, submoduleID2, moduleID1, moduleID2, rsectorID1, rsectorID2;
  Coincidences->SetBranchAddress("layerID1",&layerID1);
  Coincidences->SetBranchAddress("layerID2",&layerID2);
  Coincidences->SetBranchAddress("moduleID1",&moduleID1);
  Coincidences->SetBranchAddress("moduleID2",&moduleID2);
  Coincidences->SetBranchAddress("rsectorID1",&rsectorID1);
  Coincidences->SetBranchAddress("rsectorID2",&rsectorID2);
  Coincidences->SetBranchAddress("submoduleID1",&submoduleID1);
  Coincidences->SetBranchAddress("submoduleID2",&submoduleID2);
#else
  Int_t blockID1, blockID2;
  Coincidences->SetBranchAddress("blockID1",&blockID1);
  Coincidences->SetBranchAddress("blockID2",&blockID2);
#endif
  Coincidences->SetBranchAddress("rotationAngle",&rotationAngle);
  Coincidences->SetBranchAddress("runID",&runID);
  Coincidences->SetBranchAddress("sinogramS",&sinogramS);
  Coincidences->SetBranchAddress("sinogramTheta",&sinogramTheta);
  Coincidences->SetBranchAddress("sourceID1",&sourceID1);
  Coincidences->SetBranchAddress("sourceID2",&sourceID2);
  Coincidences->SetBranchAddress("sourcePosX1",&sourcePosX1);
  Coincidences->SetBranchAddress("sourcePosX2",&sourcePosX2);
  Coincidences->SetBranchAddress("sourcePosY1",&sourcePosY1);
  Coincidences->SetBranchAddress("sourcePosY2",&sourcePosY2);
  Coincidences->SetBranchAddress("sourcePosZ1",&sourcePosZ1);
  Coincidences->SetBranchAddress("sourcePosZ2",&sourcePosZ2);
  Coincidences->SetBranchAddress("time1",&time1);
  Coincidences->SetBranchAddress("time2",&time2);
    


  shared_ptr<Scanner> scanner_sptr(new Scanner(Scanner::Siemens_mMR));
  shared_ptr<ProjDataInfo> 
    proj_data_info_sptr(ProjDataInfo::ProjDataInfoCTI (scanner_sptr, 1,
						       scanner_sptr->get_num_rings()-1, 
						       scanner_sptr->get_max_num_views(),
						       scanner_sptr->get_max_num_non_arccorrected_bins(),
						       /*arc_corrected=*/ false));
  const ProjDataInfoCylindricalNoArcCorr& proj_data_info =
    dynamic_cast<const ProjDataInfoCylindricalNoArcCorr&>(*proj_data_info_sptr);

  VectorWithOffset<segment_type *> 
    segments (proj_data_info_sptr->get_min_segment_num(), 
	      proj_data_info_sptr->get_max_segment_num());
  const int start_segment_index=proj_data_info_sptr->get_min_segment_num();
  const int end_segment_index=proj_data_info_sptr->get_max_segment_num();
  allocate_segments(segments, start_segment_index, end_segment_index, proj_data_info_sptr.get());


  // *********** open output file
  shared_ptr<iostream> output;
  ExamInfo exam_info;
  shared_ptr<ProjData> proj_data_ptr =
    construct_proj_data(output, outputfilename, exam_info, proj_data_info_sptr);
  
  Int_t nentries = (Int_t)(Coincidences->GetEntries());
  
  //#####################################################################
  //#             SINOGRAMS AND PROJECTION PLANES BINNING               #
  //#####################################################################
  Int_t    ring1, ring2, crystal1, crystal2;

  printf("Total Number of Coincidence Events:= %d \n",nentries ); 
  const Int_t max_num_events = nentries;

  for (Int_t i = 0 ; i < std::min(max_num_events,nentries) ; i++)
    {      

      if ((i%250000)  == 0 && i!=0)  printf("... %d ",i);       
      if ((i%1000000) == 0 && i!=0)  printf("\n");       

      nbytes += Coincidences->GetEntry(i);

      // Update the number of Trues and Randoms...
      //------------------------------------------
      
      if (eventID1 == eventID2)  
	{
	  if (comptonPhantom1 == 0 && comptonPhantom2 == 0) Trues++;
	  else Scatters++;
	}
      else Randoms++;   
    
      //-----------------------------------
      //  Identify the ring# and crystal#...
      //-----------------------------------

// For 8x8:
#ifdef MODULES
      ring1 = (Int_t)(crystalID1/8) 
	    + (Int_t)(submoduleID1/N_SMOD_xy)*N_CRY_z
	    + (Int_t)(moduleID1/N_MOD_xy)*N_SMOD_z*N_CRY_z;
      ring2 = (Int_t)(crystalID2/8) 
	    + (Int_t)(submoduleID2/N_SMOD_xy)*N_CRY_z
	    + (Int_t)(moduleID2/N_MOD_xy)*N_SMOD_z*N_CRY_z;

      if ( abs(ring1 - ring2) > MAX_D_RING )  continue;  

      crystal1 = rsectorID1 * N_MOD_xy * N_SMOD_xy * N_CRY_xy
	       + (moduleID1%N_MOD_xy) * N_SMOD_xy * N_CRY_xy
	       + (submoduleID1%N_SMOD_xy) * N_CRY_xy
	       + (crystalID1%8);
      crystal2 = rsectorID2 * N_MOD_xy * N_SMOD_xy * N_CRY_xy
	       + (moduleID2%N_MOD_xy) * N_SMOD_xy * N_CRY_xy
	       + (submoduleID2%N_SMOD_xy) * N_CRY_xy
	       + (crystalID2%8);
#else
      ring1 = (Int_t)(crystalID1/N_CRY_z) 
	+ (Int_t)(blockID1/56)*N_CRY_z;
      ring2 = (Int_t)(crystalID2/N_CRY_z) 
	+ (Int_t)(blockID2/56)*N_CRY_z;

      crystal1 = (blockID1%56) * 9
	       + (crystalID1%8);
      crystal2 = (blockID2%56) * 9
	       + (crystalID2%8);
#endif

      //-----------------------------------------------------
      //  Rotate the image correctly#...
      //--------------------------------
      if (USE_OFFSET == 1)
	{
	  crystal1 = crystal1 + OFFSET;
	  crystal2 = crystal2 + OFFSET;    
	  if (crystal1 >= N_DET)  crystal1 = crystal1 - N_DET;
	  if (crystal2 >= N_DET)  crystal2 = crystal2 - N_DET;
	}


    Bin bin;
    if (proj_data_info.
        get_bin_for_det_pair(bin,
			     crystal1, ring1, 
			     crystal2, ring2) != Succeeded::yes)
      {
	error("something");
      }


    // check if it's inside the range we want to store
    if (
	bin.tangential_pos_num()>= proj_data_ptr->get_min_tangential_pos_num()
	&& bin.tangential_pos_num()<= proj_data_ptr->get_max_tangential_pos_num()
	&& bin.axial_pos_num()>=proj_data_ptr->get_min_axial_pos_num(bin.segment_num())
	&& bin.axial_pos_num()<=proj_data_ptr->get_max_axial_pos_num(bin.segment_num())
	) 
      {
	(*segments[bin.segment_num()])[bin.view_num()][bin.axial_pos_num()][bin.tangential_pos_num()] += 1;
      }
                 
    }
  printf("\n");

  save_and_delete_segments(output, segments, 
			   start_segment_index, end_segment_index, 
			   *proj_data_ptr);  
  
  return(EXIT_SUCCESS);
}

/************************* Local helper routines *************************/
#define USE_SegmentByView


void 
allocate_segments( VectorWithOffset<segment_type *>& segments,
		  const int start_segment_index, 
		  const int end_segment_index,
		  const ProjDataInfo* proj_data_info_ptr)
{
  
  for (int seg=start_segment_index ; seg<=end_segment_index; seg++)
  {
#ifdef USE_SegmentByView
    segments[seg] = new SegmentByView<elem_type>(
    	proj_data_info_ptr->get_empty_segment_by_view (seg)); 
#else
    segments[seg] = 
      new Array<3,elem_type>(IndexRange3D(0, proj_data_info_ptr->get_num_views()-1, 
				      0, proj_data_info_ptr->get_num_axial_poss(seg)-1,
				      -(proj_data_info_ptr->get_num_tangential_poss()/2), 
				      proj_data_info_ptr->get_num_tangential_poss()-(proj_data_info_ptr->get_num_tangential_poss()/2)-1));
#endif
  }
}

void 
save_and_delete_segments(shared_ptr<iostream>& output,
			 VectorWithOffset<segment_type *>& segments,
			 const int start_segment_index, 
			 const int end_segment_index, 
			 ProjData& proj_data)
{
  
  for (int seg=start_segment_index; seg<=end_segment_index; seg++)
  {
    {
#ifdef USE_SegmentByView
      proj_data.set_segment(*segments[seg]);
#else
      (*segments[seg]).write_data(*output);
#endif
      delete segments[seg];      
    }
    
  }
}



static
shared_ptr<ProjData>
construct_proj_data(shared_ptr<iostream>& output,
                    const string& output_filename, 
		    const ExamInfo& exam_info,
                    const shared_ptr<ProjDataInfo>& proj_data_info_ptr)
{
  shared_ptr<ExamInfo> exam_info_sptr(new ExamInfo(exam_info));

#ifdef USE_SegmentByView
  // don't need output stream in this case
  shared_ptr<ProjData> proj_data_sptr(new ProjDataInterfile(exam_info_sptr,
							    proj_data_info_ptr, output_filename, ios::out, 
							    ProjDataFromStream::Segment_View_AxialPos_TangPos,
							    OUTPUTNumericType));
  return proj_data_sptr;
#else
  // this code would work for USE_SegmentByView as well, but the above is far simpler...
  vector<int> segment_sequence_in_stream(proj_data_info_ptr->get_num_segments());
  { 
    std::vector<int>::iterator current_segment_iter =
      segment_sequence_in_stream.begin();
    for (int segment_num=proj_data_info_ptr->get_min_segment_num();
         segment_num<=proj_data_info_ptr->get_max_segment_num();
         ++segment_num)
      *current_segment_iter++ = segment_num;
  }
  output = new fstream (output_filename.c_str(), ios::out|ios::binary);
  if (!*output)
    error("Error opening output file %s\n",output_filename.c_str());
  shared_ptr<ProjDataFromStream> proj_data_ptr(
					       new ProjDataFromStream(exam_info_sptr, proj_data_info_ptr, output, 
								      /*offset=*/std::streamoff(0), 
								      segment_sequence_in_stream,
								      ProjDataFromStream::Segment_View_AxialPos_TangPos,
								      OUTPUTNumericType));
  write_basic_interfile_PDFS_header(output_filename, *proj_data_ptr);
  return proj_data_ptr;  
#endif
}

