//
//
/*
    Copyright (C) 2005- 2011, Hammersmith Imanet Ltd
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
  \ingroup test

  \brief Test program for stir::ArcCorrection

  \author Kris Thielemans

*/

#include "stir/ProjDataInfoCylindricalArcCorr.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/RunTests.h"
#include "stir/Viewgram.h"
#include "stir/Sinogram.h"
#include "stir/Bin.h"
#include "stir/ArcCorrection.h"
#include "stir/round.h"
#include "stir/index_at_maximum.h"

START_NAMESPACE_STIR

/*!
  \ingroup test
  \brief Test class for ArcCorrection

  The following 2 tests are performed:

  - a geometry test to see that a point in a sinogram is arc-corrected to the
    correct location.

  - a uniformity test that checks that uniform data are arc-corrected to 
    uniform data with the same value.

  This is currently only done on sinograms and viewgrams.

  The tests are performed both at the default arc-corrected bin-size, 
  but also at twice as large bin-size.
*/
class ArcCorrectionTests: public RunTests
{
public:
  void run_tests();
protected:
  void run_tests_for_specific_proj_data_info(const ArcCorrection&);
};

void
ArcCorrectionTests::
run_tests_for_specific_proj_data_info(const ArcCorrection& arc_correction)
{
  const ProjDataInfoCylindricalArcCorr& proj_data_info_arc_corr =
    arc_correction.get_arc_corrected_proj_data_info();
  const ProjDataInfoCylindricalNoArcCorr& proj_data_info_noarc_corr =
    arc_correction.get_not_arc_corrected_proj_data_info();

  const float sampling_in_s = proj_data_info_arc_corr.get_tangential_sampling();

  for (int segment_num=proj_data_info_noarc_corr.get_min_segment_num();
       segment_num<=proj_data_info_noarc_corr.get_max_segment_num();
	 ++segment_num)
      {
	const int axial_pos_num = 0;
	Sinogram<float> noarccorr_sinogram = 
	      proj_data_info_noarc_corr.get_empty_sinogram(axial_pos_num, segment_num);
	Sinogram<float> arccorr_sinogram = 
	      proj_data_info_arc_corr.get_empty_sinogram(axial_pos_num, segment_num);

	for (int view_num=proj_data_info_noarc_corr.get_min_view_num();
	     view_num<=proj_data_info_noarc_corr.get_max_view_num();
	     view_num+=3)
	  {
	    Viewgram<float> noarccorr_viewgram = 
	      proj_data_info_noarc_corr.get_empty_viewgram(view_num, segment_num);
	    Viewgram<float> arccorr_viewgram = 
	      proj_data_info_arc_corr.get_empty_viewgram(view_num, segment_num);
	    // test geometry by checking if single non-zero value gets put in the right bin
	    {
	      for (int tangential_pos_num=proj_data_info_noarc_corr.get_min_tangential_pos_num();
		   tangential_pos_num<=proj_data_info_noarc_corr.get_max_tangential_pos_num();
		   tangential_pos_num+=4)
		{
		  noarccorr_sinogram.fill(0);
		  noarccorr_viewgram.fill(0);
		  noarccorr_viewgram[axial_pos_num][tangential_pos_num]=1;
		  noarccorr_sinogram[view_num][tangential_pos_num]=1;
		  arc_correction.do_arc_correction(arccorr_sinogram, noarccorr_sinogram);
		  arc_correction.do_arc_correction(arccorr_viewgram, noarccorr_viewgram);
		  check_if_equal(noarccorr_sinogram[view_num], noarccorr_viewgram[axial_pos_num],
				 "1 line in sinogram and viewgram (geometric test)");
		  const int arccorr_tangential_pos_num_at_max=
		    index_at_maximum(arccorr_viewgram[axial_pos_num]);
		    
		  const float noarccorr_s = 
		    proj_data_info_noarc_corr.
		    get_s(Bin(segment_num,view_num,axial_pos_num,tangential_pos_num));
		  const float arccorr_s = 
		    proj_data_info_arc_corr.
		    get_s(Bin(segment_num,view_num,axial_pos_num,arccorr_tangential_pos_num_at_max));
		  check((arccorr_s - noarccorr_s)/sampling_in_s < 1.1,
			"correspondence in location of maximum after arc-correction");
		}
	    }
	    // test if uniformity and counts are preserved
	    {
	      /* We set a viewgram to 1, and check if the transformed viewgram is also 1 
		 (except at the boundary).
	      */
	      noarccorr_sinogram.fill(1);
	      noarccorr_viewgram.fill(1);
	      arc_correction.do_arc_correction(arccorr_sinogram, noarccorr_sinogram);
	      arc_correction.do_arc_correction(arccorr_viewgram, noarccorr_viewgram);
	      check_if_equal(noarccorr_sinogram[view_num], noarccorr_viewgram[axial_pos_num],
			     "1 line in sinogram and viewgram (uniformity test)");

	      const float max_s = 
		    proj_data_info_noarc_corr.
		    get_s(Bin(segment_num,view_num,axial_pos_num,
			      proj_data_info_noarc_corr.get_max_tangential_pos_num()));
	      const float min_s = 
		    proj_data_info_noarc_corr.
		    get_s(Bin(segment_num,view_num,axial_pos_num,
			      proj_data_info_noarc_corr.get_min_tangential_pos_num()));		
	      for (int tangential_pos_num= round(min_s/sampling_in_s)+2;
		   tangential_pos_num<=round(max_s/sampling_in_s)-2;
		   ++tangential_pos_num)
		check_if_equal(arccorr_viewgram[axial_pos_num][tangential_pos_num], 1.F,
			       "uniformity");
	    }
	  }
      }
}


void
ArcCorrectionTests::run_tests()
{ 
  cerr << "-------- Testing ArcCorrection --------\n";
  ArcCorrection arc_correction;
  shared_ptr<Scanner> scanner_ptr(new Scanner(Scanner::E962));
  
  shared_ptr<ProjDataInfo> proj_data_info_ptr(
    ProjDataInfo::ProjDataInfoCTI(scanner_ptr,
				  /*span*/7, 10,/*views*/ 96, /*tang_pos*/128, /*arc_corrected*/ false));
  cerr << "Using default range and bin-size\n";
  {
    arc_correction.set_up(proj_data_info_ptr);
    run_tests_for_specific_proj_data_info(arc_correction);
  }
  cerr << "Using non-default range and bin-size\n";
  {
    arc_correction.set_up(proj_data_info_ptr,
			  128,
			  scanner_ptr->get_default_bin_size()*2);
    run_tests_for_specific_proj_data_info(arc_correction);
  }
}

END_NAMESPACE_STIR


USING_NAMESPACE_STIR

int main()
{
  ArcCorrectionTests tests;
  tests.run_tests();
  return tests.main_return_value();
}
