/*
   Copyright
   */
/*!

  \file
  \ingroup utilities

  \brief Find normalisation factors given projection data of a cylinder (component-based method)

  \author Parisa Khateri

*/
//#include "stir/Scanner.h"
//#include "stir/stream.h"
//#include "stir/CPUTimer.h"
#include "stir/utilities.h"
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include "stir/shared_ptr.h"
#include "stir/ProjData.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/LORCoordinates.h"
#include "stir/Bin.h"
#include "stir/ProjDataInterfile.h"
#include "stir/IO/interfile.h"
#include <limits>
#include <tuple>

USING_NAMESPACE_STIR


std::tuple<int, int> get_crystalTangPos_from_LOR(int view_nr, int tang_pos) {
	return {(view_nr + 180 + ((tang_pos - (abs(tang_pos) % 2)) / 2))%180, (view_nr + 270 - ((tang_pos + (abs(tang_pos) % 2)) / 2))%180};
}

std::tuple<int, int> get_crystalAxPos_from_LOR(int segment_nr, int ax_pos) {
	if (segment_nr < 0)
	{
    	return  {ax_pos - segment_nr, ax_pos};
	}
	else
	{
    	return  {ax_pos, ax_pos + segment_nr};
	}
}


int main(int argc, char **argv)
{
	if (argc!=4)
	{//TODO ask the user to give the activity
		std::cerr << "Usage: "<< argv[0]
			<<" output_file_name_prefix cylider_measured_data cylinder_radius(mm)\n"
			<<"only cylinder data are supported. The radius should be the radius of the measured cylinder data.\n"
			<<"warning: mind the input order\n";
		return EXIT_FAILURE;
	}

	shared_ptr<ProjData> cylinder_projdata_ptr = ProjData::read_from_file(argv[2]);
	const std::string output_file_name = argv[1];
	const float R = atof(argv[3]); // cylinder radius
	if (R==0)
	{
		std::cerr << " Radius must be a float value\n"
			<<"Usage: "<< argv[0]
			<<" output_file_name_prefix cylider_measured_data cylinder_radius\n"
			<<"warning: mind the input order\n";
		return EXIT_FAILURE;
	}

	//output file
	shared_ptr<ProjDataInfo> cylinder_pdi_ptr(cylinder_projdata_ptr->get_proj_data_info_sptr()->clone());

	ProjDataInterfile output_projdata(cylinder_projdata_ptr->get_exam_info_sptr(), cylinder_pdi_ptr, output_file_name);
	write_basic_interfile_PDFS_header(output_file_name, output_projdata);

	CartesianCoordinate3D<float> c1, c2;
	LORInAxialAndNoArcCorrSinogramCoordinates<float> lor;

	// first find the average number of counts per LOR
	std::cout << "Finding the Average NCounts per LOR \n";
	float total_count = 0;
	float min_count = std::numeric_limits<double>::max(); // minimum number of counts per LOR
	float average_count = 0; //average number of counts per LOR in the active region
	int num_active_LORs = 0; //number of LORs which pass through the cylinder

	const int nTang_pos_max = 180;	
	const int nAx_pos_max = 64;	

	for (int seg = cylinder_projdata_ptr->get_min_segment_num(); seg <=cylinder_projdata_ptr->get_max_segment_num(); ++seg)
	{
		for (int view =cylinder_projdata_ptr->get_min_view_num(); view <=cylinder_projdata_ptr->get_max_view_num(); ++view)
		{
			Viewgram<float> cylinder_viewgram = cylinder_projdata_ptr->get_viewgram(view, seg);
			
			for (int ax = cylinder_projdata_ptr->get_min_axial_pos_num(seg); ax <=cylinder_projdata_ptr->get_max_axial_pos_num(seg); ++ax)
			{
				for (int tang = cylinder_projdata_ptr->get_min_tangential_pos_num(); tang <=cylinder_projdata_ptr->get_max_tangential_pos_num(); ++tang)
				{
					Bin bin(seg, view, ax, tang);
					cylinder_projdata_ptr->get_proj_data_info_sptr()->get_LOR(lor, bin);
					LORAs2Points<float> lor_as2points(lor);
					LORAs2Points<float> intersection_coords;
					if (find_LOR_intersections_with_cylinder(intersection_coords, lor_as2points, R) ==Succeeded::yes)
					{ //this only succeeds if LOR is intersecting with the cylinder
						float N_lor = cylinder_viewgram[ax][tang]; //counts seen by this lor
						c1 = intersection_coords.p1();
						c2 = intersection_coords.p2();
						float c12 = sqrt( pow(c1.z()-c2.z(), 2) + pow(c1.y()-c2.y(), 2) + pow(c1.x()-c2.x(), 2) );  // length of intersection of lor with the cylinder
						if (c12>0.5) // if LOR intersection is lager than 0.5 mm, check the count per LOR
						{
							float N_lor_corrected=N_lor/c12; // corrected for the length
							total_count+=N_lor_corrected;
							num_active_LORs+=1;
							if (N_lor_corrected<min_count && N_lor_corrected!=0) min_count=N_lor_corrected;
						}
					}
				}
			}
		}
	}


	average_count = total_count / num_active_LORs;
	std::cout << "num_lor = "						<< num_active_LORs	<< "\n";
	std::cout << "tot_count_per_length_unit = "		<< total_count		<< "\n";
	std::cout << "average_count_per_length_unit = "	<< average_count	<< "\n";
	std::cout << "non_zero_min_per_length_unit = "	<< min_count		<< "\n";

	// Now find the Detector Efficiencies
	std::array<std::array<float, nTang_pos_max>, nAx_pos_max> det_eff {{}};
	std::array<std::array<float, nTang_pos_max>, nAx_pos_max> n_alive_channels {{}};

	std::cout << "Finding the Detector Efficiencies \n";
	for (int view =cylinder_projdata_ptr->get_min_view_num(); view <=cylinder_projdata_ptr->get_max_view_num(); ++view)
	{
		int seg = 0;
		Viewgram<float> cylinder_viewgram = cylinder_projdata_ptr->get_viewgram(view, seg);
		for (int ax = cylinder_projdata_ptr->get_min_axial_pos_num(seg); ax <=cylinder_projdata_ptr->get_max_axial_pos_num(seg); ++ax)
		{
			for (int tang = cylinder_projdata_ptr->get_min_tangential_pos_num(); tang <=cylinder_projdata_ptr->get_max_tangential_pos_num(); ++tang)
			{
				Bin bin(seg, view, ax, tang);
				cylinder_projdata_ptr->get_proj_data_info_sptr()->get_LOR(lor, bin);
				float N_lor = cylinder_viewgram[ax][tang]; //counts seen by this lor
				if (N_lor > 1)
				{
					std::tuple<int, int> crystal_axial = get_crystalAxPos_from_LOR(seg, ax);
					std::tuple<int, int> crystal_tang = get_crystalTangPos_from_LOR(view, tang);
					det_eff[std::get<0>(crystal_axial)][std::get<0>(crystal_tang)] += N_lor;
					det_eff[std::get<1>(crystal_axial)][std::get<1>(crystal_tang)] += N_lor;
					n_alive_channels[std::get<0>(crystal_axial)][std::get<0>(crystal_tang)]++;
					n_alive_channels[std::get<1>(crystal_axial)][std::get<1>(crystal_tang)]++;
				}
			}
		}
	}
	
	const int nSectors = 12;
	float total_alive_channels = 0;
	for (int i = 0; i < (nTang_pos_max/nSectors); i++) {
		int dead_channels = 0;
		float sum = 0;
		for (int j = 0; j < nAx_pos_max; j++) {
			for (int k = i; k < nTang_pos_max; k+=(nTang_pos_max/nSectors)) {
				sum += det_eff[j][k];
				total_alive_channels += n_alive_channels[j][k];
			}
		}
		sum = sum / float(nSectors * nAx_pos_max);
		total_alive_channels = total_alive_channels / float(nSectors * nAx_pos_max);
		for (int j = 0; j < nAx_pos_max; j++) {
			for (int k = i; k < nTang_pos_max; k+=(nTang_pos_max/nSectors)) {
				float new_val = (det_eff[j][k] * total_alive_channels) / (sum * n_alive_channels[j][k]);
				if (det_eff[j][k] < 10.0) {
					std::cout << det_eff[j][k] << std::endl;
					new_val = 0.00;
				}
				det_eff[j][k] = new_val;
			}
		}
	}
		
	float mean_deteff = 0.0;
	for (int j = 0; j < nAx_pos_max; j++) {
		for (int k = 0; k < nTang_pos_max; k++) {
			mean_deteff+=det_eff[j][j];
		}
	}
	std::cout << "Mean Detector Efficiency" << mean_deteff/float(nAx_pos_max*nTang_pos_max) << std::endl;
	
	std::cout << "Apply detector efficiencies\n";
	
	total_count = 0;
	min_count = std::numeric_limits<double>::max(); // minimum number of counts per LOR
	average_count = 0; //average number of counts per LOR in the active region
	num_active_LORs = 0; //number of LORs which pass through the cylinder

	for (int seg = cylinder_projdata_ptr->get_min_segment_num(); seg <=cylinder_projdata_ptr->get_max_segment_num(); ++seg)
	{
		for (int view =cylinder_projdata_ptr->get_min_view_num(); view <=cylinder_projdata_ptr->get_max_view_num(); ++view)
		{
			Viewgram<float> cylinder_viewgram = cylinder_projdata_ptr->get_viewgram(view, seg);
			
			for (int ax = cylinder_projdata_ptr->get_min_axial_pos_num(seg); ax <=cylinder_projdata_ptr->get_max_axial_pos_num(seg); ++ax)
			{
				for (int tang = cylinder_projdata_ptr->get_min_tangential_pos_num(); tang <=cylinder_projdata_ptr->get_max_tangential_pos_num(); ++tang)
				{
					Bin bin(seg, view, ax, tang);
					cylinder_projdata_ptr->get_proj_data_info_sptr()->get_LOR(lor, bin);
					LORAs2Points<float> lor_as2points(lor);
					LORAs2Points<float> intersection_coords;
					if (find_LOR_intersections_with_cylinder(intersection_coords, lor_as2points, R) ==Succeeded::yes)
					{ //this only succeeds if LOR is intersecting with the cylinder
						float N_lor = cylinder_viewgram[ax][tang]; //counts seen by this lor
						c1 = intersection_coords.p1();
						c2 = intersection_coords.p2();
						float c12 = sqrt( pow(c1.z()-c2.z(), 2) + pow(c1.y()-c2.y(), 2) + pow(c1.x()-c2.x(), 2) );  // length of intersection of lor with the cylinder
						if (c12>0.5) // if LOR intersection is lager than 0.5 mm, check the count per LOR
						{
							std::tuple<int, int> crystal_axial = get_crystalAxPos_from_LOR(seg, ax);
							std::tuple<int, int> crystal_tang = get_crystalTangPos_from_LOR(view, tang);
							float N_lor_corrected=0.0; // corrected for the length and detector efficiencies
							if (det_eff[std::get<0>(crystal_axial)][std::get<0>(crystal_tang)] * det_eff[std::get<1>(crystal_axial)][std::get<1>(crystal_tang)] != 0)
							{
								N_lor_corrected=N_lor/(c12 * det_eff[std::get<0>(crystal_axial)][std::get<0>(crystal_tang)] * det_eff[std::get<1>(crystal_axial)][std::get<1>(crystal_tang)]); // corrected for the length and detector efficiencies
								num_active_LORs+=1;
							}
							total_count+=N_lor_corrected;
							if (N_lor_corrected<min_count && N_lor_corrected!=0)
							{
								min_count=N_lor_corrected;
							}
						}
					}
				}
			}
		}
	}
	
	average_count = total_count / num_active_LORs;
	std::cout << "num_lor = "						<< num_active_LORs	<< "\n";
	std::cout << "tot_count_per_length_unit = "		<< total_count		<< "\n";
	std::cout << "average_count_per_length_unit = "	<< average_count	<< "\n";
	std::cout << "non_zero_min_per_length_unit = "	<< min_count		<< "\n";

	
	
	std::cout << "Get geometric coefficients\n";
	const int nSeg_proj = 64;	
	const int nView_proj = 90;	
	const int nTang_pos_proj = 100;	
	
	if ((nTang_pos_proj-1)!=(cylinder_projdata_ptr->get_max_tangential_pos_num() - cylinder_projdata_ptr->get_min_tangential_pos_num()))
	{//TODO ask the user to give the activity
		std::cerr << "Error: Geometry Hardcoded for SAFIR-II. Check number of Tangential positions in code to projdata \n";
		return EXIT_FAILURE;
	}
	if ((nSeg_proj-1)!=cylinder_projdata_ptr->get_max_segment_num())
	{//TODO ask the user to give the activity
		std::cerr << "Error: Geometry Hardcoded for SAFIR-II. Check number of segments in code to projdata \n";
		return EXIT_FAILURE;
	}

	std::array<std::array<std::array<float, nTang_pos_proj>, nView_proj>, nSeg_proj> geom_coeff {{{}}};

	for (int seg = 0; seg < nSeg_proj; ++seg)
	{
		for (int tang = -(nTang_pos_proj/2); tang < (nTang_pos_proj/2); ++tang)
		{
			for (int base_view = 0; base_view < (nView_proj / (nSectors / 2)); ++base_view)
			{
				num_active_LORs = 0;
				float sum = 0;
				for (int true_view = base_view; true_view < nView_proj; true_view+=(nView_proj / (nSectors / 2)))
				{
					int valid_segments[2] = {seg, -seg};
					for (int true_seg : valid_segments)
					{
						Viewgram<float> cylinder_viewgram = cylinder_projdata_ptr->get_viewgram(true_view, true_seg);
						for (int ax = cylinder_projdata_ptr->get_min_axial_pos_num(true_seg); ax <= cylinder_projdata_ptr->get_max_axial_pos_num(true_seg); ++ax)
						{
							Bin bin(true_seg, true_view, ax, tang);
							cylinder_projdata_ptr->get_proj_data_info_sptr()->get_LOR(lor, bin);
							LORAs2Points<float> lor_as2points(lor);
							LORAs2Points<float> intersection_coords;

							if (find_LOR_intersections_with_cylinder(intersection_coords, lor_as2points, R) ==Succeeded::yes)
							{ //this only succeeds if LOR is intersecting with the cylinder

								float N_lor = cylinder_viewgram[ax][tang]; //counts seen by this lor
								c1 = intersection_coords.p1();
								c2 = intersection_coords.p2();
								float c12 = sqrt( pow(c1.z()-c2.z(), 2) + pow(c1.y()-c2.y(), 2) + pow(c1.x()-c2.x(), 2) );  // length of intersection of lor with the cylinder
								
								if (c12>0.5) // if LOR intersection is lager than 0.5 mm, check the count per LOR
								{
									if (N_lor>1) //if value is large enough
									{
										std::tuple<int, int> crystal_axial = get_crystalAxPos_from_LOR(true_seg, ax);
										std::tuple<int, int> crystal_tang = get_crystalTangPos_from_LOR(true_view, tang);
										float N_lor_corrected=0.0; // corrected for the length and detector efficiencies
										if (det_eff[std::get<0>(crystal_axial)][std::get<0>(crystal_tang)] * det_eff[std::get<1>(crystal_axial)][std::get<1>(crystal_tang)] != 0)
										{
											float N_lor_corrected=N_lor/(c12 * det_eff[std::get<0>(crystal_axial)][std::get<0>(crystal_tang)] * det_eff[std::get<1>(crystal_axial)][std::get<1>(crystal_tang)]); // corrected for the length and detector efficiencies
											num_active_LORs++;
											sum += N_lor_corrected;
										}
									}
								}
							}
						}
					}
				}

				for (int true_view = base_view; true_view < nView_proj; true_view+=(nView_proj / (nSectors / 2)))
				{
					if (sum == 0)
					{
						geom_coeff[seg][true_view][tang+(nTang_pos_proj/2)] = 0.0000;
					}
					else
					{
						geom_coeff[seg][true_view][tang+(nTang_pos_proj/2)] = average_count / (sum / num_active_LORs);
					}
				}
			}
		}
	}

	std::cout << "Find Norm Factors per LOR\n";

	for (int seg = cylinder_projdata_ptr->get_min_segment_num(); seg <= cylinder_projdata_ptr->get_max_segment_num(); ++seg)
	{
		for (int view =cylinder_projdata_ptr->get_min_view_num(); view <= cylinder_projdata_ptr->get_max_view_num(); ++view)
		{
			Viewgram<float> out_viewgram = cylinder_projdata_ptr->get_empty_viewgram(view, seg);
			for (int ax = cylinder_projdata_ptr->get_min_axial_pos_num(seg); ax <= cylinder_projdata_ptr->get_max_axial_pos_num(seg); ++ax)
			{
				for (int tang = cylinder_projdata_ptr->get_min_tangential_pos_num(); tang <= cylinder_projdata_ptr->get_max_tangential_pos_num(); ++tang)
				{
					Bin bin(seg, view, ax, tang);
					cylinder_projdata_ptr->get_proj_data_info_sptr()->get_LOR(lor, bin);

					std::tuple<int, int> crystal_axial = get_crystalAxPos_from_LOR(seg, ax);
					std::tuple<int, int> crystal_tang = get_crystalTangPos_from_LOR(view, tang);
					if (geom_coeff[abs(seg)][view][tang+(nTang_pos_proj/2)] == 0 || det_eff[std::get<0>(crystal_axial)][std::get<0>(crystal_tang)] == 0 || det_eff[std::get<1>(crystal_axial)][std::get<1>(crystal_tang)] == 0)
					{
						out_viewgram[ax][tang] = 0.0;
					}
					else
					{
						out_viewgram[ax][tang] = geom_coeff[abs(seg)][view][tang+(nTang_pos_proj/2)] / (det_eff[std::get<0>(crystal_axial)][std::get<0>(crystal_tang)] * det_eff[std::get<1>(crystal_axial)][std::get<1>(crystal_tang)]);
					}
				}
			}
			output_projdata.set_viewgram(out_viewgram);
		}
	}
}
