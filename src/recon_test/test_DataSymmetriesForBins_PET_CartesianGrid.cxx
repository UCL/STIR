//
//
/*
    Copyright (C) 2003- 2011, Hammersmith Imanet
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!

  \file
  \ingroup test

  \brief Test program for stir::DataSymmetriesForBins_PET_CartesianGrid

  Uses stir::ProjMatrixByBinUsingRayTracing.

   \author Kris Thielemans

*/

#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/IndexRange3D.h"
#include "stir/ProjData.h"
#include "stir/ProjDataInfo.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/Scanner.h"
#include "stir/IndexRange.h"
#include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h"
#include "stir/recon_buildblock/ProjMatrixElemsForOneBin.h"
#include "stir/recon_buildblock/DataSymmetriesForBins.h"
#include "stir/RunTests.h"
#include "stir/stream.h"
#include <iostream>
#include <sstream>
#include <math.h>
using std::stringstream;
using std::cerr;

START_NAMESPACE_STIR

template <typename T>
bool
coordinates_less(const BasicCoordinate<3, T>& el1, const BasicCoordinate<3, T>& el2)
{
  return el1[1] < el2[1] || (el1[1] == el2[1] && (el1[2] < el2[2] || (el1[2] == el2[2] && el1[3] < el2[3])));
}

/*!
  \ingroup test
  \brief Test class for DataSymmetriesForBins_PET_CartesianGrid


  Uses ProjMatrixByBinUsingRayTracing to compute matrix elements with using various
  symmetries. Checks if results are independent of which symmetries we use.

*/
class DataSymmetriesForBins_PET_CartesianGridTests : public RunTests
{
public:
  DataSymmetriesForBins_PET_CartesianGridTests(char const* template_proj_data_filename = 0);

  void run_tests() override;

private:
  char const* template_proj_data_filename;

  shared_ptr<ProjDataInfo> proj_data_info_sptr;

  void run_tests_2_proj_matrices_1_bin(const ProjMatrixByBin& proj_matrix_no_symm,
                                       const ProjMatrixByBin& proj_matrix_with_symm,
                                       const Bin& bin);
  void run_tests_2_proj_matrices(const ProjMatrixByBin& proj_matrix_no_symm, const ProjMatrixByBin& proj_matrix_with_symm);
  void run_tests_all_symmetries(const shared_ptr<ProjDataInfo>& proj_data_info_sptr,
                                const shared_ptr<DiscretisedDensity<3, float>>& density_sptr);
  void run_tests_for_1_projdata(const shared_ptr<ProjDataInfo>& proj_data_info_sptr);
};

DataSymmetriesForBins_PET_CartesianGridTests::DataSymmetriesForBins_PET_CartesianGridTests(
    char const* template_proj_data_filename)
    : template_proj_data_filename(template_proj_data_filename)
{}

void
DataSymmetriesForBins_PET_CartesianGridTests::run_tests_2_proj_matrices_1_bin(const ProjMatrixByBin& proj_matrix_no_symm,
                                                                              const ProjMatrixByBin& proj_matrix_with_symm,
                                                                              const Bin& bin)
{
#if 0 // SYM
  //  assert(proj_matrix_with_symm.get_symmetries_ptr()->is_basic(bin));
  // vector<Bin> related_bins;
  //proj_matrix_with_symm.get_symmetries_ptr()->get_related_bins(related_bins, bin);
  Bin basic_bin=bin;
  proj_matrix_with_symm.get_symmetries_ptr()->find_basic_bin(basic_bin);
  vector<Bin> related_bins;
  proj_matrix_with_symm.get_symmetries_ptr()->get_related_bins(related_bins, basic_bin);

  ProjMatrixElemsForOneBin elems_no_sym;
  ProjMatrixElemsForOneBin elems_with_sym;
  for (vector<Bin>::const_iterator bin_iter = related_bins.begin();
       bin_iter != related_bins.end();
       ++bin_iter)
    {
      proj_matrix_with_symm.
	get_proj_matrix_elems_for_one_bin(elems_with_sym, *bin_iter);
      proj_matrix_no_symm.
	get_proj_matrix_elems_for_one_bin(elems_no_sym, *bin_iter);
#else

  ProjMatrixElemsForOneBin elems_no_sym;
  ProjMatrixElemsForOneBin elems_with_sym;
  {
    proj_matrix_with_symm.get_proj_matrix_elems_for_one_bin(elems_with_sym, bin);
    proj_matrix_no_symm.get_proj_matrix_elems_for_one_bin(elems_no_sym, bin);
#endif
  elems_no_sym.sort();
  elems_with_sym.sort();

  if (!check(elems_no_sym == elems_with_sym, "Comparing symmetry LORs")
      || !check(elems_with_sym.get_bin() == elems_no_sym.get_bin(), "Comparing symmetry bin configuration"))
    {
      // SYM const Bin bin=*bin_iter;

      cerr << "\nCurrent bin: \tsegment = " << bin.segment_num() << ", \taxial pos " << bin.axial_pos_num()
           << ", \tview = " << bin.view_num() << ", \ttangential_pos_num = " << bin.tangential_pos_num()
           << ", timing position index = " << bin.timing_pos_num()
           << "\nSymm bin: \t\tsegment = " << elems_with_sym.get_bin().segment_num() << ", \taxial pos "
           << elems_with_sym.get_bin().axial_pos_num() << ", \tview = " << elems_with_sym.get_bin().view_num()
           << ", \ttangential_pos_num = " << elems_with_sym.get_bin().tangential_pos_num()
           << ", timing position index = " << bin.timing_pos_num() << "\n";

      if (elems_no_sym != elems_with_sym)
        {
          proj_matrix_with_symm.get_proj_matrix_elems_for_one_bin(elems_with_sym, bin);
          elems_with_sym.sort();
          std::cerr << "No Symmetries Iterator || Symmetries Iterator " << std::endl;
          ProjMatrixElemsForOneBin::const_iterator no_sym_iter = elems_no_sym.begin();
          ProjMatrixElemsForOneBin::const_iterator with_sym_iter = elems_with_sym.begin();
          cerr << "      no_sym_iter         ||        sym_iter           ||     Error       \n";
          while (no_sym_iter != elems_no_sym.end() || with_sym_iter != elems_with_sym.end())
            {
              if (no_sym_iter == elems_no_sym.end() || with_sym_iter == elems_with_sym.end()
                  || no_sym_iter->get_coords() != with_sym_iter->get_coords()
                  || fabs(no_sym_iter->get_value() / with_sym_iter->get_value() - 1) > .01)
                {
                  bool inc_no_sym_iter = false;
                  if (no_sym_iter != elems_no_sym.end()
                      && (with_sym_iter == elems_with_sym.end()
                          || coordinates_less(no_sym_iter->get_coords(), with_sym_iter->get_coords())
                          || no_sym_iter->get_coords() == with_sym_iter->get_coords()))
                    {
                      cerr << no_sym_iter->get_coords() << ':' << no_sym_iter->get_value() << "    ||   ";
                      inc_no_sym_iter = true;
                    }
                  else
                    cerr << "                       ||   ";
                  if (with_sym_iter != elems_with_sym.end()
                      && (no_sym_iter == elems_no_sym.end()
                          || !coordinates_less(no_sym_iter->get_coords(), with_sym_iter->get_coords())))
                    {
                      cerr << with_sym_iter->get_coords() << ':' << with_sym_iter->get_value();

                      // If the error is in the same
                      const float err = (no_sym_iter->get_value() / with_sym_iter->get_value() - 1);
                      const float t = 0.01;
                      if (no_sym_iter->get_coords() == with_sym_iter->get_coords())
                        {
                          if ((fabs(err) > t) && (no_sym_iter->get_value() > 1e-5 || with_sym_iter->get_value() > 1e-5))
                            {
                              cerr << "     ||    abs(" << err * 100 << "%) > " << t * 100 << "%";
                            }
                          else
                            {
                              cerr << "     ||    values less than `1e-5%`;";
                            }
                        }

                      ++with_sym_iter;
                    }
                  if (inc_no_sym_iter)
                    ++no_sym_iter;
                  cerr << "\n";
                }
              else
                {
                  if (no_sym_iter != elems_no_sym.end())
                    ++no_sym_iter;
                  if (with_sym_iter != elems_with_sym.end())
                    ++with_sym_iter;
                }
            }
        }
    }
}
}

void
DataSymmetriesForBins_PET_CartesianGridTests::run_tests_2_proj_matrices(const ProjMatrixByBin& proj_matrix_no_symm,
                                                                        const ProjMatrixByBin& proj_matrix_with_symm)
{
#if 1
  for (int s = -proj_data_info_sptr->get_max_segment_num(); s <= proj_data_info_sptr->get_max_segment_num(); ++s)
    for (int v = proj_data_info_sptr->get_min_view_num(); v <= proj_data_info_sptr->get_max_view_num(); ++v)
      for (int timing_pos = proj_data_info_sptr->get_min_tof_pos_num(); timing_pos <= proj_data_info_sptr->get_max_tof_pos_num();
           ++timing_pos)
        for (int a = proj_data_info_sptr->get_min_axial_pos_num(s); a <= proj_data_info_sptr->get_max_axial_pos_num(s); ++a)
          for (int t = -6; t <= 6; t += 3)
            {
              const Bin bin(s, v, a, t, timing_pos);
              // SYM if (proj_matrix_with_symm.get_symmetries_ptr()->is_basic(bin))
              run_tests_2_proj_matrices_1_bin(proj_matrix_no_symm, proj_matrix_with_symm, bin);
            }

#else
  const int oblique_seg_num = proj_data_info_sptr->get_max_segment_num();
  const int view45 = proj_data_info_sptr->get_num_views() / 4;
  assert(fabs(proj_data_info_sptr->get_phi(Bin(0, view45, 0, 0)) - _PI / 4) < .001);

  {
    const Bin bin(oblique_seg_num, 1, 5, 6);
    run_tests_2_proj_matrices_1_bin(proj_matrix_no_symm, proj_matrix_with_symm, bin);
  }
  {
    const Bin bin(oblique_seg_num, 1, 5, -6);
    run_tests_2_proj_matrices_1_bin(proj_matrix_no_symm, proj_matrix_with_symm, bin);
  }
  {
    const Bin bin(-oblique_seg_num, 1, 5, 6);
    run_tests_2_proj_matrices_1_bin(proj_matrix_no_symm, proj_matrix_with_symm, bin);
  }
  {
    const Bin bin(oblique_seg_num, 1, 5, -6);
    run_tests_2_proj_matrices_1_bin(proj_matrix_no_symm, proj_matrix_with_symm, bin);
  }

  {
    const Bin bin(oblique_seg_num, 1, 5, 0);
    run_tests_2_proj_matrices_1_bin(proj_matrix_no_symm, proj_matrix_with_symm, bin);
  }
  {
    const Bin bin(-oblique_seg_num, 1, 5, 0);
    run_tests_2_proj_matrices_1_bin(proj_matrix_no_symm, proj_matrix_with_symm, bin);
  }
  {
    const Bin bin(0, 1, 5, 6);
    run_tests_2_proj_matrices_1_bin(proj_matrix_no_symm, proj_matrix_with_symm, bin);
  }
  {
    const Bin bin(0, 1, 5, 0);
    run_tests_2_proj_matrices_1_bin(proj_matrix_no_symm, proj_matrix_with_symm, bin);
  }
  {
    const Bin bin(oblique_seg_num, 0, 5, 6);
    run_tests_2_proj_matrices_1_bin(proj_matrix_no_symm, proj_matrix_with_symm, bin);
  }
  {
    const Bin bin(-oblique_seg_num, 0, 5, 6);
    run_tests_2_proj_matrices_1_bin(proj_matrix_no_symm, proj_matrix_with_symm, bin);
  }
  {
    const Bin bin(oblique_seg_num, 0, 5, 0);
    run_tests_2_proj_matrices_1_bin(proj_matrix_no_symm, proj_matrix_with_symm, bin);
  }
  {
    const Bin bin(-oblique_seg_num, 0, 5, 0);
    run_tests_2_proj_matrices_1_bin(proj_matrix_no_symm, proj_matrix_with_symm, bin);
  }
  {
    const Bin bin(0, 0, 5, 6);
    run_tests_2_proj_matrices_1_bin(proj_matrix_no_symm, proj_matrix_with_symm, bin);
  }
  {
    const Bin bin(0, 0, 5, -6);
    run_tests_2_proj_matrices_1_bin(proj_matrix_no_symm, proj_matrix_with_symm, bin);
  }
  {
    const Bin bin(0, 0, 5, 0);
    run_tests_2_proj_matrices_1_bin(proj_matrix_no_symm, proj_matrix_with_symm, bin);
  }
  {
    const Bin bin(oblique_seg_num, view45, 5, 6);
    run_tests_2_proj_matrices_1_bin(proj_matrix_no_symm, proj_matrix_with_symm, bin);
  }
  {
    const Bin bin(oblique_seg_num, view45, 5, -6);
    run_tests_2_proj_matrices_1_bin(proj_matrix_no_symm, proj_matrix_with_symm, bin);
  }
  {
    const Bin bin(-oblique_seg_num, view45, 5, 6);
    run_tests_2_proj_matrices_1_bin(proj_matrix_no_symm, proj_matrix_with_symm, bin);
  }
  {
    const Bin bin(-oblique_seg_num, view45, 5, -6);
    run_tests_2_proj_matrices_1_bin(proj_matrix_no_symm, proj_matrix_with_symm, bin);
  }
  {
    const Bin bin(oblique_seg_num, view45, 5, 0);
    run_tests_2_proj_matrices_1_bin(proj_matrix_no_symm, proj_matrix_with_symm, bin);
  }
  {
    const Bin bin(-oblique_seg_num, view45, 5, 0);
    run_tests_2_proj_matrices_1_bin(proj_matrix_no_symm, proj_matrix_with_symm, bin);
  }

  {
    const Bin bin(0, view45, 5, 6);
    run_tests_2_proj_matrices_1_bin(proj_matrix_no_symm, proj_matrix_with_symm, bin);
  }
  {
    const Bin bin(0, view45, 5, 0);
    run_tests_2_proj_matrices_1_bin(proj_matrix_no_symm, proj_matrix_with_symm, bin);
  }

  {
    const Bin bin(oblique_seg_num, 2 * view45 + 1, 5, 6);
    run_tests_2_proj_matrices_1_bin(proj_matrix_no_symm, proj_matrix_with_symm, bin);
  }
  {
    const Bin bin(oblique_seg_num, 2 * view45 + 1, 5, -6);
    run_tests_2_proj_matrices_1_bin(proj_matrix_no_symm, proj_matrix_with_symm, bin);
  }
  {
    const Bin bin(-oblique_seg_num, 2 * view45 + 1, 5, 6);
    run_tests_2_proj_matrices_1_bin(proj_matrix_no_symm, proj_matrix_with_symm, bin);
  }
  {
    const Bin bin(-oblique_seg_num, 2 * view45 + 1, 5, -6);
    run_tests_2_proj_matrices_1_bin(proj_matrix_no_symm, proj_matrix_with_symm, bin);
  }
  {
    const Bin bin(oblique_seg_num, 2 * view45 + 1, 5, 0);
    run_tests_2_proj_matrices_1_bin(proj_matrix_no_symm, proj_matrix_with_symm, bin);
  }
  {
    const Bin bin(-oblique_seg_num, 2 * view45 + 1, 5, 0);
    run_tests_2_proj_matrices_1_bin(proj_matrix_no_symm, proj_matrix_with_symm, bin);
  }

  {
    const Bin bin(0, 2 * view45 + 1, 5, 6);
    run_tests_2_proj_matrices_1_bin(proj_matrix_no_symm, proj_matrix_with_symm, bin);
  }
  {
    const Bin bin(0, 2 * view45 + 1, 5, 0);
    run_tests_2_proj_matrices_1_bin(proj_matrix_no_symm, proj_matrix_with_symm, bin);
  }
#endif
}

void
DataSymmetriesForBins_PET_CartesianGridTests::run_tests_all_symmetries(
    const shared_ptr<ProjDataInfo>& proj_data_info_sptr, const shared_ptr<DiscretisedDensity<3, float>>& density_sptr)
{

  ProjMatrixByBinUsingRayTracing proj_matrix_no_sym;
  {
    stringstream str;
    str << "Ray Tracing Matrix Parameters :=\n"
           "restrict to cylindrical FOV := 1\n"
           "number of rays in tangential direction to trace for each bin := 1\n"
           "use actual detector boundaries := 0\n"
           "do symmetry 90degrees min phi := 0\n"
           "do symmetry 180degrees min phi := 0\n"
           "do_symmetry_swap_segment := 0\n"
           "do_symmetry_swap_s := 0\n"
           "do_symmetry_shift_z := 0\n"
           "End Ray Tracing Matrix Parameters :=\n";
    if (!check(proj_matrix_no_sym.parse(str), "parsing projection matrix parameters"))
      return;
    proj_matrix_no_sym.set_up(proj_data_info_sptr, density_sptr);
  }

  {
    cerr << "\t\tTesting with all symmetries\n";
    ProjMatrixByBinUsingRayTracing proj_matrix_with_sym;

    stringstream str;
    str << "Ray Tracing Matrix Parameters :=\n"
           "restrict to cylindrical FOV := 1\n"
           "number of rays in tangential direction to trace for each bin := 1\n"
           "use actual detector boundaries := 0\n"
           "do symmetry 90degrees min phi := 1\n"
           "do symmetry 180degrees min phi := 1\n"
           "do_symmetry_swap_segment := 1\n"
           "do_symmetry_swap_s := 1\n"
           "do_symmetry_shift_z := 1\n"
           "End Ray Tracing Matrix Parameters :=\n";
    if (check(proj_matrix_with_sym.parse(str), "parsing projection matrix parameters"))
      {
        proj_matrix_with_sym.set_up(proj_data_info_sptr, density_sptr);
        run_tests_2_proj_matrices(proj_matrix_no_sym, proj_matrix_with_sym);
      }
  }
  {
    cerr << "\t\tTesting with all symmetries except 90-phi\n";
    ProjMatrixByBinUsingRayTracing proj_matrix_with_sym;

    stringstream str;
    str << "Ray Tracing Matrix Parameters :=\n"
           "restrict to cylindrical FOV := 1\n"
           "number of rays in tangential direction to trace for each bin := 1\n"
           "use actual detector boundaries := 0\n"
           "do symmetry 90degrees min phi := 0\n"
           "do symmetry 180degrees min phi := 1\n"
           "do_symmetry_swap_segment := 1\n"
           "do_symmetry_swap_s := 1\n"
           "do_symmetry_shift_z := 1\n"
           "End Ray Tracing Matrix Parameters :=\n";
    if (check(proj_matrix_with_sym.parse(str), "parsing projection matrix parameters"))
      {
        proj_matrix_with_sym.set_up(proj_data_info_sptr, density_sptr);
        run_tests_2_proj_matrices(proj_matrix_no_sym, proj_matrix_with_sym);
      }
  }
  {
    cerr << "\t\tTesting with all symmetries except phi symms\n";
    ProjMatrixByBinUsingRayTracing proj_matrix_with_sym;

    stringstream str;
    str << "Ray Tracing Matrix Parameters :=\n"
           "restrict to cylindrical FOV := 1\n"
           "number of rays in tangential direction to trace for each bin := 1\n"
           "use actual detector boundaries := 0\n"
           "do symmetry 90degrees min phi := 0\n"
           "do symmetry 180degrees min phi := 0\n"
           "do_symmetry_swap_segment := 1\n"
           "do_symmetry_swap_s := 1\n"
           "do_symmetry_shift_z := 1\n"
           "End Ray Tracing Matrix Parameters :=\n";
    if (check(proj_matrix_with_sym.parse(str), "parsing projection matrix parameters"))
      {
        proj_matrix_with_sym.set_up(proj_data_info_sptr, density_sptr);
        run_tests_2_proj_matrices(proj_matrix_no_sym, proj_matrix_with_sym);
      }
  }
  {
    cerr << "\t\tTesting with all symmetries except swap_segment\n";
    ProjMatrixByBinUsingRayTracing proj_matrix_with_sym;

    stringstream str;
    str << "Ray Tracing Matrix Parameters :=\n"
           "restrict to cylindrical FOV := 1\n"
           "number of rays in tangential direction to trace for each bin := 1\n"
           "use actual detector boundaries := 0\n"
           "do symmetry 90degrees min phi := 1\n"
           "do symmetry 180degrees min phi := 1\n"
           "do_symmetry_swap_segment := 0\n"
           "do_symmetry_swap_s := 1\n"
           "do_symmetry_shift_z := 1\n"
           "End Ray Tracing Matrix Parameters :=\n";
    if (check(proj_matrix_with_sym.parse(str), "parsing projection matrix parameters"))
      {
        proj_matrix_with_sym.set_up(proj_data_info_sptr, density_sptr);
        run_tests_2_proj_matrices(proj_matrix_no_sym, proj_matrix_with_sym);
      }
  }
  {
    cerr << "\t\tTesting with all symmetries except swap_s\n";
    ProjMatrixByBinUsingRayTracing proj_matrix_with_sym;

    stringstream str;
    str << "Ray Tracing Matrix Parameters :=\n"
           "restrict to cylindrical FOV := 1\n"
           "number of rays in tangential direction to trace for each bin := 1\n"
           "use actual detector boundaries := 0\n"
           "do symmetry 90degrees min phi := 1\n"
           "do symmetry 180degrees min phi := 1\n"
           "do_symmetry_swap_segment := 1\n"
           "do_symmetry_swap_s := 0\n"
           "do_symmetry_shift_z := 1\n"
           "End Ray Tracing Matrix Parameters :=\n";
    if (check(proj_matrix_with_sym.parse(str), "parsing projection matrix parameters"))
      {
        proj_matrix_with_sym.set_up(proj_data_info_sptr, density_sptr);
        run_tests_2_proj_matrices(proj_matrix_no_sym, proj_matrix_with_sym);
      }
  }
  {
    cerr << "\t\tTesting with all symmetries except shift_z\n";
    ProjMatrixByBinUsingRayTracing proj_matrix_with_sym;

    stringstream str;
    str << "Ray Tracing Matrix Parameters :=\n"
           "restrict to cylindrical FOV := 1\n"
           "number of rays in tangential direction to trace for each bin := 1\n"
           "use actual detector boundaries := 0\n"
           "do symmetry 90degrees min phi := 1\n"
           "do symmetry 180degrees min phi := 1\n"
           "do_symmetry_swap_segment := 1\n"
           "do_symmetry_swap_s := 1\n"
           "do_symmetry_shift_z := 0\n"
           "End Ray Tracing Matrix Parameters :=\n";
    if (check(proj_matrix_with_sym.parse(str), "parsing projection matrix parameters"))
      {
        proj_matrix_with_sym.set_up(proj_data_info_sptr, density_sptr);
        run_tests_2_proj_matrices(proj_matrix_no_sym, proj_matrix_with_sym);
      }
  }
  {
    cerr << "\t\tTesting with only shift_z\n";
    ProjMatrixByBinUsingRayTracing proj_matrix_with_sym;

    stringstream str;
    str << "Ray Tracing Matrix Parameters :=\n"
           "restrict to cylindrical FOV := 1\n"
           "number of rays in tangential direction to trace for each bin := 1\n"
           "use actual detector boundaries := 0\n"
           "do symmetry 90degrees min phi := 0\n"
           "do symmetry 180degrees min phi := 0\n"
           "do_symmetry_swap_segment := 0\n"
           "do_symmetry_swap_s := 0\n"
           "do_symmetry_shift_z := 1\n"
           "End Ray Tracing Matrix Parameters :=\n";
    if (check(proj_matrix_with_sym.parse(str), "parsing projection matrix parameters"))
      {
        proj_matrix_with_sym.set_up(proj_data_info_sptr, density_sptr);
        run_tests_2_proj_matrices(proj_matrix_no_sym, proj_matrix_with_sym);
      }
  }
  {
    cerr << "\t\tTesting with only swap_s\n";
    ProjMatrixByBinUsingRayTracing proj_matrix_with_sym;

    stringstream str;
    str << "Ray Tracing Matrix Parameters :=\n"
           "restrict to cylindrical FOV := 1\n"
           "number of rays in tangential direction to trace for each bin := 1\n"
           "use actual detector boundaries := 0\n"
           "do symmetry 90degrees min phi := 0\n"
           "do symmetry 180degrees min phi := 0\n"
           "do_symmetry_swap_segment := 0\n"
           "do_symmetry_swap_s := 1\n"
           "do_symmetry_shift_z := 0\n"
           "End Ray Tracing Matrix Parameters :=\n";
    if (check(proj_matrix_with_sym.parse(str), "parsing projection matrix parameters"))
      {
        proj_matrix_with_sym.set_up(proj_data_info_sptr, density_sptr);
        run_tests_2_proj_matrices(proj_matrix_no_sym, proj_matrix_with_sym);
      }
  }
  {
    cerr << "\t\tTesting with only swap_segment\n";
    ProjMatrixByBinUsingRayTracing proj_matrix_with_sym;

    stringstream str;
    str << "Ray Tracing Matrix Parameters :=\n"
           "restrict to cylindrical FOV := 1\n"
           "number of rays in tangential direction to trace for each bin := 1\n"
           "use actual detector boundaries := 0\n"
           "do symmetry 90degrees min phi := 0\n"
           "do symmetry 180degrees min phi := 0\n"
           "do_symmetry_swap_segment := 1\n"
           "do_symmetry_swap_s := 0\n"
           "do_symmetry_shift_z := 0\n"
           "End Ray Tracing Matrix Parameters :=\n";
    if (check(proj_matrix_with_sym.parse(str), "parsing projection matrix parameters"))
      {
        proj_matrix_with_sym.set_up(proj_data_info_sptr, density_sptr);
        run_tests_2_proj_matrices(proj_matrix_no_sym, proj_matrix_with_sym);
      }
  }
  {
    cerr << "\t\tTesting with shift_z and swap_s\n";
    ProjMatrixByBinUsingRayTracing proj_matrix_with_sym;

    stringstream str;
    str << "Ray Tracing Matrix Parameters :=\n"
           "restrict to cylindrical FOV := 1\n"
           "number of rays in tangential direction to trace for each bin := 1\n"
           "use actual detector boundaries := 0\n"
           "do symmetry 90degrees min phi := 0\n"
           "do symmetry 180degrees min phi := 0\n"
           "do_symmetry_swap_segment := 0\n"
           "do_symmetry_swap_s := 1\n"
           "do_symmetry_shift_z := 1\n"
           "End Ray Tracing Matrix Parameters :=\n";
    if (check(proj_matrix_with_sym.parse(str), "parsing projection matrix parameters"))
      {
        proj_matrix_with_sym.set_up(proj_data_info_sptr, density_sptr);
        run_tests_2_proj_matrices(proj_matrix_no_sym, proj_matrix_with_sym);
      }
  }
  {
    cerr << "\t\tTesting with shift_z and swap_segment\n";
    ProjMatrixByBinUsingRayTracing proj_matrix_with_sym;

    stringstream str;
    str << "Ray Tracing Matrix Parameters :=\n"
           "restrict to cylindrical FOV := 1\n"
           "number of rays in tangential direction to trace for each bin := 1\n"
           "use actual detector boundaries := 0\n"
           "do symmetry 90degrees min phi := 0\n"
           "do symmetry 180degrees min phi := 0\n"
           "do_symmetry_swap_segment := 1\n"
           "do_symmetry_swap_s := 0\n"
           "do_symmetry_shift_z := 1\n"
           "End Ray Tracing Matrix Parameters :=\n";
    if (check(proj_matrix_with_sym.parse(str), "parsing projection matrix parameters"))
      {
        proj_matrix_with_sym.set_up(proj_data_info_sptr, density_sptr);
        run_tests_2_proj_matrices(proj_matrix_no_sym, proj_matrix_with_sym);
      }
  }
  {
    cerr << "\t\tTesting with swap_s and swap_segment\n";
    ProjMatrixByBinUsingRayTracing proj_matrix_with_sym;

    stringstream str;
    str << "Ray Tracing Matrix Parameters :=\n"
           "restrict to cylindrical FOV := 1\n"
           "number of rays in tangential direction to trace for each bin := 1\n"
           "use actual detector boundaries := 0\n"
           "do symmetry 90degrees min phi := 0\n"
           "do symmetry 180degrees min phi := 0\n"
           "do_symmetry_swap_segment := 1\n"
           "do_symmetry_swap_s := 1\n"
           "do_symmetry_shift_z := 0\n"
           "End Ray Tracing Matrix Parameters :=\n";
    if (check(proj_matrix_with_sym.parse(str), "parsing projection matrix parameters"))
      {
        proj_matrix_with_sym.set_up(proj_data_info_sptr, density_sptr);
        run_tests_2_proj_matrices(proj_matrix_no_sym, proj_matrix_with_sym);
      }
  }
}

void
DataSymmetriesForBins_PET_CartesianGridTests::run_tests_for_1_projdata(const shared_ptr<ProjDataInfo>& proj_data_info_sptr)
{
  CartesianCoordinate3D<float> origin(0, 0, 0);
  const float zoom = 1.F;

  cerr << "\tTests with usual image size\n";

  shared_ptr<DiscretisedDensity<3, float>> density_sptr(new VoxelsOnCartesianGrid<float>(*proj_data_info_sptr, zoom, origin));

  VoxelsOnCartesianGrid<float>& image = dynamic_cast<VoxelsOnCartesianGrid<float>&>(*density_sptr);

  run_tests_all_symmetries(proj_data_info_sptr, density_sptr);

  cerr << "\tTests with shifted origin\n";
  density_sptr->set_origin(image.get_grid_spacing() * CartesianCoordinate3D<float>(3, 0, 0));
  run_tests_all_symmetries(proj_data_info_sptr, density_sptr);

  const int org_z_length = density_sptr->get_length();
  const CartesianCoordinate3D<float> org_voxel_size = image.get_voxel_size();

  cerr << "\tTests with non-standard range of planes (larger)\n";
  density_sptr->set_origin(CartesianCoordinate3D<float>(0, 0, 0));
  density_sptr->grow(
      IndexRange3D(-2, org_z_length + 3, image.get_min_y(), image.get_max_y(), image.get_min_x(), image.get_max_x()));
  run_tests_all_symmetries(proj_data_info_sptr, density_sptr);
#if 0
  if (org_z_length>2)
    {
      cerr << "\tTests with non-standard range of planes (smaller)\n";    
      density_sptr->set_origin(CartesianCoordinate3D<float>(0,0,0));
      density_sptr->resize(IndexRange3D(1, org_z_length-2,
					image.get_min_y(), image.get_max_y(),
					image.get_min_x(), image.get_max_x()));
      run_tests_all_symmetries(proj_data_info_sptr, density_sptr);
    }
#  if 1
  // this test currently fails with the ray tracing projmatrix 
  // (but not with the interpolation projmatrix)
  if (org_z_length>1)
    {
      cerr << "\tTests with non-standard range of planes (smaller (even))\n";    
      density_sptr->set_origin(CartesianCoordinate3D<float>(0,0,0));
      density_sptr->resize(IndexRange3D(1, org_z_length-1,
					image.get_min_y(), image.get_max_y(),
					image.get_min_x(), image.get_max_x()));
      run_tests_all_symmetries(proj_data_info_sptr, density_sptr);
    }
  // this test currently fails with the ray tracing projmatrix 
  // because the symmetries do not work yet with origin shifted over half a voxel
  // (but not with the interpolation projmatrix)
  if (org_z_length>1)
    {
      cerr << "\tTests with non-standard range of planes (smaller (even and shifted origin))\n";    
      //      density_sptr->set_origin(CartesianCoordinate3D<float>(0,0,0));
      density_sptr->set_origin(org_voxel_size*
			   CartesianCoordinate3D<float>(.5F,0,0));
      density_sptr->resize(IndexRange3D(1, org_z_length-1,
					image.get_min_y(), image.get_max_y(),
					image.get_min_x(), image.get_max_x()));
      run_tests_all_symmetries(proj_data_info_sptr, density_sptr);
    }
#  endif
#endif

  {
    cerr << "\tTests with z voxel size 3 times smaller than usual\n";
    density_sptr->set_origin(CartesianCoordinate3D<float>(0, 0, 0));
    image.set_grid_spacing(org_voxel_size / CartesianCoordinate3D<float>(2, 1, 1));
    density_sptr->grow(IndexRange3D(-2,
                                    org_z_length * 2, // note: -2 because grow doesn't allow shrinking!
                                    image.get_min_y(),
                                    image.get_max_y(),
                                    image.get_min_x(),
                                    image.get_max_x()));
    run_tests_all_symmetries(proj_data_info_sptr, density_sptr);
  }

  if (proj_data_info_sptr->get_sampling_in_m(Bin(0, 0, 0, 0)) / org_voxel_size.z() >= 1.999)
    {
      // currently symmetries do not work when the voxel size is larger than the ring_spacing
      // so we only perform these tests when they can work
      {
        cerr << "\tTests with z voxel size 2 times larger than usual\n";
        density_sptr->set_origin(CartesianCoordinate3D<float>(0, 0, 0));
        image.set_grid_spacing(org_voxel_size * CartesianCoordinate3D<float>(2, 1, 1));
        density_sptr->resize(IndexRange3D(
            0, (org_z_length + 1) / 2 - 1, image.get_min_y(), image.get_max_y(), image.get_min_x(), image.get_max_x()));
        run_tests_all_symmetries(proj_data_info_sptr, density_sptr);
      }
      {
        cerr << "\tTests with usual z voxel size 2 times larger, 1 extra plane\n";
        density_sptr->set_origin(CartesianCoordinate3D<float>(0, 0, 0));
        image.set_grid_spacing(org_voxel_size * CartesianCoordinate3D<float>(2, 1, 1));
        density_sptr->grow(
            IndexRange3D(0, (org_z_length + 1) / 2, image.get_min_y(), image.get_max_y(), image.get_min_x(), image.get_max_x()));
        run_tests_all_symmetries(proj_data_info_sptr, density_sptr);
      }
    }
}

void
DataSymmetriesForBins_PET_CartesianGridTests::run_tests()
{
  cerr << "Tests for DataSymmetriesForBins_PET_CartesianGrid\n";
  if (template_proj_data_filename == 0)
    {
      {
        cerr << "Testing span=1\n";
        shared_ptr<Scanner> scanner_sptr(new Scanner(Scanner::E953));
        proj_data_info_sptr.reset(ProjDataInfo::ProjDataInfoCTI(scanner_sptr,
                                                                /*span=*/1,
                                                                /*max_delta=*/5,
                                                                /*num_views=*/8,
                                                                /*num_tang_poss=*/16));

        run_tests_for_1_projdata(proj_data_info_sptr);
      }
      {
        cerr << "Testing span=3\n";
        // warning: make sure that parameters are ok such that hard-wired
        // bins above are fine (e.g. segment 3 should be allowed)
        shared_ptr<Scanner> scanner_sptr(new Scanner(Scanner::E953));
        proj_data_info_sptr.reset(ProjDataInfo::ProjDataInfoCTI(scanner_sptr,
                                                                /*span=*/3,
                                                                /*max_delta=*/13,
                                                                /*num_views=*/8,
                                                                /*num_tang_poss=*/16));

        run_tests_for_1_projdata(proj_data_info_sptr);
      }
      {
        cerr << "Testing with proj_data_info with time-of-flight";
        // warning: make sure that parameters are ok such that hard-wired
        // bins above are fine (e.g. segment 3 should be allowed)
        shared_ptr<Scanner> scanner_sptr(new Scanner(Scanner::PETMR_Signa));
        proj_data_info_sptr.reset(ProjDataInfo::ProjDataInfoCTI(scanner_sptr,
                                                                /*span=*/11,
                                                                /*max_delta=*/5,
                                                                /*num_views=*/scanner_sptr->get_num_detectors_per_ring() / 8,
                                                                /*num_tang_poss=*/64,
                                                                /*arc_corrected*/ false,
                                                                /*tof_mashing*/ 116));

        run_tests_for_1_projdata(proj_data_info_sptr);
      }
      {
        cerr << "Testing with proj_data_info with time-of-flight and no-tilt";
        // warning: make sure that parameters are ok such that hard-wired
        // bins above are fine (e.g. segment 3 should be allowed)
        shared_ptr<Scanner> scanner_sptr(new Scanner(Scanner::test_scanner));
        proj_data_info_sptr.reset(ProjDataInfo::ProjDataInfoCTI(scanner_sptr,
                                                                /*span=*/1,
                                                                /*max_delta=*/3,
                                                                /*num_views=*/scanner_sptr->get_num_detectors_per_ring() / 8,
                                                                /*num_tang_poss=*/16,
                                                                /*arc_corrected*/ false,
                                                                /*tof_mashing*/ 112));
        run_tests_for_1_projdata(proj_data_info_sptr);
      }
    }
  else
    {
      shared_ptr<ProjData> proj_data_sptr = ProjData::read_from_file(template_proj_data_filename);
      proj_data_info_sptr = proj_data_sptr->get_proj_data_info_sptr()->create_shared_clone();
      run_tests_for_1_projdata(proj_data_info_sptr);
    }
}

END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int
main(int argc, char** argv)
{
  DataSymmetriesForBins_PET_CartesianGridTests tests(argc == 2 ? argv[1] : 0);
  tests.run_tests();
  return tests.main_return_value();
}
