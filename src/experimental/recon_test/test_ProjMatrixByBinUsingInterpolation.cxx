//
//
/*!

  \file
  \ingroup test

  \brief Test program for ProjMatrixByBinUsingInterpolation

   \author Kris Thielemans

*/
/*
    Copyright (C) 2003- 2004, Hammersmith Imanet
    See STIR/LICENSE.txt for details
*/

#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/IndexRange3D.h"
#include "stir/ProjData.h"
#include "stir/ProjDataInfo.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/Scanner.h"
#include "stir/IndexRange.h"
#include "stir_experimental/recon_buildblock/ProjMatrixByBinUsingInterpolation.h"
#include "stir/recon_buildblock/ProjMatrixElemsForOneBin.h"
#include "stir/recon_buildblock/DataSymmetriesForBins.h"
#include "stir/RunTests.h"
#include "stir/stream.h"
#include <iostream>
#include <sstream>
#include <math.h>
#ifndef STIR_NO_NAMESPACES
using std::stringstream;
using std::cerr;
#endif

START_NAMESPACE_STIR

template <typename T>
bool
coordinates_less(const BasicCoordinate<3, T>& el1, const BasicCoordinate<3, T>& el2) {
  return el1[1] < el2[1] || (el1[1] == el2[1] && (el1[2] < el2[2] || (el1[2] == el2[2] && el1[3] < el2[3])));
}

/*!
  \ingroup test
  \brief Test class for ProjMatrixByBinUsingInterpolation

*/
class ProjMatrixByBinUsingInterpolationTests : public RunTests {
public:
  ProjMatrixByBinUsingInterpolationTests(char const* template_proj_data_filename = 0);

  void run_tests();

private:
  char const* template_proj_data_filename;

  shared_ptr<ProjDataInfo> proj_data_info_sptr;

  void run_tests_2_proj_matrices_1_bin(const ProjMatrixByBin& proj_matrix_no_symm, const ProjMatrixByBin& proj_matrix_with_symm,
                                       const Bin& bin);
  void run_tests_2_proj_matrices(const ProjMatrixByBin& proj_matrix_no_symm, const ProjMatrixByBin& proj_matrix_with_symm);
  void run_tests_all_symmetries(const shared_ptr<ProjDataInfo>& proj_data_info_sptr,
                                const shared_ptr<DiscretisedDensity<3, float>>& density_sptr);
  void run_tests_for_1_projdata(const shared_ptr<ProjDataInfo>& proj_data_info_sptr);
};

ProjMatrixByBinUsingInterpolationTests::ProjMatrixByBinUsingInterpolationTests(char const* template_proj_data_filename)
    : template_proj_data_filename(template_proj_data_filename) {}

void
ProjMatrixByBinUsingInterpolationTests::run_tests_2_proj_matrices_1_bin(const ProjMatrixByBin& proj_matrix_no_symm,
                                                                        const ProjMatrixByBin& proj_matrix_with_symm,
                                                                        const Bin& bin) {
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
  if (!check(elems_no_sym == elems_with_sym, "comparing lors")) {
    // SYM const Bin bin=*bin_iter;
    cerr << "Current bin:  segment = " << bin.segment_num() << ", axial pos " << bin.axial_pos_num()
         << ", view = " << bin.view_num() << ", tangential_pos_num = " << bin.tangential_pos_num() << "\n";
    cerr << "no sym (" << elems_no_sym.size() << ") with sym (" << elems_with_sym.size() << ")\n";
    ProjMatrixElemsForOneBin::const_iterator no_sym_iter = elems_no_sym.begin();
    ProjMatrixElemsForOneBin::const_iterator with_sym_iter = elems_with_sym.begin();
    while (no_sym_iter != elems_no_sym.end() || with_sym_iter != elems_with_sym.end()) {
      if (no_sym_iter == elems_no_sym.end() || with_sym_iter == elems_with_sym.end() ||
          no_sym_iter->get_coords() != with_sym_iter->get_coords() ||
          fabs(no_sym_iter->get_value() / with_sym_iter->get_value() - 1) > .0002) {
        bool inc_no_sym_iter = false;
        if (no_sym_iter != elems_no_sym.end() &&
            (with_sym_iter == elems_with_sym.end() || coordinates_less(no_sym_iter->get_coords(), with_sym_iter->get_coords()) ||
             no_sym_iter->get_coords() == with_sym_iter->get_coords())) {
          cerr << no_sym_iter->get_coords() << ':' << no_sym_iter->get_value() << "    ||   ";
          inc_no_sym_iter = true;
        } else
          cerr << "                       ||   ";
        if (with_sym_iter != elems_with_sym.end() &&
            (no_sym_iter == elems_no_sym.end() || !coordinates_less(no_sym_iter->get_coords(), with_sym_iter->get_coords()))) {
          cerr << with_sym_iter->get_coords() << ':' << with_sym_iter->get_value();
          ++with_sym_iter;
        }
        if (inc_no_sym_iter)
          ++no_sym_iter;
        cerr << "\n";
      } else {
        if (no_sym_iter != elems_no_sym.end())
          ++no_sym_iter;
        if (with_sym_iter != elems_with_sym.end())
          ++with_sym_iter;
      }
    }
  }
}
}

void
ProjMatrixByBinUsingInterpolationTests::run_tests_2_proj_matrices(const ProjMatrixByBin& proj_matrix_no_symm,
                                                                  const ProjMatrixByBin& proj_matrix_with_symm) {
#if 1
  for (int s = -proj_data_info_sptr->get_max_segment_num(); s <= proj_data_info_sptr->get_max_segment_num(); ++s)
    for (int v = proj_data_info_sptr->get_min_view_num(); v <= proj_data_info_sptr->get_max_view_num(); ++v)
      for (int a = proj_data_info_sptr->get_min_axial_pos_num(s); a <= proj_data_info_sptr->get_max_axial_pos_num(s); ++a)
        for (int t = -6; t <= 6; t += 3) {
          const Bin bin(s, v, a, t);
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
ProjMatrixByBinUsingInterpolationTests::run_tests_all_symmetries(const shared_ptr<ProjDataInfo>& proj_data_info_sptr,
                                                                 const shared_ptr<DiscretisedDensity<3, float>>& density_sptr) {

  ProjMatrixByBinUsingInterpolation proj_matrix_no_sym;
  {
    stringstream str;
    str << "Interpolation Matrix Parameters :=\n"
           "do symmetry 90degrees min phi := 0\n"
           "do symmetry 180degrees min phi := 0\n"
           "do_symmetry_swap_segment := 0\n"
           "do_symmetry_swap_s := 0\n"
           "do_symmetry_shift_z := 0\n"
           "End Interpolation Matrix Parameters :=\n";
    if (!check(proj_matrix_no_sym.parse(str), "parsing projection matrix parameters"))
      return;
    proj_matrix_no_sym.set_up(proj_data_info_sptr, density_sptr);
  }

  {
    cerr << "\t\tTesting with all symmetries\n";
    ProjMatrixByBinUsingInterpolation proj_matrix_with_sym;

    stringstream str;
    str << "Interpolation Matrix Parameters :=\n"
           "do symmetry 90degrees min phi := 1\n"
           "do symmetry 180degrees min phi := 1\n"
           "do_symmetry_swap_segment := 1\n"
           "do_symmetry_swap_s := 1\n"
           "do_symmetry_shift_z := 1\n"
           "End Interpolation Matrix Parameters :=\n";
    if (check(proj_matrix_with_sym.parse(str), "parsing projection matrix parameters")) {
      proj_matrix_with_sym.set_up(proj_data_info_sptr, density_sptr);
      run_tests_2_proj_matrices(proj_matrix_no_sym, proj_matrix_with_sym);
    }
  }
  {
    cerr << "\t\tTesting with all symmetries except 90-phi\n";
    ProjMatrixByBinUsingInterpolation proj_matrix_with_sym;

    stringstream str;
    str << "Interpolation Matrix Parameters :=\n"
           "do symmetry 90degrees min phi := 0\n"
           "do symmetry 180degrees min phi := 1\n"
           "do_symmetry_swap_segment := 1\n"
           "do_symmetry_swap_s := 1\n"
           "do_symmetry_shift_z := 1\n"
           "End Interpolation Matrix Parameters :=\n";
    if (check(proj_matrix_with_sym.parse(str), "parsing projection matrix parameters")) {
      proj_matrix_with_sym.set_up(proj_data_info_sptr, density_sptr);
      run_tests_2_proj_matrices(proj_matrix_no_sym, proj_matrix_with_sym);
    }
  }
  {
    cerr << "\t\tTesting with all symmetries except phi symms\n";
    ProjMatrixByBinUsingInterpolation proj_matrix_with_sym;

    stringstream str;
    str << "Interpolation Matrix Parameters :=\n"
           "do symmetry 90degrees min phi := 0\n"
           "do symmetry 180degrees min phi := 0\n"
           "do_symmetry_swap_segment := 1\n"
           "do_symmetry_swap_s := 1\n"
           "do_symmetry_shift_z := 1\n"
           "End Interpolation Matrix Parameters :=\n";
    if (check(proj_matrix_with_sym.parse(str), "parsing projection matrix parameters")) {
      proj_matrix_with_sym.set_up(proj_data_info_sptr, density_sptr);
      run_tests_2_proj_matrices(proj_matrix_no_sym, proj_matrix_with_sym);
    }
  }
  {
    cerr << "\t\tTesting with all symmetries except swap_segment\n";
    ProjMatrixByBinUsingInterpolation proj_matrix_with_sym;

    stringstream str;
    str << "Interpolation Matrix Parameters :=\n"
           "do symmetry 90degrees min phi := 1\n"
           "do symmetry 180degrees min phi := 1\n"
           "do_symmetry_swap_segment := 0\n"
           "do_symmetry_swap_s := 1\n"
           "do_symmetry_shift_z := 1\n"
           "End Interpolation Matrix Parameters :=\n";
    if (check(proj_matrix_with_sym.parse(str), "parsing projection matrix parameters")) {
      proj_matrix_with_sym.set_up(proj_data_info_sptr, density_sptr);
      run_tests_2_proj_matrices(proj_matrix_no_sym, proj_matrix_with_sym);
    }
  }
  {
    cerr << "\t\tTesting with all symmetries except swap_s\n";
    ProjMatrixByBinUsingInterpolation proj_matrix_with_sym;

    stringstream str;
    str << "Interpolation Matrix Parameters :=\n"
           "do symmetry 90degrees min phi := 1\n"
           "do symmetry 180degrees min phi := 1\n"
           "do_symmetry_swap_segment := 1\n"
           "do_symmetry_swap_s := 0\n"
           "do_symmetry_shift_z := 1\n"
           "End Interpolation Matrix Parameters :=\n";
    if (check(proj_matrix_with_sym.parse(str), "parsing projection matrix parameters")) {
      proj_matrix_with_sym.set_up(proj_data_info_sptr, density_sptr);
      run_tests_2_proj_matrices(proj_matrix_no_sym, proj_matrix_with_sym);
    }
  }
  {
    cerr << "\t\tTesting with all symmetries except shift_z\n";
    ProjMatrixByBinUsingInterpolation proj_matrix_with_sym;

    stringstream str;
    str << "Interpolation Matrix Parameters :=\n"
           "do symmetry 90degrees min phi := 1\n"
           "do symmetry 180degrees min phi := 1\n"
           "do_symmetry_swap_segment := 1\n"
           "do_symmetry_swap_s := 1\n"
           "do_symmetry_shift_z := 0\n"
           "End Interpolation Matrix Parameters :=\n";
    if (check(proj_matrix_with_sym.parse(str), "parsing projection matrix parameters")) {
      proj_matrix_with_sym.set_up(proj_data_info_sptr, density_sptr);
      run_tests_2_proj_matrices(proj_matrix_no_sym, proj_matrix_with_sym);
    }
  }
  {
    cerr << "\t\tTesting with only shift_z\n";
    ProjMatrixByBinUsingInterpolation proj_matrix_with_sym;

    stringstream str;
    str << "Interpolation Matrix Parameters :=\n"
           "do symmetry 90degrees min phi := 0\n"
           "do symmetry 180degrees min phi := 0\n"
           "do_symmetry_swap_segment := 0\n"
           "do_symmetry_swap_s := 0\n"
           "do_symmetry_shift_z := 1\n"
           "End Interpolation Matrix Parameters :=\n";
    if (check(proj_matrix_with_sym.parse(str), "parsing projection matrix parameters")) {
      proj_matrix_with_sym.set_up(proj_data_info_sptr, density_sptr);
      run_tests_2_proj_matrices(proj_matrix_no_sym, proj_matrix_with_sym);
    }
  }
}

void
ProjMatrixByBinUsingInterpolationTests::run_tests_for_1_projdata(const shared_ptr<ProjDataInfo>& proj_data_info_sptr) {
  CartesianCoordinate3D<float> origin(0, 0, 0);
  const float zoom = 1.F;

  cerr << "\tTests with usual image size\n";

  shared_ptr<DiscretisedDensity<3, float>> density_sptr = new VoxelsOnCartesianGrid<float>(*proj_data_info_sptr, zoom, origin);

  VoxelsOnCartesianGrid<float>& image = dynamic_cast<VoxelsOnCartesianGrid<float>&>(*density_sptr);

  run_tests_all_symmetries(proj_data_info_sptr, density_sptr);

  cerr << "\tTests with shifted origin\n";

  density_sptr->set_origin(image.get_grid_spacing() * CartesianCoordinate3D<float>(3, 0, 0));

  run_tests_all_symmetries(proj_data_info_sptr, density_sptr);

  const int org_z_length = density_sptr->get_length();

  cerr << "\tTests with non-standard range of planes (larger)\n";
  density_sptr->set_origin(CartesianCoordinate3D<float>(0, 0, 0));
  density_sptr->grow(
      IndexRange3D(-2, org_z_length + 3, image.get_min_y(), image.get_max_y(), image.get_min_x(), image.get_max_x()));
  run_tests_all_symmetries(proj_data_info_sptr, density_sptr);

  if (org_z_length > 2) {
    cerr << "\tTests with non-standard range of planes (smaller)\n";
    density_sptr->set_origin(CartesianCoordinate3D<float>(0, 0, 0));
    density_sptr->resize(
        IndexRange3D(1, org_z_length - 1, image.get_min_y(), image.get_max_y(), image.get_min_x(), image.get_max_x()));
    run_tests_all_symmetries(proj_data_info_sptr, density_sptr);
  }

  cerr << "\tTests with usual z voxel size 3 times smaller\n";
  density_sptr->set_origin(CartesianCoordinate3D<float>(0, 0, 0));
  image.set_grid_spacing(image.get_grid_spacing() / CartesianCoordinate3D<float>(2, 1, 1));
  density_sptr->grow(IndexRange3D(0, density_sptr->get_length() * 2, image.get_min_y(), image.get_max_y(), image.get_min_x(),
                                  image.get_max_x()));
  run_tests_all_symmetries(proj_data_info_sptr, density_sptr);
}

void
ProjMatrixByBinUsingInterpolationTests::run_tests() {
  cerr << "Tests for ProjMatrixByBinUsingInterpolation\n";
  if (template_proj_data_filename == 0) {
    {
      cerr << "Testing span=1\n";
      shared_ptr<Scanner> scanner_sptr = new Scanner(Scanner::E953);
      proj_data_info_sptr = ProjDataInfo::ProjDataInfoCTI(scanner_sptr,
                                                          /*span=*/1,
                                                          /*max_delta=*/5,
                                                          /*num_views=*/32,
                                                          /*num_tang_poss=*/16);

      run_tests_for_1_projdata(proj_data_info_sptr);
    }
    {
      cerr << "Testing span=3\n";
      // warning: make sure that parameters are ok such that hard-wired
      // bins above are fine (e.g. segment 3 should be allowed)
      shared_ptr<Scanner> scanner_sptr = new Scanner(Scanner::E953);
      proj_data_info_sptr = ProjDataInfo::ProjDataInfoCTI(scanner_sptr,
                                                          /*span=*/3,
                                                          /*max_delta=*/12,
                                                          /*num_views=*/32,
                                                          /*num_tang_poss=*/16);

      run_tests_for_1_projdata(proj_data_info_sptr);
    }
  } else {
    shared_ptr<ProjData> proj_data_sptr = ProjData::read_from_file(template_proj_data_filename);
    proj_data_info_sptr = proj_data_sptr->get_proj_data_info_sptr()->clone();
    run_tests_for_1_projdata(proj_data_info_sptr);
  }
}

END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int
main(int argc, char** argv) {
  ProjMatrixByBinUsingInterpolationTests tests(argc == 2 ? argv[1] : 0);
  tests.run_tests();
  return tests.main_return_value();
}
