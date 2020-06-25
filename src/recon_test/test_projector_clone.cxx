//
//
/*!

  \file
  \ingroup test

  \brief Test program for projector clones.

  \author Richard Brown

*/
/*
    Copyright (C) 2020, University College London
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


#include "stir/RunTests.h"
#include "stir/num_threads.h"

// Nasty, but means we can access private and public members
#define private public
#define protected public

#include "stir/recon_buildblock/ForwardProjectorByBinUsingProjMatrixByBin.h"
#include "stir/recon_buildblock/BackProjectorByBinUsingProjMatrixByBin.h"
#include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h"
#include "stir/ProjDataInfo.h"
#include "stir/ProjDataInMemory.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#ifdef STIR_WITH_NiftyPET_PROJECTOR
#include "stir/recon_buildblock/NiftyPET_projector/ForwardProjectorByBinNiftyPET.h"
#include "stir/recon_buildblock/NiftyPET_projector/BackProjectorByBinNiftyPET.h"
#endif

START_NAMESPACE_STIR

/*!
  \ingroup test
  \brief Test class for projector clones
*/
class ProjectorCloneTests: public RunTests
{
public:
    virtual void run_tests();
    bool _verbose;
protected:
  template<class Type>
  void check_different_addresses(
          const std::string &description,
          const shared_ptr<Type> obj_1_sptr,
          const shared_ptr<Type> obj_2_sptr);

  template<class Type>
  void check_common_projector_vars(
          const shared_ptr<Type> obj_1_sptr,
          const shared_ptr<Type> obj_2_sptr);
  void test_UsingProjMatrixByBin();
  void test_UsingNiftyPET();

  void create_im_and_pdim();
  shared_ptr<VoxelsOnCartesianGrid<float> > _im_sptr;
  shared_ptr<ProjDataInMemory> _pdim_sptr;
};

void ProjectorCloneTests::run_tests()
{
    create_im_and_pdim();
    test_UsingProjMatrixByBin();
    test_UsingNiftyPET();
}

void ProjectorCloneTests::create_im_and_pdim()
{
    // Create scanner
    shared_ptr<Scanner> scanner_sptr(new Scanner(Scanner::Siemens_mMR));

    // ExamInfo
    shared_ptr<ExamInfo> exam_info_sptr(new ExamInfo);
    exam_info_sptr->imaging_modality = ImagingModality::PT;

    shared_ptr<ProjDataInfo> pdi_sptr(
                ProjDataInfo::construct_proj_data_info(
                    scanner_sptr,
                    11, // span
                    /* mMR needs maxDelta of */60,
                    scanner_sptr->get_num_detectors_per_ring()/2,
                    scanner_sptr->get_max_num_non_arccorrected_bins(),
                    /* arc_correction*/false));

    _pdim_sptr = MAKE_SHARED<ProjDataInMemory>(exam_info_sptr, pdi_sptr);

    _im_sptr = MAKE_SHARED<VoxelsOnCartesianGrid<float> >(
                _pdim_sptr->get_exam_info_sptr(),
                *pdi_sptr,
                1.f,CartesianCoordinate3D<float>(0,0,0));
}

template<class Type>
void
ProjectorCloneTests::
check_different_addresses(
        const std::string &description,
        const shared_ptr<Type> obj_1_sptr,
        const shared_ptr<Type> obj_2_sptr)
{
    if (obj_1_sptr.get() == obj_2_sptr.get()) {
        std::cout << "\nError: Expected addresses of " << description <<
                     " to differ, but they match."
                     "\n\tAddress 1: " << obj_1_sptr.get() <<
                     "\n\tAddress 2: " << obj_2_sptr.get() <<
                     ".\n";
        everything_ok = false;
    }
    if (_verbose)
        std::cout << "\nSuccess: Expected addresses of " << description <<
                     " to match, which they do."
                     "\n\tAddress 1: " << obj_1_sptr.get() <<
                     "\n\tAddress 2: " << obj_2_sptr.get() <<
                     ".\n";
}

template<class Type>
void
ProjectorCloneTests::
check_common_projector_vars(
        const shared_ptr<Type> obj_1_sptr,
        const shared_ptr<Type> obj_2_sptr)
{
    check_different_addresses("Projector", obj_1_sptr, obj_2_sptr);
    check_different_addresses("DiscDensity", obj_1_sptr->_density_sptr, obj_2_sptr->_density_sptr);
    check_different_addresses("ProjDataInfo", obj_1_sptr->_proj_data_info_sptr, obj_2_sptr->_proj_data_info_sptr);
}

void
ProjectorCloneTests::
test_UsingProjMatrixByBin()
{
    std::cout << "\n testing Projectors UsingProjMatrixByBin...\n";
    std::cout << "\n testing Forward...\n";
    shared_ptr<ProjMatrixByBinUsingRayTracing> proj_matrix_by_bin_sptr =
            MAKE_SHARED<ProjMatrixByBinUsingRayTracing>();

    shared_ptr<ForwardProjectorByBinUsingProjMatrixByBin> fwd_prj_sptr =
            MAKE_SHARED<ForwardProjectorByBinUsingProjMatrixByBin>(proj_matrix_by_bin_sptr);
    fwd_prj_sptr->set_up(_pdim_sptr->get_proj_data_info_sptr(), _im_sptr);

    shared_ptr<ForwardProjectorByBinUsingProjMatrixByBin> fwd_prj_clone_sptr =
            fwd_prj_sptr->create_shared_clone();
    check_common_projector_vars(fwd_prj_sptr, fwd_prj_clone_sptr);

    std::cout << "\n testing Back...\n";
    shared_ptr<BackProjectorByBinUsingProjMatrixByBin> bck_prj_sptr =
            MAKE_SHARED<BackProjectorByBinUsingProjMatrixByBin>(proj_matrix_by_bin_sptr);
    bck_prj_sptr->set_up(_pdim_sptr->get_proj_data_info_sptr(), _im_sptr);

    shared_ptr<BackProjectorByBinUsingProjMatrixByBin> bck_prj_clone_sptr =
            bck_prj_sptr->create_shared_clone();
    check_common_projector_vars(bck_prj_sptr, bck_prj_clone_sptr);
}

void
ProjectorCloneTests::
test_UsingNiftyPET()
{
#ifdef STIR_WITH_NiftyPET_PROJECTOR
    std::cout << "\n testing Projectors UsingNiftyPET...\n";
    std::cout << "\n testing Forward...\n";

    shared_ptr<ForwardProjectorByBinNiftyPET> fwd_prj_sptr =
            MAKE_SHARED<ForwardProjectorByBinNiftyPET>();
    fwd_prj_sptr->set_up(_pdim_sptr->get_proj_data_info_sptr(), _im_sptr);

    shared_ptr<ForwardProjectorByBinNiftyPET> fwd_prj_clone_sptr =
            fwd_prj_sptr->create_shared_clone();
    check_common_projector_vars(fwd_prj_sptr, fwd_prj_clone_sptr);
    check_different_addresses("ProjectedData", fwd_prj_sptr->_projected_data_sptr, fwd_prj_clone_sptr->_projected_data_sptr);

    std::cout << "\n testing Back...\n";
    shared_ptr<BackProjectorByBinNiftyPET> bck_prj_sptr =
            MAKE_SHARED<BackProjectorByBinNiftyPET>();
    bck_prj_sptr->set_up(_pdim_sptr->get_proj_data_info_sptr(), _im_sptr);

    shared_ptr<BackProjectorByBinNiftyPET> bck_prj_clone_sptr =
            bck_prj_sptr->create_shared_clone();
    check_common_projector_vars(bck_prj_sptr, bck_prj_clone_sptr);

#endif
}

END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int main(int argc, char** argv)
{
    set_default_num_threads();
    ProjectorCloneTests tests;
    if (argc > 1 && strcmp(argv[1],"--verbose")==0)
        tests._verbose = true;
    tests.run_tests();
    if (!tests.is_everything_ok())
        return tests.main_return_value();
}
