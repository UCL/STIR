//
//
/*
    Copyright (C) 2020, University College London
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2.0 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    See STIR/LICENSE.txt for details
*/
/*!
\file
\ingroup utilities
\brief Convert between NiftyPET and STIR imagees and projdata
\author Richard Brown
*/

#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/IndexRange3D.h"
#include "stir/IO/OutputFileFormat.h"
#include "stir/IO/read_from_file.h"
#include "stir/recon_buildblock/niftypet_projector/ProjectorByBinNiftyPETHelper.h"
#include "stir/is_null_ptr.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/ProjDataInMemory.h"
#include "def.h"

USING_NAMESPACE_STIR

static void print_usage_and_exit( const char * const program_name, const int exit_status)
{
    std::cerr << "\n\nUsage : " << program_name << " [-h|--help] output_filename input_filename <image|sinogram> <toSTIR|toNP> [--cuda_device <val>] [--stir_im_par <par_file>]\n\n";
    exit(exit_status);
}

static void save_disc_density(const DiscretisedDensity<3,float> &out_im_stir, const std::string &filename, const std::string &stir_im_par_fname)
{
    shared_ptr<OutputFileFormat<DiscretisedDensity<3,float> > > output_file_format_sptr;
    // Use the default
    if (stir_im_par_fname.empty())
          output_file_format_sptr = OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr();
    // If parameter file has been given, try to read it
    else {
        KeyParser parser;
        parser.add_start_key("OutputFileFormat Parameters");
        parser.add_parsing_key("output file format type", &output_file_format_sptr);
        parser.add_stop_key("END");
        std::ifstream in(stir_im_par_fname);
        if (!parser.parse(in) || is_null_ptr(output_file_format_sptr))
            throw std::runtime_error("Failed to parse output format file (" + stir_im_par_fname + ").");
    }
    output_file_format_sptr->write_to_file(filename, out_im_stir);
}

static shared_ptr<const DiscretisedDensity<3,float> > read_disc_density(const std::string &filename)
{
    // Read
    shared_ptr<const DiscretisedDensity<3,float> > disc_sptr(read_from_file<DiscretisedDensity<3,float> >(filename));
    // Check
    if (is_null_ptr(disc_sptr))
        throw std::runtime_error("Failed to read file: " + filename + ".");

    return disc_sptr;
}

static void save_np_vec(const std::vector<float> &vec, const std::string &filename)
{
    std::ofstream fout(filename, std::ios::out | std::ios::binary);
    fout.write(reinterpret_cast<const char*>(&vec[0]), vec.size()*sizeof(float));
    fout.close();
}

static std::shared_ptr<VoxelsOnCartesianGrid<float> > create_stir_im()
{
    // Create STIR image with correct dimensions
    int nz(SZ_IMZ), nx(SZ_IMX), ny(SZ_IMY);
    float sz(SZ_VOXZ*10.f), sx(SZ_VOXY*10.f), sy(SZ_VOXY*10.f);
    shared_ptr<VoxelsOnCartesianGrid<float> > out_im_stir_sptr =
        MAKE_SHARED<VoxelsOnCartesianGrid<float> >(
            IndexRange3D(0, nz - 1, -(ny / 2), -(ny / 2) + ny - 1, -(nx / 2), -(nx / 2) + nx - 1),
            CartesianCoordinate3D<float>(0.f, 0.f, 0.f),
            CartesianCoordinate3D<float>(sz, sy, sx));
    return out_im_stir_sptr;
}

static std::shared_ptr<ProjData> create_stir_sino()
{
    const int span=11;
    const int max_ring_diff=60;
    const int view_mash_factor=1;
    shared_ptr<ExamInfo> ei_sptr = MAKE_SHARED<ExamInfo>();
    ei_sptr->imaging_modality = ImagingModality::PT;
    shared_ptr<Scanner> scanner_sptr(Scanner::get_scanner_from_name("mMR"));
    int num_views = scanner_sptr->get_num_detectors_per_ring() / 2 / view_mash_factor;
    int num_tang_pos = scanner_sptr->get_max_num_non_arccorrected_bins();
    shared_ptr<ProjDataInfo> pdi_sptr = ProjDataInfo::construct_proj_data_info
            (scanner_sptr, span, max_ring_diff, num_views, num_tang_pos, false);
    shared_ptr<ProjDataInMemory> pd_sptr = MAKE_SHARED<ProjDataInMemory>(ei_sptr, pdi_sptr);
    return pd_sptr;
}

int
main(int argc, char **argv)
{
    try {
        const char * const program_name = argv[0];

        // Check for help request
        for (int i=1; i<argc; ++i)
            if (strcmp(argv[i],"-h")==0 || strcmp(argv[i],"--help")==0)
                print_usage_and_exit(program_name, EXIT_SUCCESS);

        // Check for all compulsory arguments
        if (argc<5)
            print_usage_and_exit(program_name, EXIT_FAILURE);

        // Get filenames
        const std::string output_filename = argv[1];
        const std::string input_filename  = argv[2];

        // Am i dealing with images or sinograms?
        bool is_image;
        if (strcmp(argv[3],"image")==0)
            is_image = true;
        else if (strcmp(argv[3],"sinogram")==0)
            is_image = false;
        else {
            std::cerr << "\nExpected \"image\" or \"sinogram\"\n, got: " << argv[3] << "\n";
            print_usage_and_exit(program_name, EXIT_FAILURE);
        }
        // to STIR or to NiftyPET?
        bool toSTIR;
        if (strcmp(argv[4],"toSTIR")==0)
            toSTIR = true;
        else if (strcmp(argv[4],"toNP")==0)
            toSTIR = false;
        else {
            std::cerr << "\nExpected \"toSTIR\" or \"toNP\"\n, got: " << argv[3] << "\n";
            print_usage_and_exit(program_name, EXIT_FAILURE);
        }

        // skip past compulsory arguments
        argc-=5;
        argv+=5;

        // Set default value for optional arguments
        char cuda_device(0);
        std::string stir_im_par_fname;

        // Loop over remaining input
        while (argc>0 && argv[0][0]=='-') {
            if (strcmp(argv[0], "--cuda_device")==0) {
            cuda_device = std::atoi(argv[1]);
                argc-=1; argv+=1;
            }
            else if (strcmp(argv[0], "--stir_im_par")==0) {
                stir_im_par_fname = argv[1];
                argc-=1; argv+=1;
                if (!(is_image && toSTIR)) {
                    std::cerr << "--stir_im_par can only be supplied when converting to a STIR image.\n";
                    print_usage_and_exit(program_name, EXIT_FAILURE);
                }
            }
            else {
                std::cerr << "Unknown option '" << argv[0] <<"'\n";
                print_usage_and_exit(program_name, EXIT_FAILURE);
            }
        }

        // Set up the niftyPET binary helper
        typedef ProjectorByBinNiftyPETHelper Helper;
        Helper helper;
        helper.set_li2rng_filename("li2rng.dat"  );
        helper.set_li2sn_filename ("li2sn.dat"   );
        helper.set_li2nos_filename("li2nos.dat"  );
        helper.set_s2c_filename   ("s2c.dat"     );
        helper.set_aw2ali_filename("aw2ali.dat"  );
        helper.set_crs_filename   ( "crss.dat"   );
        helper.set_cuda_device_id (  cuda_device );
        helper.set_span           (      11      );
        helper.set_att(0);
        helper.set_verbose(1);
        helper.set_up();

        // if image
        if (is_image) {
            // image NP -> STIR
            if (toSTIR) {
                std::vector<float> input_im_np = Helper::read_binary_file<float>(input_filename);
                shared_ptr<DiscretisedDensity<3,float> > out_im_stir_sptr = create_stir_im();
                helper.convert_image_niftyPET_to_stir(*out_im_stir_sptr, input_im_np);
                save_disc_density(*out_im_stir_sptr, output_filename, stir_im_par_fname);
            }
            // image STIR -> NP
            else {
                shared_ptr<const DiscretisedDensity<3,float> > input_im_stir_sptr =
                        read_disc_density(input_filename);
                std::vector<float> out_im_np = helper.create_niftyPET_image();
                helper.convert_image_stir_to_niftyPET(out_im_np, *input_im_stir_sptr);
                save_np_vec(out_im_np, output_filename);
            }
        }
        // if sinogram
        else {
            // sinogram NP -> STIR
            if (toSTIR) {
                std::vector<float> input_sino_np = Helper::read_binary_file<float>(input_filename);
                shared_ptr<ProjData> output_sino_stir_sptr = create_stir_sino();
                helper.convert_proj_data_niftyPET_to_stir(*output_sino_stir_sptr, input_sino_np);
                output_sino_stir_sptr->write_to_file(output_filename);
            }
            // sinogram STIR -> NP
            else {
                shared_ptr<const ProjData> input_sino_stir_sptr = ProjDataInMemory::read_from_file(input_filename);
                std::vector<float> output_sino_np = helper.create_niftyPET_sinogram_with_gaps();
                helper.convert_proj_data_stir_to_niftyPET(output_sino_np,*input_sino_stir_sptr);
                save_np_vec(output_sino_np, output_filename);
            }
        }
    }

    // If there was an error
    catch(const std::exception &error) {
        std::cerr << "\nError encountered:\n\t" << error.what() << "\n\n";
        return EXIT_FAILURE;
    }
    catch(...) {
        std::cerr << "\nError encountered.\n\n";
        return EXIT_FAILURE;
    }
    return(EXIT_SUCCESS);
}
