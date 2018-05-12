#include "stir/scatter/CreateTailMaskFromACFs.h"
#include "stir/Bin.h"
#include "boost/lambda/lambda.hpp"

START_NAMESPACE_STIR

void
CreateTailMaskFromACFs::
set_input_projdata_sptr(shared_ptr<ProjData> & arg)
{
    this->ACF_sptr = arg;
}

void
CreateTailMaskFromACFs::
set_input_projdata(std::string& arg)
{
    this->ACF_sptr =
            ProjData::read_from_file(arg);
}

void
CreateTailMaskFromACFs::
set_output_projdata_sptr(shared_ptr<ProjData>& arg)
{
    this->mask_proj_data = arg;
}

void
CreateTailMaskFromACFs::
set_output_projdata(std::string& arg)
{
    this->mask_proj_data.reset(new ProjDataInterfile(ACF_sptr->get_exam_info_sptr(),
                                                     ACF_sptr->get_proj_data_info_ptr()->create_shared_clone(),
                                                     arg));
}


shared_ptr<ProjData>
CreateTailMaskFromACFs::
get_output_projdata_sptr()
{
 return this->mask_proj_data;
}

void
CreateTailMaskFromACFs::
set_defaults()
{
    ACF_threshold = 1.1F;
    safety_margin = 4;
}

CreateTailMaskFromACFs::
CreateTailMaskFromACFs()
{
    this->set_defaults();
}

void
CreateTailMaskFromACFs::
initialise_keymap()
{
    this->parser.add_start_key("CreateTailMaskFromACFs");
    this->parser.add_stop_key("END CreateTailMaskFromACFs");
    this->parser.add_key("ACF-filename",
                         &_input_filename);
    this->parser.add_key("output-filename",
                         &_output_filename);
    this->parser.add_key("ACF-threshold",
                         &ACF_threshold);
    this->parser.add_key("safety-margin",
                         &safety_margin);
}

bool
CreateTailMaskFromACFs::
post_processing()
{
    if (ACF_threshold<=1)
        error("ACF-threshold should be larger than 1");

    if(this->_input_filename.size() > 0)
        this->set_input_projdata(this->_input_filename);


    if(this->_output_filename.size() > 0)
        this->set_output_projdata(this->_output_filename);

    return false;
}

Succeeded
CreateTailMaskFromACFs::
process_data()
{
    if (is_null_ptr(this->ACF_sptr))
        error("Check the attenuation_correct_factors file");

    if (is_null_ptr(this->mask_proj_data))
        error("Please set output file");

    Bin bin;
    {
        for (bin.segment_num()=this->mask_proj_data->get_min_segment_num();
             bin.segment_num()<=this->mask_proj_data->get_max_segment_num();
             ++bin.segment_num())
            for (bin.axial_pos_num()=
                 this->mask_proj_data->get_min_axial_pos_num(bin.segment_num());
                 bin.axial_pos_num()<=this->mask_proj_data->get_max_axial_pos_num(bin.segment_num());
                 ++bin.axial_pos_num())
            {
                const Sinogram<float> att_sinogram
                        (this->ACF_sptr->get_sinogram(bin.axial_pos_num(),bin.segment_num()));
                Sinogram<float> mask_sinogram
                        (this->mask_proj_data->get_empty_sinogram(bin.axial_pos_num(),bin.segment_num()));

                std::size_t count=0;
                for (bin.view_num()=this->mask_proj_data->get_min_view_num();
                     bin.view_num()<=this->mask_proj_data->get_max_view_num();
                     ++bin.view_num())
                {
#ifdef SCFOLD
                    for (bin.tangential_pos_num()=
                         mask_proj_data.get_min_tangential_pos_num();
                         bin.tangential_pos_num()<=
                         mask_proj_data.get_max_tangential_pos_num();
                         ++bin.tangential_pos_num())
                        if (att_sinogram[bin.view_num()][bin.tangential_pos_num()]<ACF_threshold &&
                                (mask_radius_in_mm<0 || mask_radius_in_mm>= std::fabs(scatter_proj_data.get_proj_data_info_ptr()->get_s(bin))))
                        {
                            ++count;
                            mask_sinogram[bin.view_num()][bin.tangential_pos_num()]=1;
                        }
                        else
                            mask_sinogram[bin.view_num()][bin.tangential_pos_num()]=0;
#else
                    const Array<1,float>& att_line=att_sinogram[bin.view_num()];

                    using boost::lambda::_1;

                    // find left and right mask sizes
                    // sorry: a load of ugly casting to make sure we allow all datatypes
                    std::size_t mask_left_size =
                            static_cast<std::size_t>(
                                std::max(0,
                                         static_cast<int>
                                         (std::find_if(att_line.begin(), att_line.end(),
                                                       _1>=ACF_threshold) -
                                          att_line.begin()) -
                                         safety_margin)
                                );
                    std::size_t mask_right_size =
                            static_cast<std::size_t>(
                                std::max(0,
                                         static_cast<int>
                                         (std::find_if(att_line.rbegin(), att_line.rend() - mask_left_size,
                                                       _1>=ACF_threshold) -
                                          att_line.rbegin()) -
                                         safety_margin)
                                );
#if 0
                    std::cout << "mask sizes " << mask_left_size << ", " << mask_right_size << '\n';
#endif
                    std::fill(mask_sinogram[bin.view_num()].begin(),
                            mask_sinogram[bin.view_num()].begin() + mask_left_size,
                            1.F);
                    std::fill(mask_sinogram[bin.view_num()].rbegin(),
                            mask_sinogram[bin.view_num()].rbegin() + mask_right_size,
                            1.F);
                    count += mask_left_size + mask_right_size;
#endif
                }
                std::cout << count << " bins in mask for sinogram at segment "
                          << bin.segment_num() << ", axial_pos " << bin.axial_pos_num() << "\n";
                if (this->mask_proj_data->set_sinogram(mask_sinogram) != Succeeded::yes)
                    return Succeeded::no;
            }
    }
    return Succeeded::yes;
}

END_NAMESPACE_STIR
