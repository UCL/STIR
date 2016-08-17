#include "stir/recon_buildblock/General_Reconstruction.h"
#include "stir/CPUTimer.h"
#include "stir/HighResWallClockTimer.h"

#include <iostream>
START_NAMESPACE_STIR

General_Reconstruction::
General_Reconstruction()
{
    this->set_defaults();
}

void
General_Reconstruction::set_defaults()
{

}

void
General_Reconstruction::initialise_keymap()
{
    this->parser.add_start_key("General reconstruction");
    this->parser.add_stop_key("End General reconstruction");

    this->parser.add_parsing_key("reconstruction method", &this->reconstruction_method_sptr);
}

bool
General_Reconstruction::post_processing()
{
    return false;
}

Succeeded
General_Reconstruction::process_data()
{
    HighResWallClockTimer t;
    t.reset();
    t.start();

        //return reconstruction_object.reconstruct() == Succeeded::yes ?
        //    EXIT_SUCCESS : EXIT_FAILURE;
        if (reconstruction_method_sptr->reconstruct() == Succeeded::yes)
          {
            t.stop();
            std::cout << "Total Wall clock time: " << t.value() << " seconds" << std::endl;
            return Succeeded::yes;
          }
        else
          {
            t.stop();
            return Succeeded::no;
          }
}

END_NAMESPACE_STIR
