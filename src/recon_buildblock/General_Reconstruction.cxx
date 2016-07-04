#include "stir/recon_buildblock/General_Reconstruction.h"

START_NAMESPACE_STIR

General_Reconstruction::
General_Reconstruction()
{

}

void
General_Reconstruction::ask_parameters()
{

}

void
General_Reconstruction::initialise_keymap()
{

    this->parser.add_parsing_key("reconstruction method", &this->reconstruction_method);
}

bool
General_Reconstruction::post_processing()
{

}

END_NAMESPACE_STIR
