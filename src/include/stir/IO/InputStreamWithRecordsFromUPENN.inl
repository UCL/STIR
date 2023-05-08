/*
 *  Copyright (C) 2020-2022 University of Pennsylvania
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
START_NAMESPACE_STIR

std::vector<std::streampos>
InputStreamWithRecordsFromUPENN::
get_saved_get_positions() const
{
    return saved_get_positions;
}

void
InputStreamWithRecordsFromUPENN::
set_saved_get_positions(const std::vector<std::streampos>& poss)
{
    saved_get_positions = poss;
}

END_NAMESPACE_STIR
