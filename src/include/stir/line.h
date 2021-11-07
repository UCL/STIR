//
//
/*!

  \file
  \ingroup buildblock

  \brief declaration of class stir::Line

  \author Patrick Valente
  \author Kris Thielemans
  \author PARAPET project

*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2009, IRSL
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/
#ifndef __stir_LINE_H__
#define __stir_LINE_H__


#include "stir/common.h"

#include <string>
#include <vector>

START_NAMESPACE_STIR

/*!
  \brief A class used by the Interfile parser.
  \warning This will be removed soon. Do NOT USE.
*/
class Line : public std::string
{

public :
        Line() : std::string()
	{}

	Line& operator=(const char* ch);

	std::string get_keyword();
	int get_index();
	int get_param(std::vector<int>& v);
	// KT 29/10/98 new
	int get_param(std::vector<double>& v);
	int get_param(std::vector<std::string>& v);
	int get_param(std::string& s);
	int get_param(int& i);
	// KT 01/08/98 new
	int get_param(unsigned long& i);
	// KT 01/08/98 new
	int get_param(double& i);
};

END_NAMESPACE_STIR

#endif


