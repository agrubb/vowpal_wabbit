/*
Copyright (c) by respective owners including Yahoo!, Microsoft, and
individual contributors. All rights reserved.  Released under a BSD
license as described in the file LICENSE.
 */
#ifndef OMP_H
#define OMP_H

#include "example.h"
#include "global_data.h"
#include "learner.h"
#include "parse_args.h"

namespace OMP
{
  learner* setup(vw& all, std::vector<std::string>&, po::variables_map& vm, po::variables_map& vm_file);
}

#endif
