/*
Copyright (c) by respective owners including Yahoo!, Microsoft, and
individual contributors. All rights reserved.  Released under a BSD (revised)
license as described in the file LICENSE.
 */
#include "omp.h"
#include "vw.h"

using namespace std;

namespace OMP {

  struct omp {
    vw* all;
  };

  void learn(void* d, learner& base, example* ec)
  {
    omp* o = (omp*)d;
    vw* all = o->all;

    base.learn(ec);
  }

  learner* setup(vw& all, std::vector<std::string>&opts, po::variables_map& vm, po::variables_map& vm_file)
  {
    omp* data = (omp*)calloc(1, sizeof(omp));

    data->all = &all;

    learner* l = new learner(data, learn, all.l);

    return l;
  }
}
