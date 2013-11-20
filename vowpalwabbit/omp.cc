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
    bool gradient_pass;
    bool initialization_needed;

    vw* all;
  };

  void start_gradient_pass(omp* o)
  {
    vw* all = o->all;

    if (!all->quiet) {
      cerr << " ** starting gradient pass." << endl;
    }

    o->gradient_pass = true;
  }

  void finish_gradient_pass(omp* o)
  {
    vw* all = o->all;

    if (!all->quiet) {
      cerr << " ** finishing gradient pass." << endl;
    }

    o->gradient_pass = false;
  }

  void accumulate_gradient(omp* o, example* ec)
  {
    vw* all = o->all;
  }

  void learn(void* d, learner& base, example* ec)
  {
    omp* o = (omp*)d;
    vw* all = o->all;

    if (o->initialization_needed) {
      start_gradient_pass(o);
      o->initialization_needed = false;
    }

    if (o->gradient_pass) {
      accumulate_gradient(o, ec);
    }
    else {
      base.learn(ec);
    }
  }

  void end_pass(void* d)
  {
    omp* o = (omp*)d;
    vw* all = o->all;

    if (o->gradient_pass) {
      finish_gradient_pass(o);
    }
  }

  learner* setup(vw& all, std::vector<std::string>&opts, po::variables_map& vm, po::variables_map& vm_file)
  {
    omp* data = (omp*)calloc(1, sizeof(omp));

    data->initialization_needed = true;
    data->gradient_pass = true;

    data->all = &all;

    learner* l = new learner(data, learn, all.l);
    l->set_end_pass(end_pass);

    return l;
  }
}
