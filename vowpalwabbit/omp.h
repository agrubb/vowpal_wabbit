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
  inline void vec_add_rescale_gradient(vw& all, void* p, float fx, uint32_t fi) {
    weight* w = &all.reg.weight_vector[fi & all.reg.weight_mask];
    float x_abs = fabs(fx);
    if( x_abs > w[all.normalized_idx] ) {// new scale discovered
      if( w[all.normalized_idx] > 0. ) {//If the normalizer is > 0 then rescale the gradient so it's as if the new scale was the old scale.
        float rescale = (w[all.normalized_idx]/x_abs);
        w[all.gradient_acc_idx] *= rescale;
      }
      w[all.normalized_idx] = x_abs;
    }
    *(float*)p += w[0] * fx;
  }

  learner* setup(vw& all, std::vector<std::string>&, po::variables_map& vm, po::variables_map& vm_file);
}

#endif
