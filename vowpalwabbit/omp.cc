/*
Copyright (c) by respective owners including Yahoo!, Microsoft, and
individual contributors. All rights reserved.  Released under a BSD (revised)
license as described in the file LICENSE.
 */

#include <float.h>

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

    uint32_t length = 1 << all->num_bits;
    uint32_t stride = all->reg.stride;

    // Clear out any incidental data
    for (uint32_t i = 0; i < length; i++) {
      if (all->reg.weight_vector[i*stride + all->feature_mask_idx] != 1.) {
        all->reg.weight_vector[i*stride + all->gradient_acc_idx] = 0.f;
      }
    }

    if (!all->quiet) {
      cerr << " ** starting gradient pass." << endl;
    }

    o->gradient_pass = true;
  }

  uint32_t select_best_single_weight(vw* all, omp* o, float* max_gain) {
    uint32_t length = 1 << all->num_bits;
    uint32_t stride = all->reg.stride;

    *max_gain = -FLT_MAX;
    uint32_t max_index = 0;

    for (uint32_t i = 0; i < length; i++) {
      if (all->reg.weight_vector[i*stride + all->feature_mask_idx] != 1.) {
        float gain = fabs(all->reg.weight_vector[i*stride + all->gradient_acc_idx]);
        if (gain > *max_gain) {
          *max_gain = gain;
          max_index = i*stride;
        }

        // Clear out data for training actual model
        all->reg.weight_vector[i*stride + all->gradient_acc_idx] = 0.f;
      }
    }

    return max_index;
  }

  void finish_gradient_pass(omp* o)
  {
    vw* all = o->all;

    uint32_t length = 1 << all->num_bits;
    uint32_t stride = all->reg.stride;
    uint32_t max_index;
    float max_gain;

    if (!all->quiet) {
      cerr << " ** finishing gradient pass: ";
    }

    max_index = select_best_single_weight(all, o, &max_gain);

    // Activate the best feature
    if (max_gain < 0) {
      if (!all->quiet) {
        cerr << "no feature selected, best candidate had negative gain." << endl;
      }
    } else {
      if (!all->quiet) {
        cerr << "adding feature "
             << (max_index & all->reg.weight_mask) / (stride)
             << " to model." << endl;
      }

      all->reg.weight_vector[(max_index & all->reg.weight_mask) + all->feature_mask_idx] = 1.f;
      all->reg.weight_vector[(max_index & all->reg.weight_mask) + all->gradient_acc_idx] = 0.f;
    }

    o->gradient_pass = false;
  }

  float predict(omp* o, example* ec)
  {
    vw* all = o->all;

    label_data* ld = (label_data*)ec->ld;
    float prediction;

    if (all->training && all->normalized_updates && ld->label != FLT_MAX && ld->weight > 0) {
      prediction = GD::inline_predict<vec_add_rescale_gradient>(*all, ec);
    }
    else {
      prediction = GD::inline_predict<vec_add>(*all, ec);
    }
    ec->partial_prediction = prediction;

    all->set_minmax(all->sd, ld->label);
    ec->final_prediction = GD::finalize_prediction(*all, ec->partial_prediction * (float)all->sd->contraction);

    if (ld->label != FLT_MAX)
      ec->loss = all->loss->getLoss(all->sd, ec->final_prediction, ld->label) * ld->weight;

    float gradient = 0.f;
    if (ld->label != FLT_MAX && !ec->test_only)
      gradient = all->loss->first_derivative(all->sd, ec->final_prediction, ld->label) * ld->weight;

    ec->done = true;

    return gradient;
  }

  template <void (*T)(vw&, void*, float, uint32_t)>
  void omp_train(vw& all, example* &ec, float gradient)
  {
    if (fabs(gradient) == 0.)
      return;

    GD::foreach_feature<T>(all, ec, &gradient);
  }

  template<bool normalized>
  inline void omp_update(vw& all, void* dat, float x, uint32_t fi)
  {
    if(all.reg.weight_vector[(fi & all.reg.weight_mask)+all.feature_mask_idx]!=1.){
      weight* w = &all.reg.weight_vector[fi & all.reg.weight_mask];

      float gradient = *(float*)dat;
      if (normalized) {
        x /= w[all.normalized_idx];
      }

      w[all.gradient_acc_idx] += gradient * x;
    }
  }

  void accumulate_gradient(omp* o, example* ec)
  {
    vw* all = o->all;

    predict(o, ec);

    float gradient = predict(o, ec);

    if (all->normalized_updates)
      omp_train<omp_update<true> >(*all, ec, gradient);
    else
      omp_train<omp_update<false> >(*all, ec, gradient);
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
