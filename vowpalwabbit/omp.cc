/*
Copyright (c) by respective owners including Yahoo!, Microsoft, and
individual contributors. All rights reserved.  Released under a BSD (revised)
license as described in the file LICENSE.
 */

#include <float.h>

#include "omp.h"
#include "parse_regressor.h"
#include "vw.h"

using namespace std;

namespace OMP {

  struct omp {
    bool gradient_phase;
    bool initialization_needed;
    bool example_based_phases;
    bool save_per_feature;
    bool multi_loss;
    bool feature_costs;

    size_t last_gradient_pass; // Pass of the last gradient accumulation
    size_t last_gradient_example; // Example # of the last gradient accumulation
    size_t last_model_example; // Example # of the last model training

    size_t num_model_passes; // Number of passes to train model for before adding another feature
    size_t num_model_examples; // Number of examples to train model for before adding another feature
    size_t num_gradient_examples; // Number of examples to use to estimate gradient

    size_t max_features; // Maximum number of features to select
    size_t features_selected; // Number of features selected so far

    vw* all;
  };

  void checkpoint_predictor(vw& all, string reg_name, size_t feature)
  {
    char* filename = new char[reg_name.length()+8];
    sprintf(filename,"%s.k.%lu",reg_name.c_str(),(long unsigned)feature);
    dump_regressor(all, string(filename), false);
    delete[] filename;
  }

  void start_gradient_phase(omp* o)
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
      cerr << " ** starting gradient phase." << endl;
    }

    if (o->save_per_feature) {
      checkpoint_predictor(*all, all->final_regressor_name, o->features_selected);
    }

    o->gradient_phase = true;
    o->last_model_example = all->sd->example_number;
  }

  uint32_t select_best_single_weight(vw* all, omp* o, float* max_gain) {
    uint32_t length = 1 << all->num_bits;
    uint32_t stride = all->reg.stride;

    *max_gain = -FLT_MAX;
    uint32_t max_index = 0;

    for (uint32_t i = 0; i < length; i++) {
      if (all->reg.weight_vector[i*stride + all->feature_mask_idx] != 1.) {
        float gain = fabs(all->reg.weight_vector[i*stride + all->gradient_acc_idx]);
        float cost = 1.;

        if (o->feature_costs) {
          float c = all->reg.weight_vector[i*stride + all->feature_cost_idx];

          if (c > 0.) {
            cost = c;
          }
        }

        gain /= cost;
        if (gain > *max_gain) {
          *max_gain = gain;
          max_index = i;
        }

        // Clear out data for training actual model
        all->reg.weight_vector[i*stride + all->gradient_acc_idx] = 0.f;
      }
    }

    return max_index;
  }

  uint32_t select_best_multi_weight(vw* all, omp* o, float* max_gain) {
    uint32_t length = 1 << all->num_bits;
    uint32_t stride = all->reg.stride;
    uint32_t wpp = all->wpp;

    *max_gain = -FLT_MAX;
    uint32_t max_index = 0;

    for (uint32_t i = 0; i < length; i += wpp) {
      for (uint32_t j = i; j < i + wpp; j++) {
        float gain = 0.0;
        float cost = 0.0;

        if (all->reg.weight_vector[j*stride + all->feature_mask_idx] != 1.) {
          gain += fabs(all->reg.weight_vector[j*stride + all->gradient_acc_idx]);

          if (o->feature_costs) {
            float c = all->reg.weight_vector[j*stride + all->feature_cost_idx];
            cost = max(cost, c);
          }

          // Clear out data for training actual model
          all->reg.weight_vector[j*stride + all->gradient_acc_idx] = 0.f;
        }

        if (cost <= 0.) {
          cost = 1.0;
        }

        gain /= cost;
        if (gain > *max_gain) {
          *max_gain = gain;
          max_index = i;
        }
      }
    }

    return max_index;
  }

  void finish_gradient_phase(omp* o)
  {
    vw* all = o->all;

    uint32_t stride = all->reg.stride;
    uint32_t max_index;
    float max_gain;

    if (!all->quiet) {
      cerr << " ** finishing gradient phase: ";
    }

    if (o->multi_loss) {
      max_index = select_best_multi_weight(all, o, &max_gain);
    }
    else {
      max_index = select_best_single_weight(all, o, &max_gain);
    }

    // Activate the best feature
    if (max_gain < 0) {
      if (!all->quiet) {
        cerr << "no feature selected, best candidate had negative gain." << endl;
      }
    } else {
      if (o->multi_loss) {
        uint32_t wpp = all->wpp;

        if (!all->quiet) {
          cerr << "adding feature "
               << (max_index & all->reg.weight_mask) / wpp
               << " [" << (max_index & all->reg.weight_mask)
               << "] to model." << endl;
        }

        // Turn on everything in the block
        for (uint32_t j = max_index; j < max_index + wpp; j++) {
          all->reg.weight_vector[((j*stride) & all->reg.weight_mask) + all->feature_mask_idx] = 1.f;
          all->reg.weight_vector[((j*stride) & all->reg.weight_mask) + all->gradient_acc_idx] = 0.f;
        }
      }
      else {
        if (!all->quiet) {
          cerr << "adding feature "
               << (max_index & all->reg.weight_mask)
               << " to model." << endl;
        }

        all->reg.weight_vector[((max_index*stride) & all->reg.weight_mask) + all->feature_mask_idx] = 1.f;
        all->reg.weight_vector[((max_index*stride) & all->reg.weight_mask) + all->gradient_acc_idx] = 0.f;
      }
    }

    o->gradient_phase = false;
    o->last_gradient_pass = all->current_pass;
    o->last_gradient_example = all->sd->example_number;

    o->features_selected++;
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
      start_gradient_phase(o);
      o->initialization_needed = false;
    }

    if (o->example_based_phases) {
      if (o->gradient_phase) {
        size_t gradient_examples = all->sd->example_number - o->last_model_example;
        if (gradient_examples >= o->num_gradient_examples) {
          finish_gradient_phase(o);
        }
      }
      else if (o->features_selected < o->max_features) {
        size_t model_examples = all->sd->example_number - o->last_gradient_example;
        if (model_examples >= o->num_model_examples) {
          start_gradient_phase(o);
        }
      }
    }

    if (o->gradient_phase) {
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

    if (!o->example_based_phases) {
      // Only ever do a single pass of gradient accumulation
      if (o->gradient_phase) {
        finish_gradient_phase(o);
      }
      else if (o->features_selected < o->max_features) {
        size_t model_passes = all->current_pass - o->last_gradient_pass;
        if (model_passes >= o->num_model_passes) {
          start_gradient_phase(o);
        }
      }
    }
  }

  void finish(void* d)
  {
    omp* o = (omp*)d;
    vw* all = o->all;

    if (o->save_per_feature) {
      checkpoint_predictor(*all, all->final_regressor_name, o->features_selected);
    }
  }

  learner* setup(vw& all, std::vector<std::string>&opts, po::variables_map& vm, po::variables_map& vm_file)
  {
    omp* data = (omp*)calloc(1, sizeof(omp));

    data->initialization_needed = true;
    data->gradient_phase = true;
    data->example_based_phases = false;
    data->save_per_feature = false;
    data->multi_loss = false;
    data->feature_costs = false;

    data->last_gradient_pass = 0;
    data->last_gradient_example = 0;
    data->last_model_example = 0;

    data->num_model_passes = (size_t)(-1);
    data->num_model_examples = (size_t)(-1);
    data->num_gradient_examples = (size_t)(-1);

    if (vm.count("omp_save_per_feature")) {
      data->save_per_feature = true;
    }

    if (vm.count("feature_costs")) {
      data->feature_costs = true;
    }

    if (vm.count("omp_num_model_examples") && vm.count("omp_num_gradient_examples")) {
      data->num_model_examples = vm["omp_num_model_examples"].as<size_t>();
      data->num_gradient_examples = vm["omp_num_gradient_examples"].as<size_t>();
      data->example_based_phases = true;
    }
    else {
      if (vm.count("omp_num_model_examples")) {
        cerr << "error: to use --omp_num_model_examples, --omp_num_gradient_examples must be specified as well." << endl;
        throw exception();
      }

      if (vm.count("omp_num_gradient_examples")) {
        cerr << "error: to use --omp_num_gradient_examples, --omp_num_model_examples must be specified as well." << endl;
        throw exception();
      }

      if (vm.count("omp_num_model_passes"))
        data->num_model_passes = vm["omp_num_model_passes"].as<size_t>();
      else
        data->num_model_passes = 1;
    }

    if (vm.count("omp_k")) {
      data->max_features = vm["omp_k"].as<size_t>();
    }
    else {
      data->max_features = (size_t)(-1);
    }
    data->features_selected = 0;

    if (vm.count("oaa") || vm.count("csoaa")) {
      data->multi_loss = true;
    }

    data->all = &all;

    learner* l = new learner(data, learn, all.l);
    l->set_end_pass(end_pass);

    return l;
  }
}
