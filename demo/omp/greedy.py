import itertools
import numpy as np
import os
import re
import shutil
import struct
import subprocess
import sys

PATH_TO_VW = '../../vowpalwabbit/vw'

class VWWrapper(object):
    def __init__(self, args, train_args, model_dir='', initial_model=None, verbose=False):
        self.path_to_vw = '../../vowpalwabbit/vw'
        self.vw_args = args
        self.vw_train_args = train_args

        # Set up the indexing in to the weights
        self.weights_per_problem = 1
        self.k = 1
        self.offset = 0

        if '--oaa' in args:
            self.k = int(args[args.index('--oaa') + 1])
            self.offset = 0

        if '--wap' in args:
            self.k = int(args[args.index('--wap') + 1])
            self.offset = 0

        if '--csoaa' in args:
            self.k = int(args[args.index('--csoaa') + 1])
            self.offset = 1

        while (self.k > self.weights_per_problem):
            self.weights_per_problem <<= 1

        self.default_weight = 1e-3

        self.verbose = verbose
        self.model_dir = model_dir
        self.get_prefix(initial_model)

    def feature_mask_to_vw_model(self, filename, mask):
        self.features_to_vw_model(filename, np.nonzero(mask)[0])

    def features_to_vw_model(self, filename, selected):
        f = open(filename, 'wb')
        f.write(self.prefix)

        for i in selected:
            for j in range(self.k):
                idx = i * self.weights_per_problem + j + self.offset
                f.write(struct.pack('<If', idx, self.default_weight))
        f.close()

    def costs_to_vw_model(self, filename, costs):
        f = open(filename, 'wb')
        f.write(self.prefix)

        for i,c in enumerate(costs):
            for j in range(self.k):
                idx = i * self.weights_per_problem + j + self.offset
                f.write(struct.pack('<If', idx, c))
        f.close()

    def vw_readable_model_to_feature_mask(self, filename, n_features):
        mask = np.zeros(n_features, dtype=np.bool)

        f = open(filename, 'r')
        lines = [l for l in f]
        f.close()

        i = lines.index(':0\n') + 1
        for l in lines[i:]:
            m = re.match('(\S+):(\S+)', l)
            if m:
                fi = int(m.group(1))
                fv = float(m.group(2))

                f = (fi - self.offset) // self.weights_per_problem
                if 0 <= f < mask.shape[0]:
                        mask[f] = 1

        return mask

    def vw_readable_model_to_features(self, filename, n_features):
        mask = self.vw_readable_model_to_feature_mask(filename, n_features)
        return np.nonzero(mask)[0]

    def execute_vw_and_parse(self, command):
        if self.verbose:
            print '#### EXECUTING:', ' '.join(command), '####'
            print '#### BEGIN VW RUN ####'

        p = subprocess.Popen(command, stderr=subprocess.PIPE)

        selected_features = []
        average_loss = 1.

        for line in p.stderr:
            m = re.match(' \*\* .* adding feature (\d+) \[\d+\] to model', line)
            if m:
                selected_features.append(int(m.group(1)))

            m = re.match('average loss = (\S+)', line)
            if m:
                average_loss = float(m.group(1))

            if self.verbose:
                print '  ', line,

        p.wait()
        if self.verbose:
            print '#### END VW RUN ####'

        return selected_features, average_loss

    def get_prefix(self, prefix_filename=None):
        if prefix_filename is None:
            prefix_filename = os.path.join(self.model_dir, 'prefix.model')

            command = [self.path_to_vw]
            command += self.vw_args
            command += ('-d /dev/null -f {0}'.format(prefix_filename)).split()

            _ = self.execute_vw_and_parse(command)

        f = open(prefix_filename, 'rb')
        self.prefix = f.read()
        f.close()

    def run_train(self, train_set, extra_args=None, input_model=None, input_mask=None,
                  input_costs=None, output_model=None, output_readable_model=None):
        command = [self.path_to_vw]
        command += self.vw_args
        command += self.vw_train_args
        if extra_args:
            command += extra_args
        command += ['-c', '-d', train_set]
        if input_model:
            command += ['-i',input_model]
        if input_mask:
            command += ['--feature_mask',input_mask]
        if input_costs:
            command += ['--feature_costs',input_costs]
        if output_model:
            command += ['-f',output_model]
        if output_readable_model:
            command += ['--readable_model',output_readable_model]

        return self.execute_vw_and_parse(command)

    def run_test(self, test_set, input_model=None):
        command = [self.path_to_vw]
        command += self.vw_args
        command += ['-t', '-c', '-d', test_set]
        if input_model:
            command += ['-i',input_model]

        return self.execute_vw_and_parse(command)


def recompute_costs(results, costs):
    n_features = costs.shape[0]
    new_results = results.copy()

    for i, idx in enumerate(new_results['idx']):
        mask = index_to_feature_mask(idx, n_features)
        new_results[i]['cost'] = mask.dot(costs)

    return new_results

def get_best_results(results):
    S = np.argsort(results['cost'])

    last = S[0]
    sequence = []

    losses = results['train_loss']
    costs = results['cost']

    for i in range(1, S.shape[0]):
        s = S[i]
        if losses[s] < losses[last]:
            if costs[s] > costs[last]:
                sequence.append(last)
            last = s

    sequence.append(last)
    return sequence

def run_optimal(train_set, test_set, vw, model_dir, features,
                initial_model=None, K=None, costs=None):
    n_features = len(features)

    if K is None:
        K = n_features

    if costs is None:
        costs = np.ones(n_features)

    # for keeping track of losses, etc.
    stats = []

    for k in range(K):
        print ('Training for all sets of {0} features'.format(k + 1))

        for s in itertools.combinations(features, n+1):
            selected = np.array(s, dtype=np.int)

            mask_filename = ('{0}/in.mask.model'.format(model_dir))
            model_filename = ('{0}/out.model'.format(model_dir))

            vw.features_to_vw_model(mask_filename, selected)
            _, train_loss = vw.run_train(train_set,
                                         input_model=initial_model,
                                         input_mask=mask_filename,
                                         output_model=model_filename)

            _, test_loss = vw.run_test(test_set, input_model=model_filename)

            c = np.sum(costs[selected])

            sequence.append((v, c, selected, w))

            stats.append((cost, train_loss, test_loss, selected))

    return np.asarray(stats, dtype=[('cost', np.float), ('train_loss', np.float), ('test_loss', np.float),
                                    ('selected', object)])

def run_forward(train_set, test_set, vw, model_dir, features,
                initial_model=None, K=None, costs=None):
    n_features = len(features)

    if K is None:
        K = n_features

    if costs is None:
        costs = np.ones(n_features)

    # for keeping track of losses, etc.
    stats = []

    # the feature mask for currently selected features
    selected = np.zeros(K, np.int)
    last_loss = 0.0

    cost = 0.0

    # Figure out starting loss
    mask_filename = ('{0}/empty.mask.model'.format(model_dir))
    model_filename = ('{0}/empty.model'.format(model_dir))

    vw.features_to_vw_model(mask_filename, selected[:0])
    _, previous_loss = vw.run_train(train_set,
                                    input_model=initial_model,
                                    input_mask=mask_filename,
                                    output_model=model_filename)

    last_model_filename = model_filename

    for k in range(K):
        print ('Forward regression selecting feature {0}'.format(k+1))
        best_gain = -sys.float_info.max
        best_feature = -1
        best_set = np.array([], dtype=np.int)

        print ('loss is now {0}'.format(previous_loss))

        sel = selected[:k+1]
        for f in features:
            # If feature is already selected just skip it
            if f in selected[:k]:
                continue

            mask_filename = ('{0}/{1}.{2}.mask.model'.format(model_dir, k+1, f))
            model_filename = ('{0}/{1}.{2}.model'.format(model_dir, k+1, f))

            sel[k] = f
            vw.features_to_vw_model(mask_filename, sel)
            _, train_loss = vw.run_train(train_set,
                                         input_model=initial_model,
                                         input_mask=mask_filename,
                                         output_model=model_filename)

            _, test_loss = vw.run_test(test_set, input_model=model_filename)

            gain = (previous_loss - train_loss) / costs[f]
            print ('feature {0}, loss {1}, cost {2}, gain {3}'.format(f, train_loss, costs[f], gain))

            if gain > best_gain:
                best_gain = gain
                best_feature = f
                best_stats = (best_feature, cost + costs[f], train_loss, test_loss, sel)
                best_loss = train_loss

        # No more best feature
        if best_feature == -1:
            print ('Exited early due to negative gain')
            break

        print ('Selected feature {0}'.format(best_feature))
        selected[k] = best_feature

        # Deal with files
        tmp_model_filename = ('{0}/{1}.{2}.model'.format(model_dir, k+1, best_feature))
        last_model_filename = ('{0}/{1}.model'.format(model_dir, k+1))
        shutil.copyfile(tmp_model_filename, last_model_filename)

        cost += costs[best_feature]
        previous_loss = best_loss
        stats.append(best_stats)

    return np.asarray(stats, dtype=[('feature', int), ('cost', float),
                                    ('train_loss', float), ('test_loss', float),
                                    ('selected', object)])

def run_omp(train_set, test_set, vw, model_dir, features,
            initial_model=None, K=None, costs=None):
    n_features = len(features)

    if K is None:
        K = n_features

    if costs is None:
        costs = np.ones(n_features)
        costs_filename = None
    else:
        costs_filename = ('{0}/costs.model'.format(model_dir))
        vw.costs_to_vw_model(costs_filename, costs)

    # for keeping track of losses, etc.
    stats = []

    # the feature mask for currently selected features
    selected = np.zeros(n_features, np.int)
    cost = 0.0

    # Double check training args
    if not '--omp_save_per_feature' in vw.vw_train_args:
        extra_args = ['--omp_save_per_feature']
    else:
        extra_args = []

    print 'Running OMP pass'
    model_filename = ('{0}/omp.model'.format(model_dir))

    selected_features, _ = vw.run_train(train_set, extra_args=extra_args,
                                        input_model=initial_model,
                                        input_costs=costs_filename,
                                        output_model=model_filename)

    last_model = ('{0}/omp.model.k.{1}'.format(model_dir, K))
    shutil.copyfile(model_filename, last_model)

    for i, f in enumerate(selected_features):
        model_filename = ('{0}/omp.model.k.{1}'.format(model_dir, i+1))
        _, train_loss = vw.run_test(train_set, input_model=model_filename)
        _, test_loss = vw.run_test(test_set, input_model=model_filename)

        if f < len(costs):
            cost += costs[f]
        selected[i] = f

        stats.append((f, cost, train_loss, test_loss, selected[:i+1]))
        print ('Selected feature {0}, loss {1}, total cost {2}'.format(f, train_loss, cost))

    return np.asarray(stats, dtype=[('feature', np.int), ('cost', np.float),
                                    ('train_loss', np.float), ('test_loss', np.float),
                                    ('selected', object)])

def run_omp_iterative(train_set, test_set, vw, model_dir, features,
                      initial_model=None, K=None, costs=None):
    n_features = len(features)

    if K is None:
        K = n_features

    if costs is None:
        costs = np.ones(n_features)

    # for keeping track of losses, etc.
    stats = []

    # the feature mask for currently selected features
    selected = np.zeros(n_features, np.int)
    cost = 0.0

    # Figure out starting loss
    previous_model_filename = initial_model

    for k in range(K):
        print ('OMP selecting feature {0}'.format(k+1))

        model_filename = ('{0}/{1}.model'.format(model_dir, k))

        selected_features, _ = vw.run_train(train_set,
                                            input_model=previous_model_filename,
                                            output_model=model_filename)
        _, train_loss = vw.run_test(train_set, input_model=model_filename)
        _, test_loss = vw.run_test(test_set, input_model=model_filename)

        previous_model_filename = model_filename

        if len(selected_features) != 1:
            print ('Exited early in OMP loop, single feature was not selected')
            break

        f = selected_features[0]
        selected[k] = f

        if f < len(costs):
            cost += costs[f]

        stats.append((f, cost, train_loss, test_loss, selected[:k+1]))
        print ('Selected feature {0}, loss {1}, total cost {2}'.format(f, train_loss, cost))

    return np.asarray(stats, dtype=[('feature', np.int), ('cost', np.float),
                                    ('train_loss', np.float), ('test_loss', np.float),
                                    ('selected', object)])

def lasso_check(selected, vw, train_set, n_features, l1):
    model_filename = ('{0}/lasso.check.model.txt'.format(vw.model_dir))

    if l1 < 0:
        l1 = 0.0

    extra_args = ['--l1', str(l1)]

    _ = vw.run_train(train_set, extra_args=extra_args,
                     output_readable_model=model_filename)

    new_selected = vw.vw_readable_model_to_features(model_filename, n_features)

    return not np.array_equal(new_selected, selected)

def lasso_select(selected, vw, train_set, n_features, last):
    lasso_tolerance = 1e-6

    l1 = last / 2.0

    while not lasso_check(selected, vw, train_set, n_features, l1):
        l1 /= 2.0

    hi = last
    lo = l1
    l1 = (hi + lo) / 2.0
    while (hi - lo) > lasso_tolerance:
        down = lasso_check(selected, vw, train_set, n_features, l1)

        if down:
            up = lasso_check(selected, vw, train_set, n_features, l1 + lasso_tolerance)
            if not up:
                break
            else:
                lo = l1
                l1 = (hi + lo) / 2.0
        else:
            hi = l1
            l1 = (hi + lo) / 2.0

    lasso_model = ('{0}/lasso.model'.format(vw.model_dir))
    lasso_readable_model = ('{0}/lasso.model.txt'.format(vw.model_dir))
    extra_args = ['--l1', str(l1)]

    _, lasso_loss = vw.run_train(train_set, extra_args=extra_args,
                                 output_model=lasso_model,
                                 output_readable_model=lasso_readable_model)

    lasso_selected = vw.vw_readable_model_to_features(lasso_readable_model, n_features)
    return lasso_selected, l1, lasso_loss, lasso_model

def run_lasso(train_set, test_set, vw, model_dir, features,
              initial_model=None, K=None, costs=None,
              initial_l1=1.0, scaled_set=None):
    n_features = len(features)

    if K is None:
        K = n_features

    if costs is None:
        costs = np.ones(n_features)

    if scaled_set is None:
        scaled_set = train_set

    # for keeping track of losses, etc.
    stats = []

    # the feature mask for currently selected features
    selected = np.zeros(0, np.int)
    cost = 0.0

    last_l1 = initial_l1

    k = 0
    while selected.shape[0] < K:
        k += 1

        # Select feature with biggest inner product
        selected, last_l1, lasso_loss, tmp_lasso_model = lasso_select(selected, vw, scaled_set, n_features, last_l1)

        # Deal with files
        lasso_model = ('{0}/lasso.{1}.model'.format(model_dir, k))
        shutil.copyfile(tmp_lasso_model, lasso_model)

        mask_filename = ('{0}/{1}.mask.model'.format(model_dir, k))
        model_filename = ('{0}/{1}.model'.format(model_dir, k))
        vw.features_to_vw_model(mask_filename, selected)

        _, train_loss = vw.run_train(train_set,
                                     input_model=initial_model,
                                     input_mask=mask_filename,
                                     output_model=model_filename)

        _, test_loss = vw.run_test(test_set,
                                   input_model=model_filename)

        cost = np.sum(costs[selected])

        stats.append((last_l1, cost, train_loss, test_loss, lasso_loss, selected))
        print ('L1 {0}, loss {1}, total cost {2}'.format(last_l1, train_loss, cost))

    return np.asarray(stats, dtype=[('l1', np.float), ('cost', np.float),
                                    ('train_loss', np.float), ('test_loss', np.float),
                                    ('lasso_loss', np.float), ('selected', object)])
