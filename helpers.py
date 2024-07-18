import torch
import torch.nn.functional as F
from z3 import *

from pathlib import Path

import plot
import random
import subprocess
import tempfile
import os
import sys
import re
import graphviz
import circuitsvis as cv
from toolz.curried import concat, take, compose
import networkx as nx
import community as community_louvain
from functools import partial
from math import ceil
from itertools import repeat
from sklearn.tree import _tree

import scipy
import subprocess
import tempfile
import os
import sys
import itertools
from pysat.solvers import Glucose3
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import dill
from itertools import product

device = "cuda" if torch.backends.cuda.is_built() else "cpu"
root = Path('./saved_runs')

#| export
def cross_entropy_high_precision(logits, labels):
    # Shapes: batch x vocab, batch
    # Cast logits to float64 because log_softmax has a float32 underflow on overly 
    # confident data and can only return multiples of 1.2e-7 (the smallest float x
    # such that 1+x is different from 1 in float32). This leads to loss spikes 
    # and dodgy gradients
    logprobs = F.log_softmax(logits.to(torch.float32), dim=-1)
    prediction_logprobs = torch.gather(logprobs, index=labels[:, None], dim=-1)
    loss = -torch.mean(prediction_logprobs)
    return loss

def get_acc(logits, labels):
    bool_vec = logits.argmax(1) == labels
    return bool_vec.sum()/len(bool_vec)

def is_acc(logits, labels):
    bool_vec = logits.argmax(1) == labels
    return bool_vec

parens = re.compile("\s*(\(\s*(?:x\d+|Not\(x\d+\)),?\s*,?(?:x\d+|Not\(x\d+\))\s*\))\s*")
parse_clause = re.compile("(\()\s*(x\d+|Not\(x\d+\)),?\s*,?(x\d+|Not\(x\d+\))\s*(\))")
match_not = re.compile("Not\((.*)\)")

def get_token_mappings(nvars):
    token_mappings = {f'x{i}': i for i in range(nvars)}
    token_mappings.update({f'Not(x{i})': nvars+i for i in range(nvars)})
    token_mappings.update({'(': 2*nvars, ')': 2*nvars + 1, ':': 2*nvars + 2, 's': 2*nvars + 3, 'u': 2*nvars + 4})
    token_mappings.update({'#': 2*nvars + 5}) # this is for padding, meaning DONT CARE
    return token_mappings

def decode_to_CNFs(input_idx, config):
    nvars = config.nvars
    token_mappings = get_token_mappings(nvars)
    rev_token_mappings = {val : key for key, val in token_mappings.items()}
    de_r = []
    for t in input_idx:
        de_r.append(rev_token_mappings[t])
    return " ".join(de_r)

# Returns list of form [[1, 2], [1, -2], [-1, 2], [1, -1]]
def decode_to_clause_list(input_idx, config):
    digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    cnf = decode_to_CNFs(input_idx, config)

    clauses = parens.findall(cnf)
    cls = []
    for i in range(len(clauses)):
        cl = []
        clause = clauses[i]
        j = 0
        while j < len(clause):
            c = clause[j]
            if c in ['(', ')', ':', 's', 'u']:
                j += 1
                continue
            elif c == 'x':
                if clause[j+2] in digits:
                    var_id = int(clause[j+1:j+3]) + 1
                    j += 3
                else:
                    var_id = int(clause[j+1:j+2]) + 1
                    j += 2
                cl.append(var_id)
            elif c == 'N':
                if clause[j+6] in digits:
                    var_id = int(clause[j+5:j+7]) + 1
                    j += 8
                else:
                    var_id = int(clause[j+5:j+6]) + 1
                    j += 7
                cl.append(-var_id)
            else:
                j += 1            
        
        cls.append(cl)

    return cls


def encode_to_toks(input_str, config, pad_with_tautologies=False , randomize_tautologies=False, shuffle=False):
    nvars = config.nvars
    token_mappings = get_token_mappings(nvars)
    
    clauses = parens.findall(input_str)
    
    clause_count = len(clauses)
    
    if clause_count < config.nclauses:
        if pad_with_tautologies:
            tautology_count = config.nclauses - clause_count
            
            if randomize_tautologies:
                tautology_vars = torch.randint(
                    high=config.nvars, 
                    size=(tautology_count,)
                )
            else:
                tautology_vars = take(tautology_count, itertools.cycle(range(nvars)))
            
            for i in tautology_vars:
                # pad with tautology
                clauses.append(f'(x{i} Not(x{i}))')
            
    if shuffle:
        permutation = torch.randperm(config.nclauses)
        clauses = [clauses[p] for p in permutation]
    
    formula = list(concat(
        map(lambda s: list(parse_clause.match(s).groups()), clauses)
    ))
    
    if len(input_str) >= 2 and input_str[-2] == ':':
        formula += [input_str[-2], input_str[-1]]
    else:
        formula += [':', '#']
    
    toks = []
    for t in formula:
        toks.append(token_mappings[t])

    req_len = config.n_ctx + 1
    if len(toks) != req_len:
        pad_len = req_len - len(toks)
        pad_list = [token_mappings['#']] * pad_len
        toks = pad_list + toks
    
    return toks

# Encode from list of form [[1, 2], [1, -2], [-1, 2], [1, -1]]
def encode_to_toks_from_list(formula, label, config, pad_with_tautologies=True, ensure_len_2=True):
    nvars = config.nvars
    token_mappings = get_token_mappings(nvars)

    tokens = [10]  # Start with '('
    
    if pad_with_tautologies:
        tautology_count = config.nclauses - len(formula)
        tautology_vars = torch.randint(
            high=config.nvars, 
            size=(tautology_count,)
        ) + 1
        
        for i, v in enumerate(tautology_vars):
            clause = [v, -v]
            clause_pos = torch.randint(high=len(formula) + 1, size=(1,))[0]
            formula.insert(clause_pos, clause)
        
    for i, clause in enumerate(formula):
        if len(clause) == 1 and ensure_len_2:
            clause = clause * 2
            
        if i > 0:
            tokens.append(11)  # Add ')' to separate clauses
            tokens.append(10)
        for j, literal in enumerate(clause):
            literal_str = f'x{abs(literal)-1}' if literal > 0 else f'Not(x{abs(literal)-1})'
            tokens.append(token_mappings[literal_str])
        #tokens.append(11)


    tokens.extend([11, 12])  # End with ')' and ':' to separate clauses from the label
    tokens.append(token_mappings[label])

    return tokens


def implication_graph(input, config, toks=True):
    input_str = decode_to_CNFs(input, config) if toks else input
    
    vis = graphviz.Digraph(comment="Implication Graph", strict=True)
    imp_graph = nx.DiGraph()

    for i in range(config.nvars):
        vis.node(f"x{i}", f"x{i}")
        vis.node(f"Not(x{i})", f"Not(x{i})")
        imp_graph.add_node(f"x{i}")
        imp_graph.add_node(f"Not(x{i})")

    clauses = parens.findall(input_str)
    clauses = map(lambda s: list(parse_clause.match(s).groups()), clauses)

    def parse_var(var):    
        if (match := match_not.match(var)) is not None:
            return (False, match[1])
        else:
            return (True, var)

    def gen_var(rep, flip=False):
        is_true, var = rep

        true_out = not is_true if flip else is_true

        if true_out:
            return var

        return f"Not({var})"

    for c in clauses:
        a = parse_var(c[1])
        b = parse_var(c[2])

        vis.edge(gen_var(a, flip=True), gen_var(b))
        vis.edge(gen_var(b, flip=True), gen_var(a))
        imp_graph.add_edge(gen_var(a, flip=True), gen_var(b))
        imp_graph.add_edge(gen_var(b, flip=True), gen_var(a))

    return (vis, imp_graph)

def get_graph_stats(imp_graph):
    avg_clustering = nx.average_clustering(imp_graph)
    imp_graph_undirected = imp_graph.to_undirected()
    imp_graph_part = community_louvain.best_partition(imp_graph_undirected)
    modularity = community_louvain.modularity(imp_graph_part, imp_graph_undirected)
    return (avg_clustering, modularity)

def print_graph_stats(graph_stats_acc_map):
    num_samples = np.shape(graph_stats_acc_map)[0]
    avg_clustering_acc_map = graph_stats_acc_map[:, [0,2]]
    modularity_acc_map = graph_stats_acc_map[:, [1,2]]
    sorted_avg_clustering_acc_map = avg_clustering_acc_map[np.argsort(avg_clustering_acc_map[:,0])]
    sorted_modularity_acc_map = modularity_acc_map[np.argsort(modularity_acc_map[:,0])]
    avg_clustering_cum_accuracy = np.cumsum(sorted_avg_clustering_acc_map[:,1]) / np.arange(1, num_samples + 1)
    modularity_cum_accuracy = np.cumsum(sorted_modularity_acc_map[:,1]) / np.arange(1, num_samples + 1)

    plt.plot(sorted_avg_clustering_acc_map[:,0],  avg_clustering_cum_accuracy, marker='o', linestyle='-', color='b', label='Cumulative Acc')
    plt.xlabel('Avg Clustering')
    plt.ylabel('Cumulative Accuracy')
    plt.title('Avg Clustering vs Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig("avg_clustering_vs_cum_accuracy.png")
    plt.close()

    plt.plot(sorted_modularity_acc_map[:,0],  modularity_cum_accuracy, marker='o', linestyle='-', color='b', label='Cumulative Acc')
    plt.xlabel('Modularity')
    plt.ylabel('Cumulative Accuracy')
    plt.title('Modularity vs Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig("modularity_vs_cum_accuracy.png")
    plt.close()

    num_buckets = 20

    # Calculate the number of elements per bucket
    elements_per_bucket = -(len(sorted_modularity_acc_map) // -num_buckets)

    # Initialize variables to store bucket data
    bucket_data = []
    bucket_modularities = []
    bucket_accuracies = []

    # Iterate through the sorted modularity-accuracy array
    for modularity, accuracy in sorted_modularity_acc_map:
        if len(bucket_modularities) < elements_per_bucket:
            bucket_modularities.append(modularity)
            bucket_accuracies.append(accuracy)
        else:
            bucket_data.append((bucket_modularities, bucket_accuracies))
            bucket_modularities = [modularity]
            bucket_accuracies = [accuracy]

    # Process the last bucket
    if bucket_modularities:
        bucket_data.append((bucket_modularities, bucket_accuracies))

    # Calculate mean accuracy for each bucket and construct the final array
    modularity_buckets_acc = np.array([(np.mean(modularities), np.mean(accuracies))
                                    for modularities, accuracies in bucket_data])


    plt.plot(modularity_buckets_acc[:,0],  modularity_buckets_acc[:,1], marker='o', linestyle='-', color='b', label='Acc')
    plt.xlabel('Modularity')
    plt.ylabel('Accuracy')
    plt.title('Modularity vs Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig("modularity_vs_accuracy.png")
    plt.close()

    # Calculate the number of elements per bucket
    elements_per_bucket = -(len(sorted_avg_clustering_acc_map) // -num_buckets)

    # Initialize variables to store bucket data
    bucket_data = []
    bucket_clusterings = []
    bucket_accuracies = []

    # Iterate through the sorted_avg_clustering_acc_map array
    for clustering, accuracy in sorted_avg_clustering_acc_map:
        if len(bucket_clusterings) < elements_per_bucket:
            bucket_clusterings.append(clustering)
            bucket_accuracies.append(accuracy)
        else:
            bucket_data.append((bucket_clusterings, bucket_accuracies))
            bucket_clusterings = [clustering]
            bucket_accuracies = [accuracy]

    # Process the last bucket
    if bucket_clusterings:
        bucket_data.append((bucket_clusterings, bucket_accuracies))

    # Calculate mean accuracy for each bucket and construct the final array
    clustering_buckets_acc = np.array([(np.mean(clusterings), np.mean(accuracies))
                                    for clusterings, accuracies in bucket_data])


    plt.plot(clustering_buckets_acc[:,0],  clustering_buckets_acc[:,1], marker='o', linestyle='-', color='b', label='Acc')
    plt.xlabel('Avg Clustering')
    plt.ylabel('Accuracy')
    plt.title('Avg Clustering vs Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig("avg_clustering_vs_accuracy.png")
    plt.close()


def attention_patterns_interactive(
        config=None,
        model=None,
        device=device,
        formula="", 
        layer=1, 
        spaces=4, 
        randomize_tautologies=False, 
        shuffle=False, 
        **kwargs
    ):
    toks = encode_to_toks(
        formula, 
        config,
        randomize_tautologies=randomize_tautologies, 
        shuffle=shuffle, 
        **kwargs
    )
    model_input = torch.LongTensor(toks[:-1]).unsqueeze(0).to(device)
    
    cache = {}
    model.remove_all_hooks()
    model.cache_all(cache)
    model_output = model(model_input)
    
    rev_token_mappings = {val: key for key, val in get_token_mappings(config.nvars).items()}
    
    tokens = [rev_token_mappings[t] + " " * spaces for t in toks][:-1]
    pattern = cache[f"blocks.{layer}.attn.hook_attn"][0]
    
    return cv.attention.attention_patterns(tokens=tokens, attention=pattern)


def generate_single_attention_map(config, cache, bid, **kwargs):
    attn_mat = cache[f'blocks.{bid}.attn.hook_attn']
    print(f'Attention Matrix (block {bid}):', attn_mat.shape)

    fig = plot.inputs_heatmap(
        attn_mat[0, :, :, :],
        title=f'Attention Score for all heads in block {bid}',
        return_fig=True,
        color_continuous_scale='Blues',
        facet_col=0,
        facet_labels=[f"Head {i}" for i in range(config.num_heads[bid])],
        zmin=0.,
        zmax=1.,
        **kwargs
    )
    fig.update_layout(coloraxis_colorbar_x=1.0,
                    coloraxis_colorbar_xanchor="left", xaxis_title="Context", yaxis_title="Current")
    fig.update_xaxes(tickmode = 'linear', tick0 = 0, dtick = 1, tickfont = dict(size=2))
    fig.update_layout(yaxis = dict(tickmode = 'linear', tick0 = 0, dtick = 1, tickfont = dict(size=5)))

    return fig


def generate_attention_map(config, cache, name):
    for bid in range(config.num_layers):
        fig = generate_single_attention_map(config, cache, bid)
        fig.write_image(f'att_score_block_{bid}_sample_{name}.svg')



def find_satisfying_assignment(clauses):
    solver = Solver()
    solver.add(And(*clauses))

    s_str = str(solver)

    label = solver.check()
    label = "s" if str(label) == 'sat' else "u"

    return label, s_str
    

# Find a satisfying assignment using Z3
def convert_UNSAT_to_SAT(unsat_formula, config):
    cls = decode_to_clause_list(unsat_formula, config)
    vars = [Bool(f"x{i}") for i in range(config.nvars)]
    cls_z3 = []
    for i in range(len(cls)):
        cl = []
        clause = cls[i]
        j = 0
        for j in range(len(clause)):
            c = clause[j]
            var_id = abs(c) - 1
            tmp = vars[var_id]
            if c > 0:
                cl.append(tmp)
            else:
                tmp = Not(tmp)
                cl.append(tmp)
        
        cls_z3.append(Or(*cl))

    label, solver_formula = find_satisfying_assignment(cls_z3)
    cls_z3.pop()
    while True:
        prev_label = label
        prev_solver_formula = solver_formula

        # Try to find a satisfying assignment
        label, solver_formula = find_satisfying_assignment(cls_z3)
        
        if label == "s":
            sat_formula = "".join(solver_formula.split("\n     "))[1:-1]    
            sat_formula = sat_formula[:-1]
            sat_formula = sat_formula.replace('And', '').replace('Or', '')[1:] + f':{label}'

            unsat_formula = "".join(prev_solver_formula.split("\n     "))[1:-1]    
            unsat_formula = unsat_formula[:-1]
            unsat_formula = unsat_formula.replace('And', '').replace('Or', '')[1:] + f':{prev_label}'

            return (encode_to_toks(sat_formula, config, pad_with_tautologies=False, shuffle=False), 
                    encode_to_toks(unsat_formula, config, pad_with_tautologies=False, shuffle=False))  # Formula is now satisfiable
        
        # If no satisfying assignment found, remove a clause from the formula
        if cls_z3:
            cls_z3.pop()
        else:
            return None  # Formula cannot be made satisfiable



#### SAT generation 
def dimacs(cnf, newlines=False):
    max_var = max(abs(lit) for clause in cnf for lit in clause)
    num_clauses = len(cnf)
    
    if not newlines:
        s = f'p cnf {max_var} {num_clauses} '
        for c in cnf:
            s = f'{s}{" ".join(map(str, c))} 0 '
    else:
        s = f'p cnf {max_var} {num_clauses}\n'
        for c in cnf:
            s = f'{s}{" ".join(map(str, c))} 0\n'
    return s


# from cnf formula, convert to dimacs and use DRAT to extract unsat core
def find_unsat_core(cnf_formula, verbose=False):
    dimacs_string = dimacs(cnf_formula, newlines=True)
    if verbose:
        print(dimacs_string)
    
    dimacs_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
    drat_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
    core_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
    core_file.close()
    
    dimacs_file.write(dimacs_string.strip() + '\n')
    dimacs_file.close()

    proc = subprocess.Popen(
        [  f'{os.getcwd()}/drat-trim',
            dimacs_file.name,
            drat_file.name,
            '-I',
            '-c',
            core_file.name
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stat = proc.wait()
    
    os.unlink(dimacs_file.name)
    os.unlink(drat_file.name)
    
    rval = None
    if stat == 0:
        with open(core_file.name, 'r') as f:
            lines = [line for line in f]
            if len(lines) > 1:  # Check if there are any lines after the first line
                unsat_core_lines = lines[1:]  # Extract unsat core lines
                rval = [[int(lit)] for line in unsat_core_lines for lit in line.strip().split()[:-1]]
            else:
                rval = None

    
    os.unlink(core_file.name)
    return rval

def is_satisfiable(cnf_formula):
    solver = Glucose3()
    
    for clause in cnf_formula:
        solver.add_clause(clause)
    
    return solver.solve()

# using unsat core, remove literals until sat assignment is found
# unsat_formula of the form [[1, 2], [1, -2], [-1, 2], [-1, -2], [1], [-1]]
def convert_UNSAT_to_SAT_Core(unsat_formula, unsat_core, config):
    # Generate all possible combinations of removing literals from the unsat core
    for i in range(len(unsat_core) + 1):
        for combination in itertools.combinations(unsat_core, i):
            modified_formula = [clause for clause in unsat_formula if clause not in combination]
            
            if is_satisfiable(modified_formula):
                return encode_to_toks_from_list(modified_formula, 's', config)
            
    return None  # No satisfying assignment found for any combination


def generate_random_formula(num_literals, num_clauses, min_literals_per_clause=1, max_literals_per_clause=3):
    formula = []
    
    for _ in range(num_clauses):
        num_literals_in_clause = random.randint(min_literals_per_clause, max_literals_per_clause)
        clause = []
        
        for _ in range(num_literals_in_clause):
            literal = random.choice(range(1, num_literals+1)) * random.choice([-1, 1])
            clause.append(literal)
        
        formula.append(clause)
    
    return formula

def dill_batch_runner(tup):
    f_pickled, xs_pickled = tup

    f, xs = dill.loads(f_pickled), dill.loads(xs_pickled)

    return [f(x) for x in xs]

def parmap(f, x, batch_size=1000, **kwargs):
    fn = partial(f, **kwargs)
    fn_pickled = dill.dumps(fn)
    
    batches = []
    
    if not hasattr(x, "__iter__"):
        x = [x]
        
    if hasattr(x, "__len__") and batch_size > 1:
        dataset_size = len(x)
        batch_count = ceil(dataset_size / batch_size)
            
        for i in range(batch_count):
            start = batch_size * i
            end = min(batch_size * (i+1), dataset_size)
    
            batches.append((fn_pickled, dill.dumps(x[start:end])))
    else:
        batches = [(fn_pickled, dill.dumps([xi])) for xi in x]
    
    with Pool() as p:
        out = p.map(dill_batch_runner, batches)

    return list(concat(out))
    
parmap_unbatched = partial(parmap, batch_size=1)

def generate_pair(config):
    num_literals = config.nvars
    num_clauses = config.nclauses
    
    unsat_formula = generate_random_formula(num_literals, num_clauses, 1, 2)
    
    # Find the UNSAT core
    unsat_core = find_unsat_core(unsat_formula, config.verbose)

    if unsat_core is not None:
        # Convert UNSAT to SAT
        sat_formula = convert_UNSAT_to_SAT_Core(unsat_formula, unsat_core, config)
        
        if sat_formula is not None:
            unsat_formula = encode_to_toks_from_list(unsat_formula, 'u', config)

            return unsat_formula, sat_formula
        else:
            return None
    else:
        return None
        bb
def generate_pairs(num_pairs, config):
    return list(filter(
        lambda x: x is not None, 
        parmap(generate_pair, itertools.repeat(config, num_pairs)),
    ))

def generate_dataset(num_formulas, config):
    num_pairs = num_formulas // 2

    formulas = []
    
    for u, s in generate_pairs(num_pairs, config):
        formulas += [u, s]

    return np.array(formulas)

# Tools for the axiomatic analysis
identity = lambda x: x

def compose_reversed(*args):
    fns = []
    for arg in args:
        if not hasattr(arg, "__iter__"):
            arg = [arg]
        
        fns = concat([fns, arg])

    rev_fns = reversed(list(fns))

    return compose(*rev_fns)

def batched(*fns, batch_size=5000):
    fn = compose_reversed(*fns)

    def f(inputs):
        outputs = []
        output_type = None
        is_tuple = False
        samples = len(inputs)
        n_batches = int(ceil(samples / batch_size))
    
        with torch.no_grad():
            for i in range(n_batches):
                start = batch_size * i
                end = min(batch_size * (i+1), samples)
                batch = inputs[start:end]
                output = fn(batch)

                if type(output) is tuple:
                    for i, o in enumerate(output):
                        if output_type is None:
                            outputs.append([o])
                        else:
                            outputs[i].append(o)
                        
                    if output_type is None:
                        is_tuple = True
                        output_type = [type(o) for o in output]

                    for i, o_type in enumerate(output_type):
                        if o_type is torch.Tensor:
                            outputs[i][-1] = outputs[i][-1].cpu()
                else:
                    outputs.append(output)
                    
                    if output_type is None:
                        output_type = type(output)

                    if output_type is torch.Tensor:
                        outputs[-1] = outputs[-1].cpu()

            if is_tuple:
                out_tuple = []
                for o, o_type in zip(outputs, output_type):
                     if o_type is np.ndarray:
                        out_tuple.append(np.concatenate(o, axis=0))
                     elif o_type is torch.Tensor:
                        out_tuple.append(torch.cat(o, dim=0))
                     else:
                        out_tuple.append(list(concat(o)))

                return tuple(out_tuple)
            
            if output_type is np.ndarray:
                return np.concatenate(outputs, axis=0)
            elif output_type is torch.Tensor:
                return torch.cat(outputs, dim=0)
            else:
                return list(concat(outputs))

    return f

def to_numpy(data):
    if isinstance(data, list):
        return np.array(data)
    elif isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()

    return data

def apply_or(inputs):
    return np.any(inputs, axis=-1)

# Prune decision tree
def prune(model):
    tree = model.tree_
    predictions = tree.value.argmax(axis=-1).flatten()

    def pruner(node=0):
        child_left = tree.children_left[node]
        child_right = tree.children_right[node]
        
        if child_left != _tree.TREE_LEAF:
            pruner(child_left)
            
        if child_right != _tree.TREE_LEAF:
            pruner(child_right)
            
        if child_left != _tree.TREE_LEAF and child_right != _tree.TREE_LEAF:
            prediction_left = predictions[child_left]
            prediction_right = predictions[child_right]

            if prediction_left == prediction_right:
                tree.children_left[node] = _tree.TREE_LEAF
                tree.children_right[node] = _tree.TREE_LEAF

    pruner()

    return model

def plot_confusion_matrix(
    cm, 
    ax=None, 
    cmap="viridis", 
    im_kw=None, 
    text_kw=None, 
    colorbar=True, 
    display_labels=None,
    xticks_rotation="horizontal",
):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    
    n_classes = cm.shape[0]
    count = cm.sum(axis=1)
    cm_normalized = cm / count[:, None]
    
    default_im_kw = dict(interpolation="nearest", cmap=cmap)
    im_kw = im_kw or {}
    im_kw = {**default_im_kw, **im_kw}
    text_kw = text_kw or {}
    
    im_ = ax.imshow(cm_normalized, **im_kw)
    cmap_min, cmap_max = im_.cmap(0), im_.cmap(1.0)
    
    text = np.empty_like(cm, dtype=object)

    # print text with appropriate color depending on background
    thresh = (cm_normalized.max() + cm_normalized.min()) / 2.0

    for i, j in product(range(n_classes), range(n_classes)):
        color = cmap_max if cm_normalized[i, j] < thresh else cmap_min
        text_cm = f"{cm_normalized[i,j]:.2g}\n({cm[i,j]:d}/{count[i]:d})"
        default_text_kwargs = dict(ha="center", va="center", color=color)
        text_kwargs = {**default_text_kwargs, **text_kw}

        text[i, j] = ax.text(j, i, text_cm, **text_kwargs)
    
    if display_labels is None:
        display_labels = np.arange(n_classes)
        
    if colorbar:
        fig.colorbar(im_, ax=ax)
    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=display_labels,
        yticklabels=display_labels,
        ylabel="True label",
        xlabel="Predicted label",
    )
    
    ax.set_ylim((n_classes - 0.5, -0.5))
    plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)

def get_high_eps_confidence_interval(matches, confidence_level=0.95):
    successes = matches.sum()
    trials = matches.shape[0]
    rate = successes/trials
    result = scipy.stats.binomtest(successes, trials, p=rate, alternative="greater")
    return 1 - result.proportion_ci(confidence_level, method="exact").low
    