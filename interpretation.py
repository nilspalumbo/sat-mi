import torch
from helpers import *
from toolz.curried import *
from model import *
import altair as alt
import pandas as pd
from types import SimpleNamespace
from math import ceil
from matplotlib.colors import LinearSegmentedColormap
from numba import njit
import numpy.typing as npt
import sklearn
import dill as pickle
from sklearn.tree import _tree
import graphviz

def load_model(config=None, verbose=False):
    model_dict = torch.load(config.load_model_path, map_location=device)
    if verbose:
        print("Train loss: ", model_dict['train_loss'])
        print("Test loss: ", model_dict['test_loss'])
        print("Epoch: ", model_dict['epoch'])
        
    model = Transformer(config, use_cache=False)
    model.to(device)
    model.load_state_dict(model_dict['model'])
    
    if config.data_parallel:
        model = nn.DataParallel(model, device_ids=config.device_ids)
        
    return model

def tools(config, model=None, node_models=None):
    token_mappings = get_token_mappings(config.nvars)
    rev_token_mappings = {val : key for key, val in token_mappings.items()} 
    prediction_tokens = [token_mappings['s'], token_mappings['u']]

    if model is None:
        model = load_model(config)

    train, test = gen_fixedlen_train_test(config = config, interpretation=True)
    train_idx = np.arange(train.shape[0], dtype=np.int_)
    test_idx = np.arange(test.shape[0], dtype=np.int_) + train.shape[0]
    all_data = np.vstack((train, test))
    
    default_tok_probs = {}

    for i in range(config.nvars):
        default_tok_probs[f"x{i}"] = 1/(2 * config.nvars)
        default_tok_probs[f"Not(x{i})"] = 1/(2 * config.nvars)

    phi = "\U0001D719"

    truth_table_inputs = np.unpackbits(np.arange(2**config.nvars, dtype=np.uint8)[:,np.newaxis], axis=1).astype(np.bool_)[:,-config.nvars:]
    assignment_repr = lambda assignment: "".join(map(lambda x: "T" if x else "F", assignment))
    feature_names_truth_tables = list(map(lambda arr: f'{phi}[{assignment_repr(arr)}]', truth_table_inputs))
    feature_names_truth_tables_rev_map = {name: i for i, name in enumerate(feature_names_truth_tables)}
    
    variables = list(concat(
        [[f"x{i}", f"Not(x{i})"] for i in range(config.nvars)]
    ))
    variable_token_ids = [token_mappings[v] for v in variables]
    clause_map = [(l,r) for l in variable_token_ids for r in variable_token_ids]
    rev_clause_map = {v: i for i, v in enumerate(clause_map)}

    # Stores tools specialized to the model and dataset, tools may depend on earlier steps of the sequence
    tool_dict = {
        "model": model,
        "config": config,
        "tok_probs": default_tok_probs,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "all_data": all_data,
        "train": train,
        "test": test,
        "token_mappings": token_mappings,
        "rev_token_mappings": rev_token_mappings,
        "prediction_tokens": prediction_tokens,
        "truth_table_inputs": truth_table_inputs,
        "feature_names_truth_tables": feature_names_truth_tables,
        "feature_names_truth_tables_rev_map": feature_names_truth_tables_rev_map,
        "variables": variables,
        "variable_token_ids": variable_token_ids,
        "clause_map": clause_map,
        "rev_clause_map": rev_clause_map,
    }
    
    with_config = lambda f: partial(f, **tool_dict)

    tool_dict |= {
        "attention_patterns": with_config(attention_patterns_interactive),
        "ov_out": with_config(OV_out),
        "layer_1_clause": with_config(layer_1_clause),
        "output_token_query": with_config(output_token_query),
        "clause_effect_df": with_config(clause_effect_df),            
        "intermediates": with_config(intermediates),
        "first_layer_attn_circuits": with_config(first_layer_attn_circuits),
        "render_graphviz": with_config(render_graphviz),
    } | with_config(get_truth_tables_fn)()
    
    # Require "intermediates" to be defined
    tool_dict |= {
        "get_all_intermediates": with_config(get_all_intermediates),
        "get_mean_intermediates": with_config(get_mean_intermediates),
    } 

    # Generates or loads the cache, containing truth tables, node models, etc. for a model, dataset pair
    tool_dict["generate_cache"] = with_config(generate_cache)
    
    os.makedirs(config.base_cache_path, exist_ok=True)
    if not os.path.exists(config.cache_path):
        cached = with_config(generate_cache)()
        with open(config.cache_path, "wb") as f:
            pickle.dump(cached, f)

    with open(config.cache_path, "rb") as f:
        cached = pickle.load(f)
        
    tool_dict |= cached

    if node_models is not None:
        tool_dict["node_models"] = node_models

    for direction, tool in [
            ((True, True), get_layer_1_score_fns),
            ((True, "colon_embed"), output_token_embedding),
            ("classify", classify),
            ("vis_sat", vis_sat),
            ("plot_attentions", plot_attentions),
            ("plot_average_test_attentions", plot_average_test_attentions),
            ("plot_extreme_first_layer_scores", plot_extreme_first_layer_scores),
            ("get_avg_effect_df", get_avg_effect_df),
            ("get_colon_token_first_layer_scores", get_colon_token_first_layer_scores),
            ("mean_pos_layer_1_clause", mean_pos_layer_1_clause,),
            ((True, "layer_1_clause_map"), get_layer_1_clause_map, ),
            ((True, "layer_1_clauses"), get_layer_1_clauses),
            ("hook_diffs", hook_diffs),
            ("output_diffs", output_diffs),
            ((True, "alphas"), alphas),
            ((True, "gammas"), gammas),
            ((True, True), get_concrete_model_and_layers),
            ((True, True), get_abstract_model_and_layers),
            ((True, True), get_axiom_models),
            ((True, True), get_axiom_evaluators),
    ]:
        if isinstance(direction, tuple):
            execute, name = direction
        else:
            name = direction
            execute = False
            
        wc_tool = with_config(tool)

        if execute:
            if name is True:
                tool_dict |= wc_tool()
            else:
                tool_dict[name] = wc_tool()
        else:
            tool_dict[name] = wc_tool

    return SimpleNamespace(**tool_dict)
        

def output_token_query(*args, model=None, config=None, **kwargs):
    colon_embed=output_token_embedding(*args, model=model, config=config, **kwargs)
    W_Q = model.blocks[1].attn.W_Q
    
    return torch.einsum("d,ihd->ih", colon_embed, W_Q)


def OV_out(input_embed, model=None, heads=True, **kwargs):
    attn = model.blocks[1].attn
    W_O = attn.W_O
    W_V = attn.W_V
    
    W_O_heads = einops.rearrange(W_O, 'd (i h) -> i d h', i=4)
    result = torch.einsum('ihd, d-> ih', W_V, input_embed)
    OV_heads = torch.einsum('ih, idh -> id', result, W_O_heads)
    
    if not heads:
        OV_heads = OV_heads.sum(dim=0)
    
    return OV_heads


def direct_output_contributions(input_embed, model=None, config=None, **kwargs):
    W_U = model.unembed.W_U
    token_mappings = get_token_mappings(config.nvars)
    prediction_tokens = [token_mappings['u'], token_mappings['s']]

    ov_out = OV_out(input_embed, model=model)
    
    output_contributions = torch.einsum('id, du', ov_out, W_U)
    
    relevant_output_contributions = output_contributions[:, prediction_tokens]
    
    return relevant_output_contributions


def MLP_hidden_contributions(input_embed, model, **kwargs):
    W_in = model.blocks[1].mlp.W_in
    ov_out = OV_out(input_embed, model=model)
    
    return torch.einsum('id, md -> im', ov_out, W_in)


def clause_effect_df(
        model=None,
        config=None,
        tok_probs=None,
        prev_clause=None,
        **kwargs
):
    records = []

    with_config = lambda f, *args, **kwargs: f(*args, model=model, config=config, **kwargs)
    queries = with_config(output_token_query, tok_probs=tok_probs)

    variables = list(concat([f"x{i}", f"Not(x{i})"] for i in range(config.nvars)))

    attn = model.blocks[1].attn
    n_heads = attn.W_Q.shape[0]
    W_K = attn.W_K

    for l in variables:
        for r in variables:
            for clause_id in range(config.nclauses):
                out_dict = {
                    "left": l,
                    "right": r,
                    "clause_id": clause_id,
                }
                
                if prev_clause is not None:
                    embed = with_config(layer_1_clause, l, r, ll=prev_clause[0], lr=prev_clause[1], clause_id=clause_id)
                else:
                    embed = with_config(layer_1_clause, l, r, clause_id=clause_id)
                    
                scores = torch.einsum("d, ihd, ih->i", embed, W_K, queries)

                for h in range(n_heads):
                    out_dict[f"score_head_{h}"] = scores[h].detach().cpu().item()


                logit_contributions = direct_output_contributions(embed, config=config, model=model)

                sat_score = logit_contributions[:,1]
                diff_from_negative = logit_contributions[:,1] + logit_contributions[:,0]

                for h in range(n_heads):
                    out_dict[f"sat_score_head_{h}"] = sat_score[h].detach().cpu().item()
                    out_dict[f"sat_unsat_diff_from_negative_{h}"] = diff_from_negative[h].detach().cpu().item()

                hidden_contributions = MLP_hidden_contributions(embed, model=model)

                for h in range(n_heads):
                    out_dict[f"MLP_hidden_contr_{h}"] = hidden_contributions[h].detach().cpu()


                records.append(out_dict)

    return pd.DataFrame.from_records(records)


def plot_attentions(
        model=None,
        config=None,
        tok_probs=None,
        prev_clause=None,
        show_positional_effects=False,
        **kwargs
):
    n_heads = model.blocks[1].attn.W_Q.shape[0]

    df = clause_effect_df(model=model, config=config, tok_probs=tok_probs, prev_clause=prev_clause)
    
    return alt.Chart(df).mark_rect(width=20, height=20).encode(
        alt.X("left:N").title("First token of clause"),
        alt.Y("right:N").title("Second token of clause"),
        alt.Color("value:Q").title("Attention score"),
        alt.Column("key:N"),
        **({"row": "clause_id:Q"} if show_positional_effects else {})
    ).properties(
        width=300,
        height=300
    ).transform_fold(
        [f"score_head_{i}" for i in range(n_heads)]
    )


def layer_1_clause(
        l,
        r,
        ll=None,
        lr=None,
        parens=True,
        toks=False,
        clause_id=0,
        model=None,
        config=None,
        **kwargs
):
    clause_start = clause_id * 4
    l_pos = clause_start + 1
    r_pos = clause_start + 2

    W_E = model.embed.W_E
    W_pos = model.pos_embed.W_pos

    token_mappings = get_token_mappings(config.nvars)
    attn_circuits = first_layer_attn_circuits(model)
    OV = attn_circuits.OV
    QK = attn_circuits.QK
    
    use_prev_clause = ll is not None or lr is not None
    
    if use_prev_clause:
        ll_pos = clause_start - 3
        lr_pos = clause_start - 2
    
    if not toks:
        l = token_mappings[l]
        r = token_mappings[r]
        
        if use_prev_clause:
            lr = token_mappings[lr]
            ll = token_mappings[ll]
        
    if use_prev_clause:
        attn_score_ll = QK["dst_pos_src_pos"][r_pos, ll_pos] + QK["dst_pos_src_tok"][r_pos, ll] \
            + QK["dst_tok_src_pos"][r, ll_pos] + QK["dst_tok_src_tok"][r,ll]
        attn_score_lr = QK["dst_pos_src_pos"][r_pos, lr_pos] + QK["dst_pos_src_tok"][r_pos, lr] \
            + QK["dst_tok_src_pos"][r, lr_pos] + QK["dst_tok_src_tok"][r,lr]

    attn_score_op = QK["dst_pos_src_pos"][r_pos, r_pos-2] + QK["dst_pos_src_tok"][r_pos, token_mappings['(']] \
        + QK["dst_tok_src_pos"][r, r_pos-2] + QK["dst_tok_src_tok"][r, token_mappings['(']]
    
    if r_pos > 2:
        attn_score_cp = QK["dst_pos_src_pos"][r_pos, r_pos-3] + QK["dst_pos_src_tok"][r_pos, token_mappings[')']] \
        + QK["dst_tok_src_pos"][r, r_pos-3] + QK["dst_tok_src_tok"][r, token_mappings[')']]
    
    attn_score_l = QK["dst_pos_src_pos"][r_pos, l_pos] + QK["dst_pos_src_tok"][r_pos, l] \
        + QK["dst_tok_src_pos"][r, l_pos] + QK["dst_tok_src_tok"][r,l]
    attn_score_r = QK["dst_pos_src_pos"][r_pos, r_pos] + QK["dst_pos_src_tok"][r_pos, r] \
        + QK["dst_tok_src_pos"][r, r_pos] + QK["dst_tok_src_tok"][r,r]
    
    if use_prev_clause:
        scores = [attn_score_ll, attn_score_lr]
    else:
        scores = []
        
    if parens:
        if r_pos > 2:
            scores += [attn_score_cp, attn_score_op]
        else:
            scores.append(attn_score_op)
    
    scores += [attn_score_l, attn_score_r]
    scores = torch.Tensor(scores)
    
    scores /= np.sqrt(model.blocks[0].attn.d_head)
    attn_probs = F.softmax(scores, dim=0)
    
    embed_into_attn_r = W_E[:,r] + W_pos[r_pos]
    attn_out_l = OV["pos"][l_pos] + OV["tok"][l]
    attn_out_r = OV["pos"][r_pos] + OV["tok"][r]
    
    if use_prev_clause:
        attn_out_ll = OV["pos"][ll_pos] + OV["tok"][ll]
        attn_out_lr = OV["pos"][lr_pos] + OV["tok"][lr]
        
    if parens:
        if r_pos > 2:
            attn_out_cp = OV["pos"][r_pos-3] + OV["tok"][token_mappings[')']]
        
        attn_out_op = OV["pos"][r_pos-2] + OV["tok"][token_mappings['(']]

    attn_out = attn_probs[-2] * attn_out_l + attn_probs[-1] * attn_out_r
    
    if use_prev_clause:
        attn_out += attn_probs[0] * attn_out_ll + attn_probs[1] * attn_out_lr
        
    if parens:
        if r_pos > 2:
            attn_out += attn_probs[-4] * attn_out_cp
            
        attn_out += attn_probs[-3] * attn_out_op        
        
    embed_to_mlp = embed_into_attn_r + attn_out
    
    mlp_out = model.blocks[0].mlp(embed_to_mlp.unsqueeze(0).unsqueeze(0).to(config.device)).squeeze(0).squeeze(0)
    
    return embed_to_mlp + mlp_out


def output_token_embedding(model=None, config=None, tok_probs=None, toks=False, **kwargs):
    """Approximate embedding from the first layer of the ':' token where the distribution of variables is given by tok_probs."""
    W_E = model.embed.W_E
    W_pos = model.pos_embed.W_pos
    token_mappings = get_token_mappings(config.nvars)
    OV = first_layer_attn_circuits(model).OV
    
    colon_embed = W_E[:,token_mappings[":"]] + W_pos[-1,:]

    # Our formula is 10/41 '(', 10/41 ')', 20/41 uniform variables, and 1/41 ':' (for default length)
    tok_count = 4 * config.nclauses + 1

    attn_out = OV["pos"].mean(dim=0) # add the average positional OV output given uniform attention
    attn_out += config.nclauses/tok_count * (OV["tok"][token_mappings['(']] + OV['tok'][token_mappings[')']])
    attn_out += 1/tok_count * OV["tok"][token_mappings[':']]
    
    for tok, freq in tok_probs.items():
        idx = tok if toks else token_mappings[tok]
        attn_out += freq * (2 * config.nclauses)/tok_count * OV["tok"][idx]

    colon_embed += attn_out

    return colon_embed + model.blocks[0].mlp(colon_embed.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)


def first_layer_attn_circuits(model=None, **kwargs):
    attn = model.blocks[0].attn
    W_Q = attn.W_Q
    W_K = attn.W_K
    W_V = attn.W_V
    W_O = attn.W_O

    W_E = model.embed.W_E
    W_pos = model.pos_embed.W_pos
    K = {}
    Q = {}
    OV = {}

    for mat, d in zip([W_K, W_Q, (W_V, W_O)], [K, Q, OV]):
        if type(mat) is tuple:
            first, second = mat
        else:
            first, second = mat, None

        for embed_type, embed in zip(["tok", "pos"], [W_E.T, W_pos]):
            result = torch.einsum('ihd, pd->ph', first, embed)
            if second is not None:
                result = torch.einsum('dh, ph -> pd', second, result)

            d[embed_type] = result
    QK = {}

    for dst_type in ["tok", "pos"]:
        for src_type in ["tok", "pos"]:
            QK[f"dst_{dst_type}_src_{src_type}"] = torch.einsum('ph, qh->qp', K[src_type], Q[dst_type])

    return SimpleNamespace(QK=QK, OV=OV)

def intermediates(input, model=None, config=None, **kwargs):
    if type(input) is str:
        input = torch.Tensor(
            helpers.encode_to_toks(
                input,
                config,
                pad_with_tautologies=False,
                randomize_tautologies=False,
                shuffle=False
            )
        ).long().to(config.device).unsqueeze(0)[:,:-1]
    elif type(input) is np.ndarray:
        input = torch.Tensor(input).long().to(config.device)[:, :-1]
        
    cache = {}
    model.remove_all_hooks()
    model.cache_all(cache)
    model_output = model(input)
    
    return cache

def classify(input_seq, toks=False, shuffle=True, model=None, config=None, **kwargs):
    if toks:
        tokens = input_seq
    else:
        prediction_tok = 2 * config.nvars + 2
        tokens = helpers.encode_to_toks(input_seq, config, shuffle=shuffle)
    
    x, y = tokens[:-1], tokens[-1]
    
    toks = torch.LongTensor(x).to(config.device).unsqueeze(0)
    output = model(toks)
    prediction = output[0,-1].argmax(dim=-1).cpu()
    return helpers.decode_to_CNFs(prediction.unsqueeze(0).numpy(), config)

def vis_sat(formula, toks=False, model=None, config=None, classify=None, **kwargs):
    token_mappings = get_token_mappings(config.nvars)
    rev_token_mappings = {val : key for key, val in token_mappings.items()}
    
    true = (rev_token_mappings[formula[-1]] if toks else formula[-1]) if len(formula) >= 2 and formula[-2] == (token_mappings[":"] if toks else ":") else "#"
    print(f"Predicted {classify(formula, toks=toks, model=model)}, true: {true}")
     
    return implication_graph(formula, config, toks=toks)[0]


def plot_attentions(sample, intermediates=None, path="img/attns_sample.pdf", **kwargs):
    ints = intermediates(sample[np.newaxis,:])
    attentions_layer_1 = ints["blocks.0.attn.hook_attn"][0]
    attentions_layer_2 = ints["blocks.1.attn.hook_attn"][0]
    
    cmap = LinearSegmentedColormap.from_list("wtor", [(1,1,1),(1,0,0)])
    
    im = None
    def plot_attentions(ax, attns, title):
        nonlocal im
        im = ax.imshow(attns.cpu(), vmin=0, vmax=1, cmap=cmap)
        ax.set_xlabel("Position")
        ax.set_ylabel("Position")
        ax.set_title(title)
    
    rows = 1
    cols = 5
    fig, ax = plt.subplots(rows, cols, figsize=(cols*5,rows*5))
    
    def get_ax(i):
        if rows > 1:
            row = i // rows
            col = i % rows
    
            return ax[row, col]
    
        return ax[i]
    
    plot_attentions(get_ax(0), attentions_layer_1[0], "Layer 1, Head 1: Attention Scores")
    for i in range(1,5):
        plot_attentions(get_ax(i), attentions_layer_2[i-1], f"Layer 2, Head {i}: Attention Scores")
    
    fig.subplots_adjust(right=0.94)
    cbar_ax = fig.add_axes([0.95, 0.15, 0.01, 0.7])
    fig.colorbar(im, cax=cbar_ax, cmap=cmap)
    plt.savefig(path, pad_inches=0, bbox_inches="tight")

def plot_average_test_attentions(test=None, config=None, intermediates=None, path="img/test_attns.pdf", **kwargs):
    batch_size = 5000
    dataset_size = test.shape[0]
    batches = int(ceil(dataset_size / batch_size))
    
    attentions_layer_1 = torch.zeros(config.n_ctx).to(config.device)
    attentions_layer_2 = torch.zeros(4, config.n_ctx).to(config.device)
    
    for i in range(batches):
        start = batch_size * i
        end = min(batch_size * (i+1), dataset_size)
    
        batch = test[start:end]
        ints = intermediates(batch)
        attentions_layer_1 += ints["blocks.0.attn.hook_attn"][:,0,2::4,:].sum(dim=[0,1]) / config.nclauses
        attentions_layer_2 += ints["blocks.1.attn.hook_attn"][:,:,-1,:].sum(dim=0)
    
    attentions_layer_1 /= dataset_size
    attentions_layer_2 /= dataset_size
    
    rows = 1
    cols = 5
    fig, ax = plt.subplots(rows, cols, figsize=(cols*5,rows*5))
    
    def get_ax(i):
        if rows > 1:
            row = i // rows
            col = i % rows
    
            return ax[row, col]
    
        return ax[i]
    
    get_ax(0).set_title("Average attention scores \n(second tokens of each clause):\n Layer 1, Head 1")
    
    for i in range(1,5):
        get_ax(i).set_title(f"Average attention scores (':' token):\n Layer 2, Head {i}")
    
    for i in range(5):
        get_ax(i).set_xlabel("Position")
        get_ax(i).set_ylabel("Average weight")
    
    get_ax(0).plot(attentions_layer_1.cpu())
    for i in range(1,5):
        get_ax(i).plot(attentions_layer_2[i-1].cpu())
    
    plt.savefig(path, pad_inches=0, bbox_inches="tight")


def get_layer_1_score_fns(model=None, config=None, **kwargs):
    attn = model.blocks[0].attn
    W_Q = attn.W_Q
    W_K = attn.W_K
    W_V = attn.W_V
    W_O = attn.W_O

    W_E = model.embed.W_E
    W_pos = model.pos_embed.W_pos

    K = {}
    Q = {}
    OV = {}

    for mat, d in zip([W_K, W_Q, (W_V, W_O)], [K, Q, OV]):
        if type(mat) is tuple:
            first, second = mat
        else:
            first, second = mat, None

        for embed_type, embed in zip(["tok", "pos"], [W_E.T, W_pos]):
            result = torch.einsum('ihd, pd->ph', first, embed)
            if second is not None:
                result = torch.einsum('dh, ph -> pd', second, result)

            d[embed_type] = result

    QK = {}

    for dst_type in ["tok", "pos"]:
        for src_type in ["tok", "pos"]:
            QK[f"dst_{dst_type}_src_{src_type}"] = torch.einsum('ph, qh->qp', K[src_type], Q[dst_type])

    token_mappings = get_token_mappings(config.nvars)

    sequence = [[dtype, stype] for dtype in ["tok", "pos"] for stype in ["tok", "pos"]]
    scale = 1/np.sqrt(config.d_head(0)) # All scores are scaled by this amount pre-softmax

    def get_score_data(dst_type, src_type, scale=scale):
        qk_mat = QK[f"dst_{dst_type}_src_{src_type}"]

        if src_type == "tok":
            relevant_src = list(concat(
                [[f"x{i}", f"Not(x{i})"] for i in range(config.nvars)] + [['(', ')']]
            ))
            relevant_src_map = list(concat([
                [token_mappings[f"x{i}"], token_mappings[f"Not(x{i})"]] 
                for i in range(config.nvars)
            ] + [
                    [token_mappings['('], token_mappings[')']]
            ]))
        else:
            relevant_src = list(range(40))
            relevant_src_map = relevant_src

        if dst_type == "tok":
            relevant_dst = list(concat([[f"x{i}", f"Not(x{i})"] for i in range(config.nvars)]))
            relevant_dst_map = list(concat([
                [token_mappings[f"x{i}"], token_mappings[f"Not(x{i})"]] 
                for i in range(config.nvars)
            ]))
        else:
            relevant_dst = list(range(2, 40, 4))
            relevant_dst_map = relevant_dst

        return relevant_src, relevant_src_map, relevant_dst, relevant_dst_map, qk_mat * scale

    def get_score_dict(dst_type, src_type, nan=np.nan, destination_first=True, scale=scale):
        relevant_src, relevant_src_map, relevant_dst, relevant_dst_map, qk_mat = get_score_data(dst_type, src_type, scale=scale)

        scores = defaultdict(lambda: defaultdict(float))

        for (dst, dst_id) in zip(relevant_dst, relevant_dst_map):
            for (src, src_id) in zip(relevant_src, relevant_src_map):
                score = qk_mat[dst_id, src_id].cpu().item()
                if src_type == "pos" and dst_type == "pos" and src > dst:
                    score = nan

                if destination_first:
                    scores[dst][src] = score
                else:
                    scores[src][dst] = score

        return scores

    def get_score_df(dst_type, src_type, nan=np.nan, scale=scale):
        relevant_src, relevant_src_map, relevant_dst, relevant_dst_map, qk_mat = get_score_data(dst_type, src_type, scale=scale)

        records = []

        for (dst, dst_id) in zip(relevant_dst, relevant_dst_map):
            for (src, src_id) in zip(relevant_src, relevant_src_map):
                score = qk_mat[dst_id, src_id].cpu().item()
                if src_type == "pos" and dst_type == "pos" and src > dst:
                    score = nan

                records.append({"Destination": dst, "Source": src, "Score": score})

        return pd.DataFrame.from_records(records)

    def softmax_scores(group):
        exp_scores = np.exp(np.nan_to_num(group.Score, nan=-np.inf))
        softmax_scores = exp_scores / exp_scores.sum()

        group["Softmax Score"] = softmax_scores

        return pd.DataFrame({
            "Source": group.Source,
            "Softmax Score": softmax_scores,
        })

    def plot_df(df, title, softmax=False):
        softmax_df = df.groupby("Destination").apply(softmax_scores).reset_index()
        df["Softmax Score"] = softmax_df["Softmax Score"]

        if softmax:
            return alt.Chart(df, title=title + " (Softmax)").mark_rect(width=20, height=20).encode(
                alt.Y("Destination:N"),
                alt.X("Source:N"),
                alt.Color("Softmax Score"),
            ) 
        else:
            return alt.Chart(df, title=title).mark_rect(width=20, height=20).encode(
                alt.Y("Destination:N"),
                alt.X("Source:N"),
                alt.Color("Score"),
            ) 

    def plot_score_df(dst_type, src_type, softmax=False):
        type_map = {
            "tok": "Token",
            "pos": "Position",
        }
        title = f"{type_map[src_type]} to {type_map[dst_type]} Attention Scores"

        df = get_score_df(dst_type, src_type)

        return plot_df(df, title, softmax=softmax)

    return {
        "get_score_data": get_score_data,
        "get_score_dict": get_score_dict, 
        "get_score_df": get_score_df, 
        "softmax_scores": softmax_scores, 
        "plot_df": plot_df, 
        "plot_score_df": plot_score_df
    }
    
def plot_extreme_first_layer_scores(get_score_dict=None, path="img/min_weight.pdf", **kwargs):
    pos_to_pos = get_score_dict("pos", "pos", nan=-np.inf)
    tok_to_pos = get_score_dict("pos", "tok", nan=-np.inf)
    pos_to_tok = get_score_dict("tok", "pos", nan=-np.inf)
    tok_to_tok = get_score_dict("tok", "tok", nan=-np.inf)
    vars = [f"x{i}" for i in range(5)] + [f"Not(x{i})" for i in range(5)]
    
    def tok_options(p):
        match p % 4:
            case 0:
                return ['(']
            case 1:
                return vars
            case 2:
                return vars
            case 3:
                return [')']
                
    def get_extreme_score(dst_pos, src_pos, src_toks=None, dst_toks=None, extreme=max):
        if src_toks is None:
            src_toks = tok_options(src_pos)
        
        if dst_toks is None:
            dst_toks = tok_options(dst_pos)
        
        pos_pos_score = pos_to_pos[dst_pos][src_pos]
    
        remaining_potential_scores = []
        
        for td in dst_toks:
            for ts in src_toks:
                remaining_potential_scores.append(tok_to_pos[dst_pos][ts] + pos_to_tok[td][src_pos] + tok_to_tok[td][ts])
    
        return pos_pos_score + extreme(remaining_potential_scores)
        
    min_prev = []
    min_clause = []
    min_prev_by_dst = []
    min_clause_by_dst = []
    
    dst_posns = np.arange(2, 40, 4, dtype=np.int_)
    
    for i in dst_posns:
        dst_toks = tok_options(i)
    
        min_prev_scores = {}
        min_clause_scores = {}
        
        for dt in dst_toks:
            min_prev_tok = get_extreme_score(i, i-1, dst_toks=[dt], extreme=min)
            current = get_extreme_score(i, i, dst_toks=[dt], src_toks=[dt])
            max_prev = np.array([get_extreme_score(i, j, dst_toks=[dt], extreme=max) for j in range(i-1)])
        
            min_prev_scores[dt] = np.exp(min_prev_tok) / (np.exp(min_prev_tok) + np.exp(current) + np.exp(max_prev).sum())
            min_clause_scores[dt] = (np.exp(min_prev_tok) + np.exp(current)) / (np.exp(min_prev_tok) + np.exp(current) + np.exp(max_prev).sum())
    
        min_weight_prev = min(min_prev_scores.values())
        min_weight_clause = min(min_clause_scores.values())
    
        min_prev.append(min_weight_prev)
        min_clause.append(min_weight_clause)
        min_prev_by_dst.append(min_prev_scores)
        min_clause_by_dst.append(min_clause_scores)
    
    plt.figure(figsize=(5,5))
    plt.plot(np.arange(10), min_prev, label="First token of clause")
    plt.plot(np.arange(10), min_clause, label="Full clause")
    plt.xlabel("Clause number")
    plt.ylabel("Minimum softmax attention score")
    plt.legend()
    plt.savefig(path)

def get_avg_effect_df(get_score_df=None, **kwargs):
    def get_avg_effect_src_tok(group):
            is_parens = (group.Source == "(") | (group.Source == ")")
            effect = group[is_parens]
            del effect["Destination"]
            effect = pd.concat([effect, pd.DataFrame({"Source": ["var"], "Score": [group[~is_parens].Score.mean()]})])
            return effect
            
    tok_to_pos_avg_effect = get_score_df("pos", "tok").groupby("Destination").apply(
        get_avg_effect_src_tok
    ).reset_index().groupby("Destination").apply(lambda x: x.set_index("Source"))
        
    def get_avg_effect(group):
        effect = group.Score.mean()
        return pd.DataFrame({
            "Destination": ["var"],
            "Score": effect,
        })
        
    pos_to_tok_avg_effect = get_score_df("tok", "pos").groupby("Source").apply(
        get_avg_effect
    ).reset_index()
    pos_to_tok_avg_effect = pos_to_tok_avg_effect.set_index(
        pd.MultiIndex.from_tuples(
            zip(["var"] * len(pos_to_tok_avg_effect), pos_to_tok_avg_effect.Source.values), 
            names=["Destination", "Source"]
        )
    )
    
    tok_to_tok_avg_effect = get_score_df("tok", "tok").groupby("Destination").apply(get_avg_effect_src_tok).reset_index().groupby("Source").apply(
        get_avg_effect
    ).reset_index().groupby("Destination").apply(lambda x: x.set_index("Source"))
    
    df = get_score_df("pos", "pos", nan=-np.inf)
    for i in df.index:
        row = df.loc[i]
        dst = row.Destination
        src = row.Source
        score = row.Score
        
        if src % 4 in [1,2]:
            src_tok = "var"
        elif src % 4 == 0:
            src_tok = '('
        else:
            src_tok = ')'
    
        dst_tok = "var"
        score += tok_to_pos_avg_effect.loc[dst].loc[src_tok].Score
        score += pos_to_tok_avg_effect.loc[dst_tok].loc[src].Score
        score += tok_to_tok_avg_effect.loc[dst_tok].loc[src_tok].Score
    
        df.loc[i, "Score"] = score
    
    return df

def get_colon_token_first_layer_scores(first_layer_attn_circuits=None, config=None, **kwargs):
    scores = []
    token_mappings = get_token_mappings(config.nvars)
    scale = 1/np.sqrt(config.d_head(0)) # All scores are scaled by this amount pre-softmax
    vars = [f"x{i}" for i in range(config.nvars)] + [f"Not(x{i})" for i in range(config.nvars)]
    QK = first_layer_attn_circuits().QK
    col_id = token_mappings[":"]

    for p in range(41):
        for t in vars + ["(", ")", ":"]:
            tok_id = token_mappings[t]
            score = QK["dst_pos_src_pos"][-1][p] + QK["dst_pos_src_tok"][-1][tok_id]
            score += QK["dst_tok_src_pos"][col_id][p] + QK["dst_tok_src_pos"][col_id][tok_id]
            score *= scale

            scores.append({"Position": p, "Token": t, "Score": score.cpu().item()})
    df = pd.DataFrame.from_records(scores)
    return df

@dataclass
class Clause:
    left: int
    right: int
    
    def to_var_set(self):
        return {self.left, self.right}
    
    def __eq__(self, other):
        return self.to_var_set() == other.to_var_set()

def to_clause(id, clause_map=None, use_tok_ids=True):
    l,r = clause_map[id]
    if not use_tok_ids:
        l, r = rev_token_mappings[l], rev_token_mappings[r]
    return Clause(l, r)

def mean_pos_layer_1_clause(*args, **kwargs):
    with torch.no_grad():
        embeds = []
        for clause_id in range(10):
            embeds.append(layer_1_clause(*args, **kwargs, clause_id=clause_id))
    
        return torch.cat([e.unsqueeze(0) for e in embeds]).mean(dim=0)

def get_layer_1_clause_map(mean_pos_layer_1_clause=None, config=None, **kwargs):
    token_mappings = get_token_mappings(config.nvars)
    vars = list(concat(
        [[f"x{i}", f"Not(x{i})"] for i in range(config.nvars)]
    ))
    token_ids = [token_mappings[v] for v in vars]

    return {
        (l, r): mean_pos_layer_1_clause(l, r, toks=True, parens=False)
        for l in token_ids
        for r in token_ids
    }

def get_layer_1_clauses(layer_1_clause_map=None, config=None, **kwargs):
    token_mappings = get_token_mappings(config.nvars)
    vars = list(concat(
        [[f"x{i}", f"Not(x{i})"] for i in range(config.nvars)]
    ))
    token_ids = [token_mappings[v] for v in vars]
    
    return torch.Tensor(np.array([
        [layer_1_clause_map[(l, r)].cpu().numpy() for r in token_ids] 
        for l in token_ids
    ])).to(config.device).reshape(len(token_ids) ** 2, -1)

def get_truth_tables_fn(config=None, token_mappings=None, truth_table_inputs=None, **kwargs):
    nvars = config.nvars
    
    to_num = {}
        
    for i in range(config.nvars):
        xi = f"x{i}"
        nxi = f"Not({xi})"
    
        to_num[token_mappings[xi]] = i+1
        to_num[token_mappings[nxi]] = -i-1
    
    def format_clause_list(clauses):
        return [[to_num[ci.left], to_num[ci.right]] for ci in clauses]
    
    @njit
    def eval_formula(clause_list: npt.NDArray[np.int_], assignment_vec: npt.NDArray[np.bool_]):
        def holds(v):
            if v > 0:
                return assignment_vec[v-1]
    
            return not assignment_vec[-v-1]
        
        valid = True
        for clause in clause_list:
            l, r = clause
            
            valid = valid and (holds(l) or holds(r))
            
        return valid
    
    @njit
    def truth_table(clause_list: npt.NDArray[np.int_]):
        table = np.empty(2**nvars, dtype=np.bool_)   
        
        for i, assignment in enumerate(truth_table_inputs):
            table[i] = eval_formula(clause_list, assignment)
    
        return table
    
    @njit
    def truth_tables_numba(clause_lists: npt.NDArray[np.int_]):
        samples = len(clause_lists)
        table = np.empty((samples, 2**nvars), dtype=np.bool_)   
        
        for i in range(samples):
            table[i] = truth_table(clause_lists[i])
    
        return table
    
    def decode_to_clause_lists(inputs, clauses=False):
        samples = len(inputs)
        clause_lists = np.empty((samples, config.nclauses, 2), dtype=np.int_)
    
        for i, sample in enumerate(inputs):
            if clauses:
                clause_list = sample
            else:
                clause_list = []
                for j in range(config.nclauses):
                    l, r = sample[4*j+1:4*j+3]
                    clause_list.append(Clause(l, r))
    
            clause_lists[i] = format_clause_list(clause_list)
    
        return clause_lists
    
    def truth_tables(input: list[list[Clause]] | npt.NDArray[np.int_], clauses=False):
        return truth_tables_numba(decode_to_clause_lists(input, clauses=clauses))

    return {
        "truth_tables": truth_tables,
        "truth_tables_from_clauses": partial(truth_tables, clauses=True),
    }

def get_all_intermediates(data, keys=None, model=None, config=None, intermediates=None, batch_size=5000, device=torch.device("cpu"), **kwargs):
    dataset_size = data.shape[0]
    batches = int(ceil(dataset_size / batch_size))

    batch_results = defaultdict(list)

    if type(keys) is str:
        keys = [keys]
    
    for i in range(batches):
        start = batch_size * i
        end = min(batch_size * (i+1), dataset_size)
    
        batch = data[start:end]
        ints = intermediates(batch, model=model)
        
        for k, v in ints.items():
            if keys is None or k in keys:
                batch_results[k].append(v.to(device))

    for k, v in batch_results.items():
        batch_results[k] = torch.cat(v)

    if keys is not None and len(keys) == 1:
        return batch_results[keys[0]]

    return batch_results

def get_mean_intermediates(data, keys=None, model=None, intermediates=None, batch_size=5000, device=torch.device("cpu"), **kwargs):
    dataset_size = data.shape[0]
    batches = int(ceil(dataset_size / batch_size))

    batch_results = {}

    if type(keys) is str:
        keys = [keys]
    
    for i in range(batches):
        start = batch_size * i
        end = min(batch_size * (i+1), dataset_size)
    
        batch = data[start:end]
        ints = intermediates(batch, model=model)
        
        for k, v in ints.items():
            if keys is None or k in keys:
                past_count, avg = batch_results.get(k, (0,0))
                total_count = batch.shape[0] + past_count
                batch_results[k] = (total_count, (avg * past_count + v.sum(dim=0))/total_count)

    for k, v in batch_results.items():
        batch_results[k] = v[1].to(device)

    if keys is not None and len(keys) == 1:
        return batch_results[keys[0]]

    return batch_results

def hook_diffs(substitute_model, data, key_set=None, model=None, intermediates=None, batch_size=5000, device=torch.device("cpu"), **kwargs):
    with torch.no_grad():
        dataset_size = data.shape[0]
        batches = int(ceil(dataset_size / batch_size))
        
        batch_results = defaultdict(list)
        
        for i in range(batches):
            start = batch_size * i
            end = min(batch_size * (i+1), dataset_size)
        
            batch = data[start:end]
            ints_original = intermediates(batch, model=model)
            ints_substitute = intermediates(batch, model=substitute_model)
            matching_keys = set(ints_original.keys()).intersection(set(ints_substitute.keys()))

            if key_set is not None:
                matching_keys = matching_keys.intersection(set(key_set))
            
            for k in matching_keys:
                original = ints_original[k]
                substitute = ints_substitute[k]
                
                batch_results[k].append((substitute - original).to(device))

        for k, v in batch_results.items():
            batch_results[k] = torch.cat(v)

        if key_set is not None and len(key_set) == 1:
            return batch_results[list(key_set)[0]]
        
        return batch_results

def output_diffs(substitute_model, data, key_set=None, config=None, model=None, intermediates=None, batch_size=5000, metric="label_flip", device=torch.device("cpu"), **kwargs):
    with torch.no_grad():
        dataset_size = data.shape[0]
        batches = int(ceil(dataset_size / batch_size))
        
        diffs = []
        
        for i in range(batches):
            start = batch_size * i
            end = min(batch_size * (i+1), dataset_size)
        
            batch = torch.Tensor(data[start:end]).long().to(config.device)[:,:-1]
            original = model(batch)
            substitute = substitute_model(batch)

            if metric == "softmax":
                original = F.softmax(original[:,-1,:], dim=-1)
                substitute = F.softmax(substitute[:,-1,:], dim=-1)
    
                diffs.append((substitute - original).to(device))
            elif metric == "score":
                diffs.append((substitute - original).to(device))
            elif metric == "label_flip":
                original = original[:,-1,:].argmax(dim=-1)
                substitute = substitute[:,-1,:].argmax(dim=-1)
                
                diffs.append(original != substitute)
                    
        return torch.cat(diffs)

def alphas(model=None, config=None, layer_1_clauses=None, units_for_sat=None, clause_map=None, **kwargs):
    @batched
    def alpha_layer_1(inputs):
        inputs = inputs.to(config.device)
        clause_idxs = torch.LongTensor([4*i+2 for i in range(config.nclauses)]).to(config.device)
    
        clause_embeds = inputs[:, clause_idxs]
        cosine_sims = F.cosine_similarity(
            layer_1_clauses.unsqueeze(0).unsqueeze(1), 
            clause_embeds.unsqueeze(2), 
            dim=-1,
        )
        
        clause_ids = cosine_sims.argmax(dim=-1).detach().cpu().numpy()
        
        return [
            [to_clause(clause_ids[i, j], clause_map=clause_map) for j in range(clause_ids.shape[-1])]
            for i in range(clause_ids.shape[0])
        ]
    
    def alpha_mlp_hidden(inputs):
        mlp_x = inputs[1]
        return (mlp_x[:, -1, units_for_sat.to(mlp_x.device)] >= config.activation_threshold).detach().cpu().numpy()
    
    return [
        identity,
        alpha_layer_1,
        alpha_mlp_hidden,
        identity,
    ]

def gammas(model=None, config=None, units_for_sat=None, means=None, rev_clause_map=None, layer_1_clauses=None, **kwargs):
    def gamma_layer_1(inputs):
        embeds = means["blocks.0.hook_resid_post"].to(config.device).unsqueeze(0).tile((len(inputs), 1, 1))
        clauses_flat = torch.empty(embeds.shape[0] * config.nclauses, dtype=torch.long)
        
        for i, clause in enumerate(concat(inputs)):
            clauses_flat[i] = rev_clause_map[(clause.left, clause.right)]
    
        clause_embeds = layer_1_clauses[clauses_flat].reshape(embeds.shape[0], config.nclauses, -1)
        embeds[:,2::4] = clause_embeds
    
        return embeds
    
    def gamma_hidden_layer(inputs):
        mean_mlp_hidden = means["blocks.1.mlp.hook_post"].to(config.device).unsqueeze(0).tile((len(inputs), 1, 1))
        mean_attn_out = means["blocks.1.hook_attn_out"].to(config.device).unsqueeze(0).tile((len(inputs), 1, 1))
    
        slice = mean_mlp_hidden[:,-1,units_for_sat]
        activated = torch.BoolTensor(inputs).to(config.device)
        slice[~activated] = 0
        slice[activated] = config.high_activation
        mean_mlp_hidden[:,-1,units_for_sat] = slice
    
        return mean_attn_out, mean_mlp_hidden
    
    return [
        identity,
        gamma_layer_1,
        gamma_hidden_layer,
        identity,
    ]

def get_concrete_model_and_layers(model=None, config=None, token_mappings=None, **kwargs):
    class Layer1(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
    
        def forward(self, x):
            x = torch.LongTensor(x[:,:-1]).to(config.device)
            x = self.model.embed(x)
            x = self.model.pos_embed(x)
            return self.model.blocks[0](x)
    
    class AttnMLPHiddenLayer(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            layer = self.model.blocks[1]
            self.attn = layer.attn
            mlp = layer.mlp
            self.W_in = mlp.W_in
            self.b_in = mlp.b_in
            self.act_type = mlp.act_type
            
            self.hook_attn_out = HookPoint()
            self.hook_resid_pre = HookPoint()
            self.hook_resid_mid = HookPoint()
            self.hook_pre = HookPoint()
            self.hook_post = HookPoint()
    
        def forward(self, x):
            x = self.hook_resid_mid(x + self.hook_attn_out(self.attn((self.hook_resid_pre(x)))))
            mlp_x = self.hook_pre(torch.einsum('md,bpd->bpm', self.W_in, x) + self.b_in)
            if self.act_type=='ReLU':
                mlp_x = F.relu(mlp_x)
            elif self.act_type=='GeLU':
                mlp_x = F.gelu(mlp_x)
            mlp_x = self.hook_post(mlp_x)
            
            return x, mlp_x
    
    class ArgmaxMLPOutputLayer(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            layer = self.model.blocks[1]
            mlp = layer.mlp
            self.W_out = mlp.W_out
            self.b_out = mlp.b_out
            self.hook_mlp_out = HookPoint()
            self.hook_resid_post = HookPoint()
    
        def forward(self, inputs):
            x, mlp_x = inputs
    
            mlp_x = torch.einsum('dm,bpm->bpd', self.W_out, mlp_x) + self.b_out
            x = self.hook_resid_post(x + self.hook_mlp_out(mlp_x))
    
            x = self.model.unembed(x)
    
            x = x[:,-1,:].argmax(dim=-1) == token_mappings['s']
    
            x = x.detach().cpu().numpy()
            return x
    
    model_layers = [
        Layer1(model),
        AttnMLPHiddenLayer(model),
        # We take the output of the model as SAT/UNSAT, ignoring confidence scores
        ArgmaxMLPOutputLayer(model),
    ]
    
    return {
        "concrete_model_layers": model_layers,
        "concrete_model": compose_reversed(model_layers),
    }

def generate_cache(config=None, model=None, intermediates=None, all_data=None, train_idx=None, truth_tables=None, token_mappings=None, get_mean_intermediates=None, train=None, units_for_sat=None, feature_names_truth_tables=None, **kwargs):
    attn = model.blocks[1].attn
    W_Q = attn.W_Q
    W_K = attn.W_K
    W_V = attn.W_V
    W_O = attn.W_O
    n_heads = W_Q.shape[0]
    
    W_out = model.blocks[1].mlp.W_out
    W_U = model.unembed.W_U
    
    W_E = model.embed.W_E
    W_pos = model.pos_embed.W_pos
    
    batch_size = 5000
    dataset_size = all_data.shape[0]
    batches = int(ceil(dataset_size / batch_size))

    with torch.no_grad():
        MLP_out = torch.einsum('et, em -> tm', W_U, W_out)[token_mappings['s']].cpu().numpy()
        activation_scores = np.empty((dataset_size, MLP_out.shape[0]))
        outputs = np.empty((dataset_size))
        
        for i in range(batches):
            start = batch_size * i
            end = min(batch_size * (i+1), dataset_size)
        
            batch = all_data[start:end]
            ints = intermediates(batch)
            activation_scores[start:end] = ints["blocks.1.mlp.hook_post"][:,-1,:].cpu()
            outputs[start:end] = ((ints["blocks.1.hook_resid_post"][:, -1] @ W_U).argmax(dim=1) ==  token_mappings['s']).cpu()
        
        MLP_max_effect = activation_scores[train_idx].max(axis=0) * MLP_out
        
    all_truth_tables = truth_tables(all_data)
    truth_tables_train = all_truth_tables[train_idx]
    train_activation_scores = activation_scores[train_idx]

    def models(i, max_leaf_nodes=4):
        m = sklearn.tree.DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes)
        m.fit(truth_tables_train, train_activation_scores[:, i] >= config.activation_threshold)
        return prune(m)

    node_models = list(parmap_unbatched(models, range(MLP_out.shape[0])))

    means = get_mean_intermediates(train)

    # Determine evaluating neurons
    units_for_sat = (torch.Tensor(MLP_max_effect) > 0.1).nonzero().flatten()

    # Compute the mean activation of each of the evaluating neurons given formulas satisfiable with each assignment
    # and SAT/UNSAT formulas in general
    y = all_data[train_idx,-1] == token_mappings['s']
    train_activations_scaled = train_activation_scores * MLP_out
    
    df = pd.DataFrame(
        data=np.hstack((truth_tables_train, train_activation_scores[:, units_for_sat.cpu()], train_activations_scaled[:, units_for_sat.cpu()], y[:, np.newaxis])),
        columns=feature_names_truth_tables + [f"SCORE({u})" for u in units_for_sat] + [f"SAT_LOGIT_INCREASE({u})" for u in units_for_sat] + ["SAT"]
    )
    
    unit_scores = defaultdict(defaultdict)
    unit_impacts = defaultdict(defaultdict)
    
    for k in feature_names_truth_tables + ["SAT"]:
        grouped = df.groupby(k).mean()
        for u in units_for_sat:
            u = u.item()
            
            unit_scores[u][k] = grouped.loc[1, f"SCORE({u})"]
            unit_impacts[u][k] = grouped.loc[1, f"SAT_LOGIT_INCREASE({u})"]
            
    is_not_sat_with = lambda assgn: df[assgn] == 0
    unsat_series = [is_not_sat_with(assgn) for assgn in feature_names_truth_tables]
    unsat = unsat_series[0]
    for s in unsat_series:
        unsat = unsat & s
        
    mean_effect = df[unsat].mean()
    for u in units_for_sat:
        u = u.item()
        
        unit_scores[u]["UNSAT"] = mean_effect[f"SCORE({u})"]
    
    return {
        "outputs": outputs,
        "MLP_out": MLP_out,
        "MLP_max_effect": MLP_max_effect,
        "activation_scores": activation_scores,
        "node_models": node_models,
        "means": means,
        "all_truth_tables": all_truth_tables,
        "unit_scores": unit_scores,
        "unit_impacts": unit_impacts,
        "units_for_sat": units_for_sat, 
        "sat_coeffs": MLP_out[units_for_sat],
    }

def get_abstract_model_and_layers(node_models=None, truth_tables_from_clauses=None, units_for_sat=None, config=None, **kwargs):
    clause_idxs = np.array([4*i+2 for i in range(config.nclauses)])
    def parse_clauses(inputs):
        return [
            [Clause(inputs[i, idx-1], inputs[i, idx]) for idx in clause_idxs]
            for i in range(inputs.shape[0])
        ]
    
    def calculate_decision_trees(inputs, parallel=False):
        input_truth_tables = truth_tables_from_clauses(inputs)
    
        def true_abstract_state(m):    
            return m.predict(input_truth_tables)
        
        get_true_abstract_states = compose(
            lambda a: a[units_for_sat.cpu()].T, 
            partial(np.array, dtype=np.bool_), 
            list, 
            partial(parmap_unbatched if parallel else map, true_abstract_state)
        )
        return get_true_abstract_states(node_models)
    
    abstract_layers = [
        parse_clauses,
        calculate_decision_trees,
        apply_or,
    ]

    return {
        "abstract_model_layers": abstract_layers,
        "abstract_model": compose_reversed(abstract_layers),
    }

def get_axiom_models(concrete_model_layers=None, abstract_model_layers=None, alphas=None, gammas=None, **kwargs):
    def get_prefix_equivalence_model_rhs(i):
        return compose_reversed(
            alphas[0], 
            abstract_model_layers[:i+1],
        )
    
    def get_component_equivalence_model_rhs(i):
        return compose_reversed(
            batched(concrete_model_layers[:i]),
            alphas[i], 
            abstract_model_layers[i],
        )
    
    def get_equivalence_model_lhs(i):
        return compose_reversed(
            batched(concrete_model_layers[:i+1]),
            alphas[i+1],
        )
    
    def get_prefix_replaceability_model(i):
        return compose_reversed(
            alphas[0],
            abstract_model_layers[:i+1],
            batched(gammas[i+1], concrete_model_layers[i+1:]),
        )
    
    def get_component_replaceability_model(i):
        return compose_reversed(
            batched(concrete_model_layers[:i]),
            alphas[i], 
            abstract_model_layers[i], 
            batched(gammas[i+1], concrete_model_layers[i+1:]),
        )

    return {
        "get_prefix_equivalence_model_rhs": get_prefix_equivalence_model_rhs, 
        "get_component_equivalence_model_rhs": get_component_equivalence_model_rhs,
        "get_equivalence_model_lhs": get_equivalence_model_lhs, 
        "get_prefix_replaceability_model": get_prefix_replaceability_model, 
        "get_component_replaceability_model": get_component_replaceability_model,
    }

def get_axiom_evaluators(get_prefix_equivalence_model_rhs=None, get_component_equivalence_model_rhs=None, get_equivalence_model_lhs=None, get_prefix_replaceability_model=None, get_component_replaceability_model=None, test=None, concrete_model=None, **outer_kwargs):
    def get_epsilon(lhs, rhs, inputs=test, allowable_deviation=0, return_match_rate=False, confidence_level=0.95):
        """
        Returns a `confidence_level` one-sided Clopper-Pearson confidence-interval upper bound for the
        disagreement rate on the `inputs`. If the output on a sample is vector-valued, distinct outputs
        on up to `allowable_deviation` components are treated as identical.
        """
        lhs_out = to_numpy(lhs(inputs))
        rhs_out = to_numpy(rhs(inputs))
    
        if len(lhs_out.shape) == 1:
            matches = lhs_out == rhs_out
        else:
            matches = (lhs_out != rhs_out).sum(axis=-1) <= allowable_deviation

        eps = get_high_eps_confidence_interval(matches, confidence_level = confidence_level)

        if not return_match_rate:
            return eps

        match_rate = matches.astype(np.float_).mean()

        return {
            f"epsilon (confidence {confidence_level})": eps,
            "disagreement rate": 1 - match_rate,
        }
    
    def get_prefix_equivalence_eps(layer, **kwargs):
        lhs = get_equivalence_model_lhs(layer)
        rhs = get_prefix_equivalence_model_rhs(layer)
    
        return get_epsilon(lhs, rhs, **kwargs)
    
    def get_component_equivalence_eps(layer, **kwargs):
        lhs = get_equivalence_model_lhs(layer)
        rhs = get_component_equivalence_model_rhs(layer)
    
        return get_epsilon(lhs, rhs, **kwargs)
    
    def get_prefix_replaceability_eps(layer, **kwargs):
        lhs = batched(concrete_model)
        rhs = get_prefix_replaceability_model(layer)
    
        return get_epsilon(lhs, rhs, **kwargs)
    
    def get_component_replaceability_eps(layer, **kwargs):
        lhs = batched(concrete_model)
        rhs = get_component_replaceability_model(layer)
    
        return get_epsilon(lhs, rhs, **kwargs)

    return {
        "get_epsilon": get_epsilon,
        "get_prefix_equivalence_eps": get_prefix_equivalence_eps, 
        "get_component_equivalence_eps": get_component_equivalence_eps, 
        "get_prefix_replaceability_eps": get_prefix_replaceability_eps, 
        "get_component_replaceability_eps": get_component_replaceability_eps,
    }

def render_graphviz(
    m, 
    path=None,
    classifier=True,
    config=None,
    truth_table_inputs=None,
    feature_names_truth_tables=None,
    **kwargs
):    
    label_names = ["Low Activation", "High Activation"]
    dtree_graph = graphviz.Digraph()
    tree = m.tree_
    
    def node_to_str(node_id):
        is_leaf = tree.children_left[node_id] == _tree.TREE_LEAF
        if is_leaf:
            if classifier:
                return label_names[tree.value[node_id][0].argmax(axis=-1)]
                
            return f"{tree.value[node_id][0][0] :.3g}"
            
        return feature_names_truth_tables[tree.feature[node_id]]
    
    def recurse(node_id=0, root=True):
        if root:
            dtree_graph.node(f"Node{node_id}", node_to_str(node_id))
            
        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]
        if left_child != _tree.TREE_LEAF:
            dtree_graph.node(f"Node{left_child}", node_to_str(left_child))
            dtree_graph.edge(f"Node{node_id}", f"Node{left_child}", label="False" if root else None)
            recurse(left_child, root=False)
        if right_child != _tree.TREE_LEAF:
            dtree_graph.node(f"Node{right_child}", node_to_str(right_child))
            dtree_graph.edge(f"Node{node_id}", f"Node{right_child}", label="True" if root else None)
            recurse(right_child, root=False)
    
    recurse()
    if path is not None:
        filename, _, format = path.partition(".")
        dtree_graph.render(filename=filename, format=format)
        
    return dtree_graph