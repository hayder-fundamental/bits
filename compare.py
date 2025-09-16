import os

import pandas as pd

import training.models.ftm_tabicl_v0
import training.models.tabicl_v0

tabicl_params = {
    "n_classes": 10,  # data?
    "embed_dim": 128,  # paper
    "col_n_inducing_points": 128,  # paper
    "col_n_blocks": 3,  # paper
    "col_n_heads": 4,  # paper
    "col_ff_dim": 256,  # code': 2 * embed_dim
    "row_n_blocks": 3,  # paper
    "row_n_heads": 8,  # paper
    "row_ff_dim": 256,  # code': 2 * embed_dim
    "row_max_cols": 1000,  # data?
    "row_rope_base": 100_000,  # paper
    "row_n_cls_tokens": 4,  # paper
    "icl_n_blocks": 12,  # paper
    "icl_n_heads": 4,  # paper
    "icl_ff_dim": 1024,  # ff_factor * n_cls_tokens * embed_dim
    "clf_ff_dim": 1024,  # ff_factor * n_cls_tokens * embed_dim
    "no_query_attn": True,
}

our_block_names = [
    "column_embedding.isab_blocks",
    "row_interaction.transformer_blocks",
    "transformer_blocks",
]

# ordering should match above
their_block_names = [
    "col_embedder.tf_col.blocks",
    "row_interactor.tf_row.blocks",
    "icl_predictor.tf_icl.blocks",
]


def nparams(model):
    named = {}
    total = 0
    for n, p in model.named_parameters():
        count = p.nelement()
        named[n] = count
        total += count
    return total, pd.Series(named, name="count").rename(index="module")


def group_block(series, name):
    selection = series[series.index.str.startswith(name)]

    def clean_index(s):
        s.index = s.index.map(lambda x: x.replace(name, "").split(".", maxsplit=2)[-1])
        return s

    def block_id(key):
        return key.replace(name, "").split(".")[1]

    return pd.concat(
        {k: clean_index(v) for k, v in selection.groupby(block_id)},
        axis=1,
    )


def check_blocks_consistent(data: pd.Series, block_names: list[str]):
    for block_name in block_names:
        col = group_block(data, block_name)
        assert col.nunique(axis=1).eq(1).all(), f"Inconsistent counts in {block_name}."


def take_counts(srs, block_name):
    return group_block(srs, block_name).iloc[:, 0].to_frame().reset_index(drop=False)


if __name__ == "__main__":
    mine = training.models.ftm_tabicl_v0.TabICL(**tabicl_params)
    their = training.models.tabicl_v0.TabICL()

    our_total, our_c = nparams(mine)
    their_total, their_c = nparams(their)

    delta = our_total - their_total

    def banner():
        print("=" * 15)

    banner()
    banner()
    print("PARAM COUNTS")
    banner()
    print("Total param count:")
    print(f"our: {our_total}")
    print(f"their: {their_total}")
    print(f"delta: {delta}")

    our_ln_c = our_c[our_c.index.str.contains("ln_")]
    their_ln_c = their_c[their_c.index.str.contains("ln|norm")]
    ln_delta = our_ln_c.sum() - their_ln_c.sum()
    their_embed_bias_c = their_c["icl_predictor.y_encoder.bias"]
    print(f"Our layer norm params: {our_ln_c.sum()}")
    print(f"Their layer norm params: {their_ln_c.sum()}")
    print(f"LayerNorm delta: {ln_delta}")

    print(f"Remaining delta after Layer Norm {delta - ln_delta}")
    print(f"Their embedding bias count {their_embed_bias_c.sum()}")

    their_rope_c = their_c[their_c.index.str.contains("rope")]
    print(f"Their rope freqs count {their_rope_c.sum()}")

    banner()
    banner()
    print(
        f"Total delta after accounting for:"
        f"\n - Our extra layer norm params {ln_delta}"
        f"\n - Their extra class embedding bias params {their_embed_bias_c.sum()}"
        f"\n - Their extra rope freq params {their_rope_c.sum()}"
    )
    print(f">>> {delta - ln_delta + their_embed_bias_c + their_rope_c.sum()}")
    banner()
    banner()
    check_blocks_consistent(our_c, our_block_names)
    print("Our blocks have consistent counts")

    check_blocks_consistent(their_c, their_block_names)
    print("Their blocks have consistent counts")
    banner()
    banner()

    os.makedirs("counts", exist_ok=True)
    for our_block_name, their_block_name in zip(our_block_names, their_block_names):
        our_block_c = take_counts(our_c, our_block_name)
        their_block_c = take_counts(their_c, their_block_name)
        table = pd.concat({"ours": our_block_c, "theirs": their_block_c}, axis=1)
        table.to_csv(f"counts/{our_block_name}.csv")

    our_other_c = (
        our_c[~our_c.index.str.startswith(tuple(our_block_names))]
        .to_frame()
        .reset_index(drop=False)
    )
    their_other_c = (
        their_c[~their_c.index.str.startswith(tuple(their_block_names))]
        .to_frame()
        .reset_index(drop=False)
    )
    table = pd.concat({"ours": our_other_c, "theirs": their_other_c}, axis=1)
    table.to_csv("counts/other_params.csv")

    full_table = pd.concat(
        {
            "ours": our_c.reset_index(drop=False),
            "theirs": their_c.reset_index(drop=False),
        },
        axis=1,
    )
    full_table.to_csv("counts/full_table.csv")
