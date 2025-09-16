import argparse

import matplotlib.pyplot as plt
import torch

import training.tests.utils
import training.models.ftm_tabicl_v0
import training.models.layers.set_transformer


parser = argparse.ArgumentParser()
parser.add_argument("n_steps", type=int, help="Max number of training steps")
args = parser.parse_args()


class config:
    lr = 1e-3
    n_steps = args.n_steps
    atol = 1e-3

    def __init__(self):
        self.batch_size = 3
        self.n_cols = 3
        self.n_ctx = 6
        self.n_q = 2

        self.n_classes = 10
        self.embed_dim = 16

        self.col_n_inducing_points = 8
        self.col_n_blocks = 2
        self.col_n_heads = 2
        self.col_ff_dim = 24

        self.row_n_blocks = 2
        self.row_n_heads = 4
        self.row_ff_dim = 24
        self.row_max_cols = 1000
        self.row_rope_base = 10_000
        self.row_n_cls_tokens = 4

        self.icl_n_blocks = 2
        self.icl_n_heads = 8
        self.icl_ff_dim = 24
        self.clf_ff_dim = 24

    def new_model(self):
        return training.models.ftm_tabicl_v0.TabICLColumnEmbedding(
            n_blocks=self.col_n_blocks,
            n_inducing_points=self.col_n_inducing_points,
            embed_dim=self.embed_dim,
            n_heads=self.col_n_heads,
            ff_dim=self.col_ff_dim,
        )
        # return training.models.ftm_tabicl_v0.TabICL(
        # n_classes=self.n_classes,
        # embed_dim=self.embed_dim,
        # col_n_inducing_points=self.col_n_inducing_points,
        # col_n_blocks=self.col_n_blocks,
        # col_n_heads=self.col_n_heads,
        # col_ff_dim=self.col_ff_dim,
        # row_n_blocks=self.row_n_blocks,
        # row_n_heads=self.row_n_heads,
        # row_ff_dim=self.row_ff_dim,
        # row_max_cols=self.row_max_cols,
        # row_rope_base=self.row_rope_base,
        # row_n_cls_tokens=self.row_n_cls_tokens,
        # icl_n_blocks=self.icl_n_blocks,
        # icl_n_heads=self.icl_n_heads,
        # icl_ff_dim=self.icl_ff_dim,
        # clf_ff_dim=self.clf_ff_dim,
        # )

    # def make_y(self, n):
    # return torch.testing.make_tensor(
    # self.batch_size,
    # n,
    # dtype=torch.long,
    # low=0,
    # high=self.n_classes - 1,
    # device="cpu",
    # )

    # def make_train_inputs(self):
    # x_ctx = training.tests.utils.float_tensor_cpu(
    # self.batch_size, self.n_ctx, self.n_cols
    # )
    # y_ctx = self.make_y(self.n_ctx)
    # x_q = training.tests.utils.float_tensor_cpu(
    # self.batch_size, self.n_q, self.n_cols
    # )
    # return x_ctx, y_ctx, x_q


# x_ctx, y_ctx, x_q = cfg.make_train_inputs()
# target = cfg.make_y(cfg.n_q)
# model_inputs = dict(x_ctx=x_ctx, y_ctx=y_ctx, x_q=x_q)

cfg = config()
model = cfg.new_model()


x = training.tests.utils.float_tensor_cpu(cfg.batch_size, cfg.n_ctx, cfg.n_cols)
target = training.tests.utils.float_tensor_cpu(
    cfg.batch_size, cfg.n_ctx, cfg.n_cols, cfg.embed_dim
)
model_inputs = dict(x=x, q_mask=None)

optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)


def loss_fn(output, target):
    b, n_q, n_cls = output.shape
    output = output.view(-1, n_cls)
    target = target.view(-1)
    loss = torch.nn.functional.cross_entropy(output, target, reduction="none")
    loss = loss.view(b, n_q)
    return loss.mean(0).mean()


loss_fn = torch.nn.functional.mse_loss

output, losses, metrics = training.tests.utils.train_model_to_memorize(
    model,
    optimizer,
    loss_fn=loss_fn,
    model_inputs=model_inputs,
    target=target,
    max_steps=cfg.n_steps,
    atol=cfg.atol,
    metric_fn=loss_fn,
    _pbar=True,
)

# print(torch.allclose(output, target.to(output.dtype), atol=1e-5, rtol=1))

fig, ax = plt.subplots(1)
ax.plot(losses, label="Train Loss")
ax.legend()
ax1 = ax.twinx()
ax1.plot(metrics, color="C1", label="Overfit Metric")
ax1.axhline(cfg.atol, color="k", ls="--")
plt.show()
