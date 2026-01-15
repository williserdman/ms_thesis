AT_ONE = 0.9
LAP_FILTER_LBOUND = 0.0001
LAP_FILTER_UBOUND = 2


class SimplifiedArgs:
    """Lightweight container used for building MyArgs in this module."""

    def __init__(
        self,
        sampling_method: str,
        my_filter: str,
        lr: float,
        dropout_rate: float,
        hidden_dim: int,
    ):
        self.sampling_method = sampling_method
        self.my_filter = my_filter
        self.lr = lr
        self.dropout_rate = dropout_rate
        self.hidden_dim = hidden_dim

    def __repr__(self):
        return (
            f"SimplifiedArgs(sampling_method={self.sampling_method}, my_filter={self.my_filter}, "
            f"lr={self.lr}, dropout_rate={self.dropout_rate}, hidden_dim={self.hidden_dim})"
        )


class MyArgs:
    def __init__(self, args: SimplifiedArgs):
        use_laplacian = (
            False if args.my_filter in {"g_0", "g_1", "g_2", "g_3"} else True
        )
        name = args.my_filter
        self.K = 10
        self.alpha = 0.1
        self.Init = self.ArnoldiInit = self.e = args.sampling_method
        self.nameFunc = self.FuncName = name
        self.homophily = (
            not use_laplacian
        )  # DO NOT CHANGE, homophily is the determining argument
        self.Vandermonde = False
        self.lower = LAP_FILTER_LBOUND if use_laplacian else -AT_ONE  # 0.000001
        self.upper = LAP_FILTER_UBOUND if use_laplacian else AT_ONE  # 2.0
        self.Arnoldippnp = "GArnoldi_prop"
        self.Gamma = None
        self.dprate = self.dropout = args.dropout_rate
        self.hidden = self.hidden_dim = args.hidden_dim

    def update_filter(self, f: str):
        self.nameFunc = self.FuncName = f
        self.use_laplacian = False if f in {"g_0", "g_1", "g_2", "g_3"} else True
        # may not need to set these?
        self.lower = LAP_FILTER_LBOUND if self.use_laplacian else -AT_ONE  # 0.000001
        self.upper = LAP_FILTER_UBOUND if self.use_laplacian else AT_ONE  # 2.0
