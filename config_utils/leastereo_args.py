from dataclasses import dataclass


@dataclass
class LEAStereoArgsNoArch:
    fea_num_layers: int = 6
    mat_num_layers: int = 12
    fea_filter_multiplier: int = 8
    mat_filter_multiplier: int = 8
    fea_block_multiplier: int = 4
    mat_block_multiplier: int = 4
    fea_step: int = 3
    mat_step: int = 3


@dataclass
class LEAStereoArgs(LEAStereoArgsNoArch):
    net_arch_fea: str = None
    cell_arch_fea: str = None
    net_arch_mat: str = None
    cell_arch_mat: str = None


def add_leastereo_args_without_arch(parser):
    parser.add_argument('--fea_num_layers', type=int, default=6)
    parser.add_argument('--mat_num_layers', type=int, default=12)
    parser.add_argument('--fea_filter_multiplier', type=int, default=8)
    parser.add_argument('--mat_filter_multiplier', type=int, default=8)
    parser.add_argument('--fea_block_multiplier', type=int, default=4)
    parser.add_argument('--mat_block_multiplier', type=int, default=4)
    parser.add_argument('--fea_step', type=int, default=3)
    parser.add_argument('--mat_step', type=int, default=3)


def add_leastereo_args(parser):
    add_leastereo_args_without_arch(parser)
    parser.add_argument('--net_arch_fea', default=None, type=str)
    parser.add_argument('--cell_arch_fea', default=None, type=str)
    parser.add_argument('--net_arch_mat', default=None, type=str)
    parser.add_argument('--cell_arch_mat', default=None, type=str)
