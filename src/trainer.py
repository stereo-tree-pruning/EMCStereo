from stereo.modeling.trainer_template import TrainerTemplate
from .emc_stereo import EMCStereo

__all__ = {
    'EMCStereo': EMCStereo,
}


class Trainer(TrainerTemplate):
    def __init__(self, args, cfgs, local_rank, global_rank, logger, tb_writer):
        model = __all__[cfgs.MODEL.NAME](cfgs.MODEL)
        super().__init__(args, cfgs, local_rank, global_rank, logger, tb_writer, model)
