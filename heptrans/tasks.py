import torch

from multiml import logger, const
from multiml.task.pytorch import PytorchBaseTask


class MyPytorchTask(PytorchBaseTask):
    def __init__(self, transfer='N', **kwargs):
        super().__init__(**kwargs)

        self._transfer = transfer

    def set_hps(self, params):
        super().set_hps(params)

        torch.backends.cudnn.benchmark = True

        if ('features' in self._model_args):
            feature = self._model_args['features']

            if isinstance(self._save_weights, str):
                self._save_weights += '.' + feature

            if isinstance(self._load_weights, str):
                self._load_weights += '.' + feature

                if self._transfer == 'N':
                    self._load_weights = False

                elif self._transfer in ('F', 'T'):
                    self._load_weights += ':features'

        self._storegate.to_memory('features', phase='all')
        self._storegate.to_memory('labels', phase='all')
        self._storegate.set_mode('numpy')
        self._storegate.show_info()

    @logger.logging
    def execute(self):
        """ Execute a task.
        """
        self.compile()

        model_feature = self._model_args['features']
        if self._transfer == 'F':
            self.fix_submodule('features')

        dataloaders = self.prepare_dataloaders()
        result = self.fit(dataloaders=dataloaders, dump=True)

        if 'gat' in model_feature:
            self.ml.model.set_attn(True)

        pred = self.predict(dataloader=dataloaders['test'])
        self.update(data=pred, phase='test')

        if 'gat' in model_feature:
            self._storegate.set_mode('zarr')
            for ii in range(self._model_args['layers']):
                attn_name = f'attn{ii+1}_{model_feature}_{self._transfer}'
                self.storegate.delete_data(attn_name, phase='test')

            self._storegate.set_mode('numpy')
            for ii in range(self._model_args['layers']):
                attn_name = f'attn{ii+1}_{model_feature}_{self._transfer}'
                self.storegate.to_storage(f'attn{ii+1}',
                                          output_var_names=attn_name,
                                          phase='test')

        self._storegate.set_mode('zarr')
        self._storegate.show_info()
        self._storegate.set_mode('numpy')
