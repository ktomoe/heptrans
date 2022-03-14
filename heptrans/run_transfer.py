from multiml import Saver, StoreGate
from multiml.agent import GridSearchAgent
from multiml.task.pytorch import PytorchBaseTask

from modules import TransferModel
from tasks import MyPytorchTask
from callbacks import get_dgl
import yaml

##############################################################################
# configuration
##############################################################################
yml = yaml.load(open('config.yml'), Loader=yaml.FullLoader)

sg_args = yml['sg_args_a']

task_args = yml['task_args']
task_args['num_epochs'] = 10
task_args['num_workers'] = 4
task_args['model'] = TransferModel
task_args['dataset_args'] = dict(callbacks=[get_dgl])
task_args['max_patience'] = 100

agent_args = yml['agent_args']

agent_args['num_workers'] = 16
agent_args['num_trials'] = 3

task_hps = dict(
    load_weights = ['./weights/ttbar_2hdm425-325_10000000'],
    data_id=[
        'heptrans_znunu_gogo_500', # 500
        #'heptrans_ttbar_2hdm500-400_500',
        #'heptrans_ttbb_tth_500',
        #'heptrans_ttbar_zprime1000_500', 
        #'heptrans_znunu_gogo_5000', # 5000
        #'heptrans_ttbar_2hdm500-400_5000',
        #'heptrans_ttbb_tth_5000',
        #'heptrans_ttbar_zprime1000_5000',
        #'heptrans_znunu_gogo_50000', # 50000
        #'heptrans_ttbar_2hdm500-400_50000',
        #'heptrans_ttbb_tth_50000',
        #'heptrans_ttbar_zprime1000_50000',
        #'heptrans_znunu_gogo_500000', # 500000
        #'heptrans_ttbar_2hdm500-400_500000',
        #'heptrans_ttbb_tth_500000',
        #'heptrans_ttbar_zprime1000_500000',
    ],
    model__features = ['gnn', 'gat'],
    transfer = ['N', 'F', 'T'],
    model__layers = [6], # [5, 6, 7, 8],
    model__nodes = [256], # [128, 256, 512, 1024],
    model__num_heads = [4], # [2, 4, 8, 16],
)

##############################################################################

if __name__ == "__main__":
    sg = StoreGate(**sg_args)
    sv = Saver(save_dir=yml['save_dir'], mode='zarr')

    task = MyPytorchTask(**task_args)
    agent = GridSearchAgent(storegate=sg,
                            saver=sv,
                            task_scheduler=[[(task, task_hps)]],
                            **agent_args)
    agent.execute_finalize()
