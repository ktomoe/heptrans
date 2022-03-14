from multiml import StoreGate
from multiml.agent import GridSearchAgent

from modules import TransferModel
from tasks import MyPytorchTask
from callbacks import get_dgl
import yaml

##############################################################################
# configuration
##############################################################################
yml = yaml.load(open('config.yml'), Loader=yaml.FullLoader)

sg_args = yml['sg_args_a']
sg_args['data_id'] = 'heptrans_ttbar_2hdm425-325_10000000'

task_args = yml['task_args']
task_args['num_epochs'] = 1
task_args['batch_size'] = 2048
task_args['model'] = TransferModel
task_args['dataset_args'] = dict(callbacks=[get_dgl])
task_args['save_weights'] = './weights/ttbar_2hdm425-325_10000000'

agent_args = yml['agent_args']
agent_args['num_workers'] = 2
agent_args['num_trials'] = 1

task_hps = dict(
    model__features = ['gnn', 'gat'],
    model__layers = [6], # [5, 6, 7, 8],
    model__nodes = [256], # [128, 256, 512, 1024],
    model__num_heads = [4], # [2, 4, 8, 16],
)

##############################################################################

if __name__ == "__main__":
    sg = StoreGate(**sg_args) 
    sg.show_info()

    task = MyPytorchTask(**task_args)
    agent = GridSearchAgent(storegate=sg,
                            saver=yml['save_dir'],
                            task_scheduler=[[(task, task_hps)]],
                            **agent_args)
    agent.execute_finalize()
