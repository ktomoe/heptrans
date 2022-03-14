from multiml import StoreGate
import numpy as np
import yaml

##############################################################################
# configuration
##############################################################################
yml = yaml.load(open('config.yml'), Loader=yaml.FullLoader)

##############################################################################

def add_data(sg, process, phase):
    data_path = f'{yml["data_path"]}/{process}.npy'
    data = np.load(data_path).astype('f4')
    data = data[:sum(phase)]
    sg.add_data(process, data, phase=phase)


if __name__ == "__main__":
    sg_args = yml['sg_args_w']
    sg_args['data_id'] = 'heptrans_source'
    sg = StoreGate(**sg_args)

    # source data
    for process in yml['source_domains']:
        add_data(sg, process, yml['source_phase'])        

    # target data
    sg.set_data_id('heptrans_target')
    for process in yml['target_domains']:
        add_data(sg, process, yml['target_phase'])        

    sg.compile(show_info=True)
