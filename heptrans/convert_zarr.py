from multiml.storegate import StoreGate
import yaml
import numpy as np

##############################################################################
# configuration
##############################################################################
yml = yaml.load(open('config.yml'), Loader=yaml.FullLoader)
var_name0 = yml['var_names'][0]
var_name1 = yml['var_names'][1]

# data_id, processes, target_events, num_nodes
outputs = [
    ['heptrans_source', ['ttbar', '2hdm425-325'], [10000000], 6],                  # ttbar, 2hdm
    ['heptrans_target', ['ttbar', '2hdm500-400'], [500, 5000, 50000, 500000], 6],  # ttbar, 2hdm
    ['heptrans_target', ['ttbar', 'zprime1000'],  [500, 5000, 50000, 500000], 6],  # ttbar, zprime
    ['heptrans_target', ['ttbb', 'tth'],          [500, 5000, 50000, 500000], 8],  # ttbb, tth
    ['heptrans_target', ['znunu', 'gogo'],        [500, 5000, 50000, 500000], 5],  # znunu, gogo
]

##############################################################################

def reshape_data(data, num_nodes):
    num_batch = data.shape[0]
    data = data.reshape(num_batch, 8, 5)
    data = data[:, :num_nodes]

    data[:, :, 0] = np.ma.log10(data[:, :, 0]).filled(0.) # pt
    data[:, :, 3] = np.ma.log10(data[:, :, 3]).filled(0.) # mass

    return data

def fill(sg, data_id_org, processes, max_event, num_nodes):
    data_id = f'heptrans_{processes[0]}_{processes[1]}_{max_event}'

    for phase in ('train', 'test', 'valid'):
        if phase != 'train':
            max_event = 50000

        sg.set_data_id(data_id_org)
        data0 = sg.get_data(processes[0], phase)[:max_event]
        data1 = sg.get_data(processes[1], phase)[:max_event]

        data0 = reshape_data(data0, num_nodes)
        data1 = reshape_data(data1, num_nodes)

        label0 = np.zeros(len(data0), dtype='i8')
        label1 = np.ones(len(data1), dtype='i8')

        sg.set_data_id(data_id)
        sg.delete_data(var_name0, phase)
        sg.delete_data(var_name1, phase)

        sg.add_data(var_name0, data0, phase)
        sg.add_data(var_name0, data1, phase)
        sg.add_data(var_name1, label0, phase)
        sg.add_data(var_name1, label1, phase)

    sg.compile()
    sg.shuffle()

    sg.show_info()

if __name__ == "__main__":
    sg = StoreGate(**yml['sg_args_a'])

    for data_id, output, max_events, num_nodes in outputs:
        sg.set_data_id(data_id)
        sg.compile()

        for max_event in max_events:
            fill(sg, data_id, output, max_event, num_nodes)
