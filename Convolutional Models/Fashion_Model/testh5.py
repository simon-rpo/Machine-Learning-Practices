import h5py


ftrain = h5py.File(
    'C:\\Users\\PC\\Downloads\\test_Conv\\animal_data\\train_animals.h5', 'r')
ftest = h5py.File(
    'C:\\Users\\PC\\Downloads\\test_Conv\\animal_data\\test_animals.h5', 'r')

train_data, train_labels = ftest['test_set_x'], ftest['test_set_y']
eval_data, eval_labels = ftrain['train_set_x'],  ftrain['train_set_y']


