import SimpleITK as sitk
import numpy as np
import os
from keras.callbacks import Callback

# Put a volume through 3D slicer to see how the mask coincides with the segmentation. 

# volume: the values of the mask
# target_centre refers to the (i,j,k)th slice of the tumour centre in the entire (128,128,128) volume
# target_radius refers to half of the size of the bounding box (in number of slices)
def output_numpy_mask_to_nrrd(patient_name, volume, target_centre, target_radius, dataset, filename_tag=''): 
    
    reference_volume = None
    
    if dataset == 'HeadNeckCancer':
        # there should only be one file here
        reference_file = None
        for root, dirs, files in os.walk(os.path.join('/home/jzhe0882/datasets/LabelMaps-Processed/', patient_name)):
            for name in files:
                reference_file = os.path.join(root, name)
        
        reader = sitk.ImageFileReader()
        reader.SetFileName(reference_file)
        reference_volume = reader.Execute()
                
    elif dataset == 'BreastCancer':
        reader = sitk.ImageSeriesReader()
        files = reader.GetGDCMSeriesFileNames(os.path.join('/home/jzhe0882/datasets/Breast Cancer Scans-Abridged/', patient_name, 'PET_before'))
        reader.SetFileNames(files)
        reference_volume = reader.Execute()
        
    else:
        print('Wrong dataset:', dataset)
        return
        
    if reference_volume is None:
        print('cant find patient', patient_name)
        return
    
    test_volume = sitk.GetImageFromArray(volume)
        
    #tumour_centre is in real-world millimeter coordinates
    #The 'Origin' in the SITK API refers to the position of a corner in the volume, not the centre of the volume
    #The 'Direction' refers to which corner
    #tumour_centre = np.array(reference_volume.GetOrigin()) + target_centre * np.array(reference_volume.GetSpacing())
    #target_origin = tumour_centre - target_radius * np.array(reference_volume.GetSpacing())
    size_ratio = np.divide(reference_volume.GetSize(), [128,128,128]) #the bounding boxes were obtained from resampled volumes of size [128,128,128]
    reference_index = np.multiply(size_ratio, (target_centre - target_radius))
    target_origin = reference_volume.TransformContinuousIndexToPhysicalPoint(reference_index)
        
    test_volume.SetOrigin((target_origin[0], target_origin[1], target_origin[2]))
    
    target_spacing = np.multiply(size_ratio, reference_volume.GetSpacing())
    test_volume.SetSpacing(target_spacing)
    test_volume.SetDirection(reference_volume.GetDirection())
    writer = sitk.ImageFileWriter()
    writer.SetFileName(os.path.join('/home/jzhe0882/SegmentationOutput', '{}{}.nrrd'.format(patient_name, filename_tag)))
    writer.Execute(test_volume)
    
#from https://github.com/keras-team/keras/issues/2768

class GetBest(Callback):
    """Get the best model at the end of training.
    # Arguments
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        mode: one of {auto, min, max}.
            The decision
            to overwrite the current stored weights is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        period: Interval (number of epochs) between checkpoints.
    # Example
        callbacks = [GetBest(monitor='val_acc', verbose=1, mode='max')]
        mode.fit(X, y, validation_data=(X_eval, Y_eval),
                 callbacks=callbacks)
    """

    def __init__(self, monitor='val_loss', verbose=0,
                 mode='auto', period=1):
        super(GetBest, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.period = period
        self.best_epochs = 0
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('GetBest mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf
                
    def on_train_begin(self, logs=None):
        self.best_weights = self.model.get_weights()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            #filepath = self.filepath.format(epoch=epoch + 1, **logs)
            current = logs.get(self.monitor)
            if current is None:
                warnings.warn('Can pick best model only with %s available, '
                              'skipping.' % (self.monitor), RuntimeWarning)
            else:
                if self.monitor_op(current, self.best):
                    if self.verbose > 0:
                        print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                              ' storing weights.'
                              % (epoch + 1, self.monitor, self.best,
                                 current))
                    self.best = current
                    self.best_epochs = epoch + 1
                    self.best_weights = self.model.get_weights()
                else:
                    if self.verbose > 0:
                        print('\nEpoch %05d: %s did not improve' %
                              (epoch + 1, self.monitor))            
                    
    def on_train_end(self, logs=None):
        if self.verbose > 0:
            print('Using epoch %05d with %s: %0.5f' % (self.best_epochs, self.monitor,
                                                       self.best))
        self.model.set_weights(self.best_weights)