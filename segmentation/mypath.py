class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return '/dltraining/datasets/VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
