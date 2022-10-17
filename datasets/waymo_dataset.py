import glob
import os
import os.path as osp
import torch.utils.data as data    
import pickle
from functools import partial
import numpy as np
__all__ = ['WaymoDataset']


def remove_out_of_bounds_points(pc, y, x_min, x_max, y_min, y_max, z_min, z_max):
    # Max needs to be exclusive because the last grid cell on each axis contains
    # [((grid_size - 1) * cell_size) + *_min, *_max).
    #   E.g grid_size=512, cell_size = 170/512 with min=-85 and max=85
    # For z-axis this is not necessary, but we do it for consistency
    mask = (pc[:, 0] >= x_min) & (pc[:, 0] < x_max) \
           & (pc[:, 1] >= y_min) & (pc[:, 1] < y_max) \
           & (pc[:, 2] >= z_min) & (pc[:, 2] < z_max)
    pc_valid = pc[mask]
    y_valid = None
    if y is not None:
        y_valid = y[mask]
    return pc_valid, y_valid


def get_coordinates_and_features(point_cloud, transform=None):
    """
    Parse a point clound into coordinates and features.
    :param point_cloud: Full [N, 9] point cloud
    :param transform: Optional parameter. Transformation matrix to apply
    to the coordinates of the point cloud
    :return: [N, 5] where N is the number of points and 5 is [x, y, z, intensity, elongation]
    """
    points_coord, features, flows = point_cloud[:, 0:3], point_cloud[:, 3:5], point_cloud[:, 5:]
    if transform is not None:
        ones = np.ones((points_coord.shape[0], 1))
        points_coord = np.hstack((points_coord, ones))
        points_coord = transform @ points_coord.T
        points_coord = points_coord[0:-1, :]
        points_coord = points_coord.T
    point_cloud = np.hstack((points_coord, features))
    return point_cloud

class WaymoDataset(data.Dataset):
    """
    Args:
        train (bool): If True, creates dataset from training set, otherwise creates from test set.
        transform (callable):
        gen_func (callable):
        args:
    """

    def __init__(self,
                 train,
                 transform,
                 num_points,
                 data_root,
                 remove_ground = True):
        # self.root = osp.join(data_root, 'KITTI_processed_occ_final')
        # print(self.root)
        #assert train is False
        self.train = train
        self.transform = transform
        # self.num_points = num_points
        self.remove_ground = remove_ground

        # Config parameters
        metadata_path = os.path.join(data_root, 'metadata')
        # It has information regarding the files and transformations

        self.data_path = data_root
        self.ph = None
        
        if self.data_path.startswith("s3://"):
            from petrel_helper import PetrelHelper
            self.ph = PetrelHelper()

        # self._point_cloud_transform = point_cloud_transform

        # This parameter is useful when visualizing, since we need to pass
        # the pillarized point cloud to the model for infer but we would
        # like to display the points without pillarizing them
        # self._apply_pillarization = apply_pillarization

        try:
            if metadata_path.startswith("s3://"):
                metadata_file = self.ph.open(metadata_path, 'rb')
                self.metadata = pickle.load(metadata_file)
            else:
                with open(metadata_path, 'rb') as metadata_file:
                    self.metadata = pickle.load(metadata_file)
        except FileNotFoundError:
            raise FileNotFoundError("Metadata not found, please create it by running preprocess.py")

        self._n_points = num_points
        self.compensate_ego_motion_to_pcl = False
        self._drop_invalid_point_function = self.drop_points_function(x_min=-85,
                                                          x_max=85, y_min=-85, y_max=85,
                                                          z_min=-3, z_max=3)

    def __len__(self):
        return len(self.metadata['look_up_table'])
    
    @staticmethod
    def drop_points_function(x_min, x_max, y_min, y_max, z_min, z_max):
        inner = partial(remove_out_of_bounds_points,
                                            x_min=x_min,
                                            y_min=y_min,
                                            z_min=z_min,
                                            z_max=z_max,
                                            x_max=x_max,
                                            y_max=y_max
                                            )

        return inner
    
    def read_point_cloud_pair(self, index):
        """
        Read from disk the current and previous point cloud given an index
        """
        # In the lookup table entries with (current_frame, previous_frame) are stored
        #print(self.metadata['look_up_table'][index][0][0])
        data_path = os.path.join(self.data_path, self.metadata['look_up_table'][index][0][0])
        data_str = None
        if data_path.startswith("s3://"):
            # np load need the file-like object supports the seek operation
            data_str = io.BytesIO(self.ph.open(data_path, 'rb').read())
        else:
            data_str = open(data_path, 'rb')
        
        current_frame = np.load(data_str)['frame']
        
        
        data_path = os.path.join(self.data_path, self.metadata['look_up_table'][index][1][0])
        data_str = None
        if data_path.startswith("s3://"):
            data_str = io.BytesIO(self.ph.open(data_path, 'rb').read())
        else:
            data_str = open(data_path, 'rb')
        previous_frame = np.load(data_str)['frame']
        return current_frame, previous_frame
    
    def get_pose_transform(self, index):
        """
        Return the frame poses of the current and previous point clouds given an index
        """
        current_frame_pose = self.metadata['look_up_table'][index][0][1]
        previous_frame_pose = self.metadata['look_up_table'][index][1][1]
        return current_frame_pose, previous_frame_pose

    def get_flows(self, frame):
        """
        Return the flows given a point cloud
        """
        flows = frame[:, -4:]
        return flows


    def subsample_points(self, current_frame, previous_frame, flows):
        # current_frame.shape[0] == flows.shape[0]
        if current_frame.shape[0] > self._n_points:
            indexes_current_frame = np.linspace(0, current_frame.shape[0]-1, num=self._n_points).astype(int)
            current_frame = current_frame[indexes_current_frame, :]
            flows = flows[indexes_current_frame, :]
        if previous_frame.shape[0] > self._n_points:
            indexes_previous_frame = np.linspace(0, previous_frame.shape[0]-1, num=self._n_points).astype(int)
            previous_frame = previous_frame[indexes_previous_frame, :]
        return current_frame, previous_frame, flows
    
    def __getitem__(self, index):
        
        # pc1_loaded, pc2_loaded = self.pc_loader(self.samples[index])
        # pc1_transformed, pc2_transformed, sf_transformed = self.transform([pc1_loaded, pc2_loaded])
        # if pc1_transformed is None:
        #     print('path {} get pc1 is None'.format(self.samples[index]), flush=True)
        #     index = np.random.choice(range(self.__len__()))
        #     return self.__getitem__(index)

        # pc1_norm = pc1_transformed
        # pc2_norm = pc2_transformed
        # return pc1_transformed, pc2_transformed, pc1_norm, pc2_norm, sf_transformed, self.samples[index]
        
        
        """
        Return two point clouds, the current point and its previous one. It also
        return the flow per each point of the current cloud

        A point cloud has a shape of [N, F], being N the number of points and the
        F to the number of features, which is [x, y, z, intensity, elongation]
        """
        current_frame, previous_frame = self.read_point_cloud_pair(index)
        current_frame_pose, previous_frame_pose = self.get_pose_transform(index)
        flows = self.get_flows(current_frame)
        
        # Drop invalid points according to the method supplied
        if self._drop_invalid_point_function is not None:
            current_frame, flows = self._drop_invalid_point_function(current_frame, flows)
            previous_frame, _ = self._drop_invalid_point_function(previous_frame, None)
            
        if self.remove_ground:
            cur_valid_inx = current_frame[:, 2] > 0.15
            current_frame = current_frame[cur_valid_inx]
            flows = flows[cur_valid_inx]
            previous_frame = previous_frame[previous_frame[:, 2] > 0.15]
        
        
        if self._n_points is not None:
            current_frame, previous_frame, flows = self.subsample_points(current_frame, previous_frame, flows)

        # G_T_C -> Global_TransformMatrix_Current
        G_T_C = np.reshape(np.array(current_frame_pose), [4, 4])

        # G_T_P -> Global_TransformMatrix_Previous
        G_T_P = np.reshape(np.array(previous_frame_pose), [4, 4])
        C_T_P = np.linalg.inv(G_T_C) @ G_T_P
        # https://github.com/waymo-research/waymo-open-dataset/blob/bbcd77fc503622a292f0928bfa455f190ca5946e/waymo_open_dataset/utils/box_utils.py#L179
        # compensate ego
        if self.compensate_ego_motion_to_pcl:
            previous_frame = get_coordinates_and_features(previous_frame, transform=C_T_P)
        # do not compensate ego
        else:
            C_T_P_inv = np.linalg.inv(C_T_P)
            previous_frame = get_coordinates_and_features(previous_frame, transform=None)
            # retrieve the initial flow
            # flow is the velocity, frame rate = 10hz
            frm_time_interval = 0.10 # unit: s
            flows[:, :3] = (current_frame[:, :3] - get_coordinates_and_features(current_frame[:, :3] - flows[:, :3] * frm_time_interval, transform=C_T_P_inv)) / frm_time_interval
            # note : if u decide not to compensate the flow as the code above, the flow information saving in the metadata is not valid.
            # because in this repo, there is no effect to the the training procedure, we 【do not】 update the data saved in the metadata. 
            
        current_frame = get_coordinates_and_features(current_frame, transform=None)



        # # Perform the pillarization of the point_cloud
        # if self._point_cloud_transform is not None and self._apply_pillarization:
        #     current_frame = self._point_cloud_transform(current_frame)
        #     previous_frame = self._point_cloud_transform(previous_frame)
        # else:
        #     # output must be a tuple
        #     previous_frame = (previous_frame, None)
        #     current_frame = (current_frame, None)
        # # This returns a tuple of augmented pointcloud and grid indices
        
        
        # waymo dataset is so large... so for now we do not add data augmentation
        # pc1_transformed, pc2_transformed, sf_transformed = self.transform([pc1_loaded, pc2_loaded])
        # if pc1_transformed is None:
        #     print('path {} get pc1 is None'.format(self.samples[index]), flush=True)
        #     index = np.random.choice(range(self.__len__()))
        #     return self.__getitem__(index)

        previous_frame = previous_frame[:, :3]
        current_frame = current_frame[:, :3]
        flows = -flows[:, :3]
        
        pc1_norm = np.zeros([self._n_points, 3])
        pc2_norm = np.zeros([self._n_points, 3])
        mask = np.ones([self._n_points])
        return current_frame, previous_frame, pc1_norm, pc2_norm, flows, mask

    
    
    


    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Number of points per point cloud: {}\n'.format(self._n_points)
        fmt_str += '    is removing ground: {}\n'.format(self.remove_ground)
        fmt_str += '    Root Location: {}\n'.format(self.data_path)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))

        return fmt_str


