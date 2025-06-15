import numpy as np
import glob
import os

from Adafold.dataloader.dataloader import PointcloudDataset

def make_dir(path):
    tot_path = ''
    for folder in path.split('/'):
        if not folder == '.' and not folder == '':
            tot_path = tot_path + folder + '/'
            if not os.path.exists(tot_path):
                os.mkdir(tot_path)
                # print(tot_path)
        else:
            if folder == '.':
                tot_path = tot_path + folder + '/'


def get_all_envs(elast=np.arange(20, 70, 5),
                 bend=np.arange(20, 50, 5),
                 scale=np.arange(7,12)*0.01,
                 frame_skip=np.asarray([2, 5, 8])):
    params = []
    for e in elast:
        for b in bend:
            for s in scale:
                for f in frame_skip:
                    params.append([e, b, s, f])
    return params



def load_datasets(args,
                  checkpoint_folder=None,
                  num_folders_train=-1,
                  num_folders_val=-1,
                  num_folders_test=0,
                  num_envs=-1,
                  num_trajectories=-1,
                  dataload=PointcloudDataset,
                  train=True,
                  sort=True):

    dataset_train = None
    dataset_test = None
    dataset_val = None
    if num_folders_train != 0:
        print("Loading training data")
        dataset_names = glob.glob(args.dataset_path + args.dataset_name + f'/*')
        names = list(set([n.split('/')[-1] for n in dataset_names]))
        if sort:
            names.sort()
        if num_envs != -1 and num_envs < len(names):
            names = names[:num_envs]
        if num_trajectories == -1:
            full_folders = [glob.glob(args.dataset_path + args.dataset_name + f'/' + names[i] + f'/*') for i in range(len(names))]
        else:
            full_folders = []
            for i in range(len(names)):
                traj_list = glob.glob(args.dataset_path + args.dataset_name + f'/' + names[i] + f'/*')
                traj_list.sort()
                full_folders.append(traj_list[:num_trajectories])
            # full_folders = [glob.glob(args.dataset_path + args.dataset_name + f'/' + names[i] + f'/*')[:num_trajectories] for i in range(len(names))]
        train_folders = list(set(sum(full_folders, [])))  # f'/train{args.fold}/env*')
        if sort:
            train_folders.sort()
        if num_folders_train != -1 and num_folders_train < len(train_folders):
            train_folders = train_folders[:num_folders_train]
        dataset_train = dataload(train_folders, args, train=train)

    if num_folders_test != 0:
        dataset_names = glob.glob(args.dataset_path + args.dataset_name + f'/*')
        names = list(set([n.split('/')[-1] for n in dataset_names]))
        print(f'Number of folders: {len(names)}')
        if sort:
            names.sort()
        if num_envs != -1 and num_envs < len(names):
            names = names[:num_envs]
        if num_trajectories == -1:
            full_folders = [glob.glob(args.dataset_path + args.dataset_name + f'/' + names[i] + f'/*') for i in range(len(names))]
        else:
            full_folders = []
            for i in range(len(names)):
                traj_list = glob.glob(args.dataset_path + args.dataset_name + f'/' + names[i] + f'/*')
                traj_list.sort()
                full_folders.append(traj_list[:num_trajectories])
            # full_folders = [glob.glob(args.dataset_path + args.dataset_name + f'/' + names[i] + f'/*')[:num_trajectories] for i in range(len(names))]
        test_folders = list(set(sum(full_folders, [])))  # f'/train{args.fold}/env*')
        if sort:
            test_folders.sort()
        if num_folders_test != -1 and num_folders_test < len(test_folders):
            test_folders = test_folders[:num_folders_test]
        dataset_test = dataload(test_folders, args, train=False)

    if args.validation and (num_folders_val != 0):
        print("Loading validation data")
        name = args.dataset_name.replace('train', 'test')
        dataset_names = glob.glob(args.dataset_path + name + f'/*')
        names = list(set([n.split('/')[-1] for n in dataset_names]))
        if sort:
            names.sort()
        if num_envs != -1 and num_envs < len(names):
            names = names[:num_envs]
        if num_trajectories == -1:
            full_folders = [glob.glob(args.dataset_path + name + f'/' + names[i] + f'/*') for i in range(len(names))]
        else:
            full_folders = [glob.glob(args.dataset_path + name + f'/' + names[i] + f'/*')[:num_trajectories]  for i in range(len(names))]
        val_folders = list(set(sum(full_folders, [])))
        if sort:
            val_folders.sort()
        if num_folders_val != -1 and num_folders_val < len(val_folders):
            val_folders = val_folders[:num_folders_val]
        dataset_val = dataload(val_folders, args, train=False)

    return dataset_train, dataset_test, dataset_val


