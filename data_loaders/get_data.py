from torch.utils.data import DataLoader
from data_loaders.tensors import collate as all_collate
from data_loaders.tensors import ccollate as all_ccollate

def get_dataset_class(name):
    if name in ["chi3d", "interx"]:
        from .a2m.feeder import Feeder
        return Feeder
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')


def get_collate_fn(name, setting, hml_mode='train'):
    if setting == 'mdm':
        return all_collate
    elif setting in ['cmdm', 'cnet_v1', 'cnet_v2', 'cnet_v3', 'cnet_v4', 'cnet_v5', 'cnet_v5_actor_bodyhand', 'cnet_v5_actor_globalonly', 'cnet_v5_reactor_singlestream']:
        return all_ccollate


def get_dataset(
    name,
    num_frames,
    num_person,
    data_path='',
    pose_rep='rot6d',
    body_model='smpl',
    ar_shuffle=False,
    split='train',
    hml_mode='train',
    shard=0,
    num_shards=1,
    **dataset_kwargs,
):
    DATA = get_dataset_class(name)
    dataset = DATA(
        split=split,
        num_frames=num_frames,
        num_person=num_person,
        datapath=data_path,
        pose_rep=pose_rep,
        dataname=name,
        body_model=body_model,
        ar_shuffle=ar_shuffle,
        shard=shard,
        num_shards=num_shards,
        **dataset_kwargs,
    )
    return dataset


def get_dataset_loader(
    name,
    batch_size,
    num_frames,
    num_person,
    data_path='',
    pose_rep='rot6d',
    body_model='smpl',
    ar_shuffle=False,
    setting='mdm',
    split='train',
    hml_mode='train',
    shard=0,
    num_shards=1,
    **dataset_kwargs,
):
    dataset = get_dataset(
        name,
        num_frames,
        num_person,
        data_path,
        pose_rep,
        body_model,
        ar_shuffle,
        split,
        hml_mode,
        shard=shard,
        num_shards=num_shards,
        **dataset_kwargs,
    )
    collate = get_collate_fn(name, setting, hml_mode)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=8, drop_last=True, collate_fn=collate, persistent_workers=True
    )

    return loader
