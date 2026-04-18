import torch
import utils.rotation_conversions as geometry

from stage2_old.common.restored.restored_space import gender_id_to_name
from model.smpl import JOINTSTYPE_ROOT, SMPLX


class RestoredBodyModelForward:
    """
    Shape-aware SMPL-X forward that preserves pair-space translations.

    Unlike Rotation2xyz_x, this utility does not re-center translations to the first
    frame. That is required once stage2 switches from canonical clip space to restored
    pair space, where actor/reactor absolute relative placement matters.
    """

    def __init__(self, body_model="smplx", pose_rep="rot6d", translation=True, glob=True, device="cpu"):
        self.body_model = body_model
        self.pose_rep = pose_rep
        self.translation = translation
        self.glob = glob
        self.device = torch.device(device)
        self._smplx_models = {}

    def to(self, device):
        self.device = torch.device(device)
        for model in self._smplx_models.values():
            model.to(self.device)
        return self

    def _get_smplx_model(self, gender_name):
        gender_name = gender_id_to_name(gender_name)
        if gender_name not in self._smplx_models:
            self._smplx_models[gender_name] = SMPLX(gender=gender_name).eval().to(self.device)
        return self._smplx_models[gender_name]

    def _ensure_inputs(self, motion, betas=None, gender_id=None, mask=None):
        if motion.dim() != 4:
            raise ValueError("motion must be [B, J, F, T]")
        device = motion.device
        self.to(device)
        batch, _, _, num_frames = motion.shape
        if mask is None:
            mask = torch.ones(batch, num_frames, device=device, dtype=torch.bool)
        else:
            mask = mask.to(device=device, dtype=torch.bool)
        if betas is None:
            betas = torch.zeros(batch, 10, device=device, dtype=motion.dtype)
        else:
            betas = betas.to(device=device, dtype=motion.dtype)
        if gender_id is None:
            gender_id = torch.zeros(batch, device=device, dtype=torch.long)
        else:
            gender_id = gender_id.to(device=device, dtype=torch.long).view(-1)
        return mask, betas, gender_id

    def _rotations_from_motion(self, motion):
        if self.pose_rep != "rot6d":
            raise NotImplementedError("RestoredBodyModelForward currently supports pose_rep=rot6d only.")
        x_rotations = motion[:, :-1].permute(0, 3, 1, 2).contiguous()
        batch, num_frames, num_joints, _ = x_rotations.shape
        rotations = geometry.rotation_6d_to_matrix(x_rotations.reshape(batch * num_frames, num_joints, 6))
        global_orient = rotations[:, 0]
        rotations = rotations[:, 1:]
        body_pose = rotations[:, 0:21]
        jaw_pose = rotations[:, 21:22]
        leye_pose = rotations[:, 22:23]
        reye_pose = rotations[:, 23:24]
        left_hand_pose = rotations[:, 24:39]
        right_hand_pose = rotations[:, 39:54]
        return {
            "body_pose": body_pose,
            "jaw_pose": jaw_pose,
            "leye_pose": leye_pose,
            "reye_pose": reye_pose,
            "left_hand_pose": left_hand_pose,
            "right_hand_pose": right_hand_pose,
            "global_orient": global_orient,
        }

    def motion_to_xyz(
        self,
        motion,
        jointstype="vertices",
        betas=None,
        gender_id=None,
        mask=None,
        body_model_type=None,
    ):
        if isinstance(body_model_type, (list, tuple)):
            body_model_type = body_model_type[0] if body_model_type else None
        if body_model_type is not None and str(body_model_type).lower() != "smplx":
            raise NotImplementedError("RestoredBodyModelForward currently supports body_model_type=smplx only.")
        if self.body_model != "smplx":
            raise NotImplementedError("RestoredBodyModelForward currently supports body_model=smplx only.")

        mask, betas, gender_id = self._ensure_inputs(motion, betas=betas, gender_id=gender_id, mask=mask)
        batch, _, _, num_frames = motion.shape
        translations = motion[:, -1, :3, :].permute(0, 2, 1).reshape(batch * num_frames, 3)
        bt_index = torch.arange(batch, device=motion.device).view(-1, 1).expand(batch, num_frames).reshape(-1)
        betas_bt = betas.index_select(0, bt_index)
        gender_bt = gender_id.index_select(0, bt_index)

        rot_parts = self._rotations_from_motion(motion)
        out_flat = None
        for gid in torch.unique(gender_bt, sorted=True).tolist():
            select = torch.nonzero(gender_bt == gid, as_tuple=False).flatten()
            if select.numel() == 0:
                continue
            model = self._get_smplx_model(gid)
            model_out = model(
                betas=betas_bt.index_select(0, select),
                body_pose=rot_parts["body_pose"].index_select(0, select),
                left_hand_pose=rot_parts["left_hand_pose"].index_select(0, select),
                right_hand_pose=rot_parts["right_hand_pose"].index_select(0, select),
                global_orient=rot_parts["global_orient"].index_select(0, select),
                return_verts=True,
            )
            joints = model_out[jointstype]
            if out_flat is None:
                out_flat = torch.zeros(
                    batch * num_frames,
                    joints.shape[1],
                    3,
                    device=motion.device,
                    dtype=motion.dtype,
                )
            out_flat.index_copy_(0, select, joints.to(dtype=motion.dtype))

        if out_flat is None:
            raise RuntimeError("Failed to run restored body-model forward.")

        out_flat = out_flat + translations[:, None, :]
        out_bt = out_flat.view(batch, num_frames, out_flat.shape[1], 3)
        if jointstype != "vertices":
            root_index = JOINTSTYPE_ROOT[jointstype]
            out_bt = out_bt - out_bt[:, :, root_index : root_index + 1, :]
            out_bt = out_bt + translations.view(batch, num_frames, 1, 3)

        if mask is not None:
            out_bt = out_bt * mask[:, :, None, None].to(dtype=out_bt.dtype)

        return out_bt.permute(0, 2, 3, 1).contiguous()
