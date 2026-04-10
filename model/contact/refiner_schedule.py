class RefinerWindowSchedule:
    def __init__(
        self,
        teacher_stage_ratio=0.3,
        mix_stage_ratio=0.4,
        predict_stage_ratio=0.3,
        mix_mode="per_sample",
        tol=1e-6,
    ):
        total = float(teacher_stage_ratio + mix_stage_ratio + predict_stage_ratio)
        if abs(total - 1.0) > tol:
            raise ValueError("Stage ratios must sum to 1.0")
        self.teacher_stage_ratio = float(teacher_stage_ratio)
        self.mix_stage_ratio = float(mix_stage_ratio)
        self.predict_stage_ratio = float(predict_stage_ratio)
        self.mix_mode = mix_mode

    def get_state(self, step, total_steps):
        if total_steps <= 1:
            return {
                "stage": "predict",
                "teacher_ratio": 0.0,
                "predict_ratio": 1.0,
            }

        progress = float(step) / float(max(total_steps - 1, 1))
        teacher_end = self.teacher_stage_ratio
        mix_end = teacher_end + self.mix_stage_ratio

        if progress < teacher_end or self.mix_stage_ratio == 0.0:
            return {
                "stage": "teacher",
                "teacher_ratio": 1.0,
                "predict_ratio": 0.0,
            }

        if progress < mix_end:
            mix_progress = (progress - teacher_end) / max(self.mix_stage_ratio, 1e-6)
            mix_progress = min(max(mix_progress, 0.0), 1.0)
            return {
                "stage": "mix",
                "teacher_ratio": 1.0 - mix_progress,
                "predict_ratio": mix_progress,
            }

        return {
            "stage": "predict",
            "teacher_ratio": 0.0,
            "predict_ratio": 1.0,
        }
