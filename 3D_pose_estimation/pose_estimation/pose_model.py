import tarfile
from pathlib import Path
import torch
import numpy as np
import openvino as ov

class PoseModel:
    """
        모델 초기화 설정.
        :param model_dir: 모델 디렉토리 경로
        :param ckpt_name: PyTorch 체크포인트 파일 이름
    """
    def __init__(self, model_dir="model", ckpt_name="human-pose-estimation-3d-0001.pth"):
        self.model_dir = Path(model_dir)
        self.ckpt_file = self.model_dir / ckpt_name                    # PyTorch 모델 체크포인트 경로
        self.model_path = self.model_dir / "human-pose-estimation-3d-0001.xml"  # OpenVINO IR 모델 경로
        self.core = ov.Core()                                          # OpenVINO Core 객체 생성
        self.compiled_model = None                                     # 컴파일된 모델 객체 저장용
        self.model = None                                              # 원시 OpenVINO 모델
        self.stride = 8                                                # 모델 입력 stride 설정

    def prepare_model(self):
        """
            모델을 준비합니다.
            - XML(OpenVINO IR) 모델이 없을 경우 PyTorch 모델을 로드하고 변환합니다.
            - 모델을 OpenVINO 형식으로 저장합니다.
        """
        
        if not self.model_path.exists():
            from model.model import PoseEstimationWithMobileNet
            pose_estimation_model = PoseEstimationWithMobileNet(is_convertible_by_mo=True)
            pose_estimation_model.load_state_dict(torch.load(self.ckpt_file, map_location="cpu"))
            pose_estimation_model.eval()
             # PyTorch → OpenVINO 변환
            with torch.no_grad():
                ov_model = ov.convert_model(
                    pose_estimation_model, example_input=torch.zeros([1, 3, 256, 448])
                )
                ov.save_model(ov_model, self.model_path)
        self.model = self.core.read_model(self.model_path)

    def compile_model(self, device="CPU"):
        """
            모델을 지정한 디바이스에 컴파일합니다.
            :param device: 예) "CPU", "GPU", "AUTO"
        """
        self.compiled_model = self.core.compile_model(model=self.model, device_name=device)

    def infer(self, image):
        """
        추론을 수행합니다.
        :param image: 입력 이미지 (numpy array, HWC 형식)
        :return: 모델의 3가지 출력 (heatmap, paf, affinity 등)
        """
        img = image[
            0 : image.shape[0] - (image.shape[0] % self.stride),
            0 : image.shape[1] - (image.shape[1] % self.stride),
        ]
        img = (img - 128.0) / 255.0
        img = np.transpose(img, (2, 0, 1))[None]
        result = self.compiled_model(img)
        return result[0][0], result[1][0], result[2][0]