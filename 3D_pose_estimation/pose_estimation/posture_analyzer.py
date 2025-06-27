import collections
import time
from IPython.display import display, clear_output
import cv2
import numpy as np
import ipywidgets as widgets

from .pose_model import PoseModel
from pose_estimation.visualizer import Visualizer
import engine3js as engine
import notebook_utils as utils


# 3D 포즈 추정 시 사용되는 연결 구조 (몸의 관절 쌍)
body_edges = np.array(
    [
        [0, 1],
        [0, 9],
        [9, 10],
        [10, 11],  # neck - r_shoulder - r_elbow - r_wrist
        [0, 3],
        [3, 4],
        [4, 5],  # neck - l_shoulder - l_elbow - l_wrist
        [1, 15],
        [15, 16],  # nose - l_eye - l_ear
        [1, 17],
        [17, 18],  # nose - r_eye - r_ear
        [0, 6],
        [6, 7],
        [7, 8],  # neck - l_hip - l_knee - l_ankle
        [0, 12],
        [12, 13],
        [13, 14],  # neck - r_hip - r_knee - r_ankle
    ]
)

class PoseEstimatorApp:
    def __init__(self, source, use_popup=True, use_webcam=False):
        """
        포즈 추정 앱 초기화
        :param source: 비디오 소스 (파일 경로 또는 웹캠 ID)
        :param use_popup: True이면 OpenCV 팝업창 사용, False이면 노트북 내에서 시각화
        :param use_webcam: 웹캠 사용 여부
        """
        self.source = source
        self.use_popup = use_popup
        self.use_webcam = use_webcam
        self.pose_model = PoseModel()
        self.visualizer = Visualizer()
        self.focal_length = -1
        # self.skeleton = engine.Skeleton(body_edges=engine.body_edges)
        self.skeleton = engine.Skeleton(body_edges)
        
        self.engine3D = None
        self.skeleton_set = None

    def setup(self):
        """
        모델 로딩 및 컴파일 수행
        """
        self.pose_model.prepare_model()
        self.pose_model.compile_model(device=utils.device_widget().value)

    def run(self):
        """
        메인 실행 루프: 비디오 프레임을 처리하고 포즈를 시각화
        """
        player = utils.VideoPlayer(self.source, flip=isinstance(self.source, int), fps=30)
        player.start()
        input_image = player.next()
        # 시각화 크기 설정
        scale = 450 / input_image.shape[1]
        width, height = int(input_image.shape[1] * scale), int(input_image.shape[0] * scale)

        self.engine3D = engine.Engine3js(grid=True, axis=True, view_width=width, view_height=height)

        if self.use_popup:
            display(self.engine3D.renderer)
            cv2.namedWindow("Pose Estimation", cv2.WINDOW_KEEPRATIO)
        else:
            imgbox = widgets.Image(format="jpg", height=height, width=width)
            display(widgets.HBox([self.engine3D.renderer, imgbox]))

        processing_times = collections.deque()

        while True:
            frame = player.next()
            if frame is None:
                break
            # 모델 입력 크기에 맞게 영상 리사이즈
            scaled_img = cv2.resize(frame, dsize=(self.pose_model.model.inputs[0].shape[3], self.pose_model.model.inputs[0].shape[2]))
            # 초점거리 설정 (처음 한 번 계산)
            if self.focal_length < 0:
                self.focal_length = np.float32(0.8 * scaled_img.shape[1])
            # 추론 시작
            start = time.time()
            outputs = self.pose_model.infer(scaled_img)
            poses_3d, poses_2d = engine.parse_poses(outputs, 1, self.pose_model.stride, self.focal_length, True)
            processing_time = (time.time() - start) * 1000
            processing_times.append(processing_time)
            if len(processing_times) > 200:
                processing_times.popleft()

            fps = 1000 / np.mean(processing_times)
             # 2D 관절 시각화
            frame = self.visualizer.draw_2d_pose(frame, poses_2d, scaled_img)
            
            # 3D 시각화
            if len(poses_3d) > 0:
                poses_3d_copy = poses_3d.copy()
                x, y, z = poses_3d_copy[:, 0::4], poses_3d_copy[:, 1::4], poses_3d_copy[:, 2::4]
                # 카메라 시점 보정을 위한 좌표 변형
                poses_3d[:, 0::4], poses_3d[:, 1::4], poses_3d[:, 2::4] = -z + 200, -y + 100, -x
                poses_3d = poses_3d.reshape(poses_3d.shape[0], 19, -1)[:, :, 0:3]
                people = self.skeleton(poses_3d=poses_3d)
                try:
                    self.engine3D.scene_remove(self.skeleton_set)
                except:
                    pass
                self.engine3D.scene_add(people)
                self.skeleton_set = people
            else:
                if self.skeleton_set:
                    self.engine3D.scene_remove(self.skeleton_set)
                    self.skeleton_set = None

            cv2.putText(frame, f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)", (10, 30),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)

            if self.use_popup:
                cv2.imshow("Pose Estimation", frame)
                if cv2.waitKey(1) == 27:
                    break
            else:
                imgbox.value = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
                self.engine3D.renderer.render(self.engine3D.scene, self.engine3D.cam)

        clear_output()
        player.stop()
        if self.use_popup:
            cv2.destroyAllWindows()