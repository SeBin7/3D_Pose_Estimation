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
import numpy as np
import time


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
    def __init__(self, source, use_popup=True, use_webcam=True):
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
        
        self.ema_values = {
            "hunchback": None,
            "forward_head": None,
            "rounded_shoulders": None,
        }
        self.pose_state = {
            "hunchback": {"bad": False, "start_time": None},
            "forward_head": {"bad": False, "start_time": None},
            "rounded_shoulders": {"bad": False, "start_time": None},
        }
        


    # ---- EMA 업데이트 함수 ----
    def update_ema(self, new_val, ema, alpha=0.2):
        if ema is None:
            return new_val
        return alpha * new_val + (1 - alpha) * ema

    # ---- 각도 계산 ----
    def get_angle_between(self, v1, v2):
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)
        dot = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
        return np.degrees(np.arccos(dot))

    # ---- 자세 판단 ----
    def check_hunchback_ema(self, keypoints):
        neck = keypoints[1]
        l_hip = keypoints[11]
        r_hip = keypoints[8]
        mid_hip = (l_hip + r_hip) / 2
        upper_vec = neck - mid_hip
        angle = self.get_angle_between(upper_vec, np.array([0, 1, 0]))
        self.ema_values["hunchback"] = self.update_ema(angle, self.ema_values["hunchback"])
        return self.ema_values["hunchback"] > 35

    def check_forward_head_ema(self, keypoints):
        nose_z = keypoints[0][2]
        neck_z = keypoints[1][2]
        delta = neck_z - nose_z
        self.ema_values["forward_head"] = self.update_ema(delta, self.ema_values["forward_head"])
        return self.ema_values["forward_head"] > 0.03

    def check_rounded_shoulders_ema(self, keypoints):
        l_shoulder_z = keypoints[5][2]
        r_shoulder_z = keypoints[2][2]
        l_hip_z = keypoints[11][2]
        r_hip_z = keypoints[8][2]
        shoulder_avg_z = (l_shoulder_z + r_shoulder_z) / 2
        hip_avg_z = (l_hip_z + r_hip_z) / 2
        delta = hip_avg_z - shoulder_avg_z
        self.ema_values["rounded_shoulders"] = self.update_ema(delta, self.ema_values["rounded_shoulders"])
        return self.ema_values["rounded_shoulders"] > 0.02

    WARNING_THRESHOLD = 3

    def update_posture_state(self, name, is_now_bad):
        state = self.pose_state[name]
        current_time = time.time()

        if is_now_bad:
            if not state["bad"]:
                state["bad"] = True
                state["start_time"] = current_time
            elif state["start_time"] and (current_time - state["start_time"]) >= self.WARNING_THRESHOLD:
                print(f"❌ {name} 나쁜 자세가 10초 넘게 유지됨!")
                state["start_time"] = current_time + 999
        else:
            if state["bad"]:
                print(f"✅ {name} 정상으로 복귀!")
            state["bad"] = False
            state["start_time"] = None

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
        self.source = 0
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

            for person in poses_3d:
                self.update_posture_state("hunchback", self.check_hunchback_ema(person))
                self.update_posture_state("forward_head", self.check_forward_head_ema(person))
                self.update_posture_state("rounded_shoulders", self.check_rounded_shoulders_ema(person))

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