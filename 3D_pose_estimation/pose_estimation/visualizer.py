# visualizer.py
import cv2
import numpy as np
# 2D 관절 연결 정보 (body_edges_2d)
# 각 원소는 두 관절 인덱스를 연결하는 선을 의미합니다.
body_edges_2d = np.array([
    [0, 1], [1, 16], [16, 18], [1, 15], [15, 17],
    [0, 3], [3, 4], [4, 5], [0, 9], [9, 10], [10, 11],
    [0, 6], [6, 7], [7, 8], [0, 12], [12, 13], [13, 14],
])

class Visualizer:
    def __init__(self):
        pass

    def draw_2d_pose(self, frame, poses_2d, scaled_img):
        """
            입력 프레임 위에 2D 포즈(관절 점 및 연결 선)를 그립니다.

            :param frame: 원본 이미지 (카메라 또는 영상)
            :param poses_2d: 2D 관절 좌표 리스트 (사람마다 하나씩)
            :param scaled_img: 추론용으로 리사이즈된 입력 이미지 (크기 기준 정렬용)
            :return: 포즈가 그려진 이미지
        """
        for pose in poses_2d:
            pose = np.array(pose[0:-1]).reshape((-1, 3)).transpose()
            was_found = pose[2] > 0
            pose[0], pose[1] = (
                pose[0] * frame.shape[1] / scaled_img.shape[1],
                pose[1] * frame.shape[0] / scaled_img.shape[0],
            )
            for edge in body_edges_2d:
                if was_found[edge[0]] and was_found[edge[1]]:
                    cv2.line(
                        frame,
                        tuple(pose[0:2, edge[0]].astype(np.int32)),
                        tuple(pose[0:2, edge[1]].astype(np.int32)),
                        (255, 255, 0), 4, cv2.LINE_AA,
                    )
            for kpt_id in range(pose.shape[1]):
                if pose[2, kpt_id] != -1:
                    cv2.circle(
                        frame,
                        tuple(pose[0:2, kpt_id].astype(np.int32)),
                        3, (0, 255, 255), -1, cv2.LINE_AA,
                    )
        return frame  # 시각화된 프레임 반환