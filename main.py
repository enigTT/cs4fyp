from src.detectors.hands_det import start_hands
from src.detectors.pose_det import start_pose
from src.detectors.holistic_det import start_holistic


def pose():
    start_pose()


def holistic():
    start_holistic()


def hands():
    start_hands()


if __name__ == "__main__":
    hands()
