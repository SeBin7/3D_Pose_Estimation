from pose_estimation.posture_analyzer import PoseEstimatorApp

app = PoseEstimatorApp(source="face-demographics-walking.mp4", use_popup=True)
app.setup()
app.run()