import os
import torch
from .ai_model import GradCAMDenseNet

class ProgressionAnalyzer:
    def __init__(self):
        # Re-use the GradCAMDenseNet model pipeline
        self.detector = GradCAMDenseNet()

    def analyze_progression(self, prior_image_path, current_image_path, output_dir):
        # 1. Generate predictions and Grad-CAM for both scans
        prior_gradcam_path = os.path.join(output_dir, "prior_gradcam.jpg")
        current_gradcam_path = os.path.join(output_dir, "current_gradcam.jpg")

        prior_res = self.detector.predict_and_explain(prior_image_path, prior_gradcam_path)
        current_res = self.detector.predict_and_explain(current_image_path, current_gradcam_path)

        # 2. Extract confidence scores and diseases
        p_disease, p_conf = prior_res["predicted_disease"], prior_res["confidence"]
        c_disease, c_conf = current_res["predicted_disease"], current_res["confidence"]

        # 3. Determine progression status
        # Threshold margin of 5.0% to account for minor variation
        conf_diff = c_conf - p_conf

        if c_disease != p_disease:
            status = f"Condition changed from {p_disease} to {c_disease}"
        elif conf_diff > 5.0:
            status = "Worsening"
        elif conf_diff < -5.0:
            status = "Improving"
        else:
            status = "Stable"

        return {
            "prior_scan": {
                "disease": p_disease,
                "confidence": p_conf,
                "gradcam_path": prior_res["gradcam_path"]
            },
            "current_scan": {
                "disease": c_disease,
                "confidence": c_conf,
                "gradcam_path": current_res["gradcam_path"]
            },
            "progression_status": status,
            "confidence_change": round(conf_diff, 2)
        }

if __name__ == "__main__":
    analyzer = ProgressionAnalyzer()
    print("Progression Analyzer initialized successfully!")