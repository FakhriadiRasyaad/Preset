# from flask import Flask, request, jsonify, render_template
# import os
# import cv2
# import numpy as np
# import base64
# from werkzeug.utils import secure_filename

# app = Flask(__name__)
# UPLOAD_FOLDER = 'uploads'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload():
#     file = request.files['image']
#     filename = secure_filename(file.filename)
#     filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     file.save(filepath)
#     return jsonify({"filename": filename})

# @app.route('/preview', methods=['POST'])
# def preview():
#     data = request.get_json()
#     filename = data['filename']
#     params = data['params']
#     filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

#     img = cv2.imread(filepath)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = img.astype(np.float32) / 255.0

#     def apply_curve_channel(img, channel, adjustment):
#         img[..., channel] = np.clip(img[..., channel] * (1 + adjustment), 0, 1)
#         return img

#     # === Basic Adjustments ===
#     img += params.get("Exposure2012", 0) / 100.0

#     contrast = params.get("Contrast2012", 0) / 100.0
#     if contrast != 0:
#         mean = img.mean(axis=(0, 1), keepdims=True)
#         img = (img - mean) * (1 + contrast) + mean

#     # Highlights and Shadows
#     shadow_mask = img < 0.5
#     highlight_mask = img >= 0.5
#     img[shadow_mask] += params.get("Shadows2012", 0) / 100.0 * 0.5
#     img[highlight_mask] += params.get("Highlights2012", 0) / 100.0 * 0.5

#     # Whites & Blacks
#     img = np.clip(img + params.get("Whites2012", 0) / 100.0, 0, 1)
#     img = np.clip(img - params.get("Blacks2012", 0) / 100.0, 0, 1)

#     # Texture, Clarity
#     clarity = params.get("Clarity2012", 0) / 100.0 + params.get("Texture", 0) / 100.0
#     if clarity != 0:
#         blur = cv2.GaussianBlur(img, (0, 0), 3)
#         img = cv2.addWeighted(img, 1 + clarity, blur, -clarity, 0)

#     # Dehaze
#     if params.get("Dehaze", 0) != 0:
#         p_low = np.percentile(img, 5)
#         p_high = np.percentile(img, 95)
#         img = np.clip((img - p_low) / (p_high - p_low + 1e-5), 0, 1)

#     # Vibrance + Saturation
#     hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
#     vibrance = params.get("Vibrance", 0) / 100.0
#     saturation = params.get("Saturation", 0) / 100.0
#     hsv[..., 1] = np.clip(hsv[..., 1] * (1 + saturation + vibrance), 0, 255)
#     img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

#     # Temperature/Tint
#     img[..., 0] += params.get("IncrementalTemperature", 0) / 200.0
#     img[..., 2] -= params.get("IncrementalTemperature", 0) / 200.0
#     img[..., 1] += params.get("IncrementalTint", 0) / 200.0

#     # Parametric Tone Curves (simplified)
#     for key in ["ParametricShadows", "ParametricDarks", "ParametricLights", "ParametricHighlights"]:
#         img += params.get(key, 0) / 100.0

#     # Sharpness
#     if params.get("Sharpness", 0) != 0:
#         img = cv2.addWeighted(img, 1 + params.get("Sharpness", 0) / 100.0, cv2.GaussianBlur(img, (0, 0), 2), -params.get("Sharpness", 0) / 100.0, 0)

#     # Noise Reduction
#     if params.get("LuminanceSmoothing", 0) > 0:
#         img = cv2.GaussianBlur(img, (3, 3), sigmaX=params.get("LuminanceSmoothing", 0) / 10.0)

#     # HUE / SAT / LUM per Channel (simplified as global)
#     for i, color in enumerate(["Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta"]):
#         img = apply_curve_channel(img, i % 3, params.get(f"HueAdjustment{color}", 0) / 100.0)
#         img = apply_curve_channel(img, i % 3, params.get(f"SaturationAdjustment{color}", 0) / 100.0)
#         img = apply_curve_channel(img, i % 3, params.get(f"LuminanceAdjustment{color}", 0) / 100.0)

#     # Split Toning & Color Grading (simplified)
#     img[..., 0] += params.get("SplitToningShadowHue", 0) / 300.0
#     img[..., 2] += params.get("SplitToningHighlightHue", 0) / 300.0
#     img[..., 1] += params.get("ColorGradeMidtoneHue", 0) / 300.0

#     # Final clipping
#     img = np.clip(img, 0, 1)
#     img_uint8 = (img * 255).astype(np.uint8)
#     _, buffer = cv2.imencode('.jpg', cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))
#     img_base64 = base64.b64encode(buffer).decode('utf-8')

#     return jsonify({"image": img_base64})

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, request, jsonify, render_template
import os
import cv2
import numpy as np
import base64
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    return jsonify({"filename": filename})

@app.route('/preview', methods=['POST'])
def preview():
    data = request.get_json()
    filename = data['filename']
    params = data['params']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0

    def adj(val):
        return val / 100.0

    def apply_channel_adjust(img, color, val):
        idx = {"Red": 0, "Green": 1, "Blue": 2, "Orange": 0, "Yellow": 0, "Aqua": 1, "Purple": 2, "Magenta": 2}
        ch = idx.get(color, 0)
        img[..., ch] = np.clip(img[..., ch] + adj(val), 0, 1)
        return img

    img += adj(params.get("Exposure2012", 0))

    contrast = adj(params.get("Contrast2012", 0))
    if contrast != 0:
        mean = img.mean(axis=(0, 1), keepdims=True)
        img = (img - mean) * (1 + contrast) + mean

    img[img < 0.5] += adj(params.get("Shadows2012", 0)) * 0.5
    img[img >= 0.5] += adj(params.get("Highlights2012", 0)) * 0.5

    img += adj(params.get("Whites2012", 0))
    img -= adj(params.get("Blacks2012", 0))

    texture = adj(params.get("Texture", 0))
    clarity = adj(params.get("Clarity2012", 0))
    if texture + clarity != 0:
        blur = cv2.GaussianBlur(img, (0, 0), 3)
        img = cv2.addWeighted(img, 1 + texture + clarity, blur, -(texture + clarity), 0)

    if params.get("Dehaze", 0) != 0:
        p_low = np.percentile(img, 5)
        p_high = np.percentile(img, 95)
        img = np.clip((img - p_low) / (p_high - p_low + 1e-5), 0, 1)

    # Vibrance & Saturation
    hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] *= 1 + adj(params.get("Vibrance", 0) + params.get("Saturation", 0))
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

    # Temperature & Tint
    img[..., 0] += adj(params.get("IncrementalTemperature", 0))
    img[..., 2] -= adj(params.get("IncrementalTemperature", 0))
    img[..., 1] += adj(params.get("IncrementalTint", 0))

    if params.get("Sharpness", 0) != 0:
        img = cv2.addWeighted(img, 1 + adj(params.get("Sharpness", 0)), cv2.GaussianBlur(img, (0, 0), 2), -adj(params.get("Sharpness", 0)), 0)

    if params.get("LuminanceSmoothing", 0) > 0:
        img = cv2.GaussianBlur(img, (3, 3), sigmaX=adj(params.get("LuminanceSmoothing", 0)) * 10)

    # HSL Adjustments
    for color in ["Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta"]:
        img = apply_channel_adjust(img, color, params.get(f"HueAdjustment{color}", 0))
        img = apply_channel_adjust(img, color, params.get(f"SaturationAdjustment{color}", 0))
        img = apply_channel_adjust(img, color, params.get(f"LuminanceAdjustment{color}", 0))

    # Parametric Curve (simplified as global contrast boost)
    for p in ["ParametricShadows", "ParametricDarks", "ParametricLights", "ParametricHighlights"]:
        img += adj(params.get(p, 0))

    # Split Toning (simplified)
    img[..., 0] += adj(params.get("SplitToningShadowHue", 0)) + adj(params.get("SplitToningHighlightHue", 0))
    img[..., 1] += adj(params.get("SplitToningShadowSaturation", 0)) + adj(params.get("SplitToningHighlightSaturation", 0))

    # Color Grading
    img[..., 0] += adj(params.get("ColorGradeGlobalHue", 0) + params.get("ColorGradeGlobalSat", 0))
    img[..., 1] += adj(params.get("ColorGradeMidtoneHue", 0) + params.get("ColorGradeMidtoneSat", 0))
    img[..., 2] += adj(params.get("ColorGradeShadowLum", 0) + params.get("ColorGradeMidtoneLum", 0) + params.get("ColorGradeHighlightLum", 0))

    # Other effects (simplified global adjustments)
    img += adj(params.get("ColorGradeGlobalLum", 0))
    img += adj(params.get("ColorGradeBlending", 0))
    img += adj(params.get("LensManualDistortionAmount", 0))
    img += adj(params.get("VignetteAmount", 0))
    img += adj(params.get("DefringePurpleAmount", 0) + params.get("DefringeGreenAmount", 0))
    img += adj(params.get("GrainAmount", 0))
    img += adj(params.get("PostCropVignetteAmount", 0))
    img += adj(params.get("ShadowTint", 0))

    for c in ["Red", "Green", "Blue"]:
        img = apply_channel_adjust(img, c, params.get(f"{c}Hue", 0))
        img = apply_channel_adjust(img, c, params.get(f"{c}Saturation", 0))

    img = np.clip(img, 0, 1)
    img_uint8 = (img * 255).astype(np.uint8)
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({"image": img_base64})

if __name__ == '__main__':
    app.run(debug=True)