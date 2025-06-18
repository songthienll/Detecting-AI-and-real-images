import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torchvision.models as models
from torchvision import transforms
import tempfile
import os
import time

# Initialize session state for storing previous predictions
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None  # For image predictions
if 'last_video_predictions' not in st.session_state:
    st.session_state.last_video_predictions = None  # For video predictions
if 'last_frame_label' not in st.session_state:
    st.session_state.last_frame_label = None  # For last frame label
if 'last_frame_confidence' not in st.session_state:
    st.session_state.last_frame_confidence = None  # For last frame confidence

# Check device (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"**Device**: {'GPU' if torch.cuda.is_available() else 'CPU'}")
if device.type == "cuda":
    st.write(f"**GPU Name**: {torch.cuda.get_device_name(0)}")

# Title and description
st.title("AI vs Real Image/Video Detector")
st.markdown("""
Upload an image (JPEG, PNG) or a video (MP4, AVI, MOV) to determine if it's AI-generated or real.
Select a model from the dropdown below.
""")

# Model selection
model_choice = st.selectbox("Choose a model", ["MobileViT", "ConvNext", "SwinT"])

# Load models with caching
@st.cache_resource
def load_mobilevit():
    try:
        processor = AutoImageProcessor.from_pretrained("songthienll/mobilevit_ai_real_classifier")
        model = AutoModelForImageClassification.from_pretrained("songthienll/mobilevit_ai_real_classifier").to(device)
        return processor, model
    except Exception as e:
        st.error(f"Error loading MobileViT model: {str(e)}")
        return None, None

@st.cache_resource
def load_convnext():
    try:
        model = models.convnext_tiny(weights=None)
        model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, 2)
        state_dict = torch.load(r"D:\DAT301m\project1\best_model_convnext.pt", map_location=device, weights_only=True)
        if list(state_dict.keys())[0].startswith("module."):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading ConvNext model: {str(e)}")
        return None

@st.cache_resource
def load_swint():
    try:
        model = models.swin_t(weights=None)
        model.head = torch.nn.Linear(model.head.in_features, 2)
        state_dict = torch.load(r"D:\DAT301m\project1\best_model_swint.pt", map_location=device, weights_only=True)
        if list(state_dict.keys())[0].startswith("module."):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading SwinT model: {str(e)}")
        return None

# Define preprocessing for ConvNext and SwinT
def get_preprocess():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4502, 0.4242, 0.3926], std=[0.2224, 0.2084, 0.2035])
    ])

# Prediction functions for each model
def predict_mobilevit(image, processor, model):
    start_time = time.time()
    if image.mode != "RGB":
        image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        predicted_label = model.config.id2label[predicted_class]
        # Standardize label
        if predicted_label.lower() in ["ai", "ai-generated", "fake"]:
            standard_label = "AI-generated"
        elif predicted_label.lower() == "real":
            standard_label = "Real"
        else:
            standard_label = predicted_label
    inference_time = time.time() - start_time
    return standard_label, confidence, inference_time

def predict_convnext(image, model, preprocess):
    start_time = time.time()
    if image.mode != "RGB":
        image = image.convert("RGB")
    img_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_class = torch.argmax(outputs, dim=1).item()
    inference_time = time.time() - start_time
    label = "AI-generated" if pred_class == 0 else "Real"
    confidence = probs[pred_class]
    return label, confidence, inference_time

def predict_swint(image, model, preprocess):
    return predict_convnext(image, model, preprocess)  # Same as ConvNext

# Load the selected model and define predict function
if model_choice == "MobileViT":
    processor, model = load_mobilevit()
    if processor is None or model is None:
        st.stop()
    predict_func = lambda img: predict_mobilevit(img, processor, model)
elif model_choice == "ConvNext":
    model = load_convnext()
    if model is None:
        st.stop()
    preprocess = get_preprocess()
    predict_func = lambda img: predict_convnext(img, model, preprocess)
elif model_choice == "SwinT":
    model = load_swint()
    if model is None:
        st.stop()
    preprocess = get_preprocess()
    predict_func = lambda img: predict_swint(img, model, preprocess)

# File uploader
uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "png", "mp4", "avi", "mov"])

# Display previous results if no new file is uploaded
if uploaded_file is None and (st.session_state.last_prediction or st.session_state.last_video_predictions):
    st.write("### Previous Results")
    if st.session_state.last_prediction:
        label, confidence, inference_time = st.session_state.last_prediction
        st.write("#### Last Image Prediction")
        st.write(f"Total inference time: {inference_time:.3f} seconds")
        color = "green" if label == "Real" else "red"
        st.markdown(f"<p style='color:{color};'>Prediction: {label} (Confidence: {confidence:.2f})</p>", unsafe_allow_html=True)
    if st.session_state.last_video_predictions:
        predictions, total_inference_time, num_frames_processed, final_label, avg_confidence, real_count, ai_count = st.session_state.last_video_predictions
        st.write("#### Last Video Prediction")
        st.write(f"Total inference time: {total_inference_time:.3f} seconds")
        st.write("### Detailed Frame Predictions")
        with st.expander("Show all frame predictions", expanded=False):
            for frame_num, label, confidence in predictions:
                color = "green" if label == "Real" else "red"
                st.markdown(f"<p style='color:{color};'>Frame {frame_num}: {label} (Confidence: {confidence:.2f})</p>", unsafe_allow_html=True)
        st.markdown(f"<b>Overall Result: {final_label} (Average Confidence: {avg_confidence:.2f})</b>", unsafe_allow_html=True)
        st.write(f"**Frame Analysis**: {real_count} Real frames, {ai_count} AI-generated frames")

if uploaded_file is not None:
    file_extension = uploaded_file.name.split(".")[-1].lower()
    is_image = file_extension in ["jpg", "jpeg", "png"]
    is_video = file_extension in ["mp4", "avi", "mov"]

    if is_image:
        st.write("### Image Prediction")
        image = Image.open(uploaded_file)
        if image.mode != "RGB":
            image = image.convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        label, confidence, inference_time = predict_func(image)
        # Store prediction in session state
        st.session_state.last_prediction = (label, confidence, inference_time)
        st.write("### Total Inference Time")
        st.write(f"Total inference time: {inference_time:.3f} seconds")
        st.write("### Detailed Prediction")
        color = "green" if label == "Real" else "red"
        st.markdown(f"<p style='color:{color};'>Prediction: {label} (Confidence: {confidence:.2f})</p>", unsafe_allow_html=True)

    elif is_video:
        # Save uploaded video to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
            temp_file.write(uploaded_file.read())
            video_path = temp_file.name

        # Process video
        def process_video(video_path, predict_func, sampling_rate=5):
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.error("Error: Could not open video. Ensure the file is a valid MP4, AVI, or MOV.")
                return None, None, None, None

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            st.write(f"Total frames: {frame_count}, FPS: {fps:.2f}, Resolution: {width}x{height}")

            # Create a temporary file for the output video
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_file.close()

            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            out = cv2.VideoWriter(temp_file.name, fourcc, fps, (width, height))

            if not out.isOpened():
                st.warning("Codec failed, trying alternative codecs...")
                for codec in ["XVID", "MJPG", "mp4v"]:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    out = cv2.VideoWriter(temp_file.name, fourcc, fps, (width, height))
                    if out.isOpened():
                        st.info(f"Using {codec} codec")
                        break

            if not out.isOpened():
                st.error("Error: Could not initialize video writer with any codec.")
                cap.release()
                return None, None, None, None

            predictions = []
            total_inference_time = 0.0
            current_frame = 0
            sampled_frame_count = 0
            num_frames_to_process = (frame_count + sampling_rate - 1) // sampling_rate

            progress_bar = st.progress(0)
            status_text = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                current_frame += 1

                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if current_frame % sampling_rate == 0:
                    sampled_frame_count += 1
                    label, confidence, inference_time = predict_func(frame_pil)
                    predictions.append((current_frame, label, confidence))
                    total_inference_time += inference_time
                    # Store the latest frame prediction
                    st.session_state.last_frame_label = label
                    st.session_state.last_frame_confidence = confidence
                else:
                    # Use the last frame's prediction if available
                    label = st.session_state.last_frame_label if st.session_state.last_frame_label else "Unknown"
                    confidence = st.session_state.last_frame_confidence if st.session_state.last_frame_confidence else 0.0

                color = (0, 255, 0) if label == "Real" else (0, 0, 255) if label != "Unknown" else (128, 128, 128)
                cv2.putText(
                    frame, f"{label} ({confidence:.2f})", (10, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA
                )

                progress = min(sampled_frame_count / num_frames_to_process, 1.0)
                progress_bar.progress(progress)
                status_text.text(f"Processing frame {current_frame}: {label} ({confidence:.2f})")

                out.write(frame)

            cap.release()
            out.release()

            try:
                with open(temp_file.name, "rb") as f:
                    video_bytes = f.read()
                os.unlink(temp_file.name)
                return video_bytes, predictions, total_inference_time, sampled_frame_count
            except Exception as e:
                st.error(f"Error reading processed video: {str(e)}")
                if os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)
                return None, None, None, None

        output_video_bytes, predictions, total_inference_time, num_frames_processed = process_video(video_path, predict_func, sampling_rate=5)

        if output_video_bytes:
            st.write("### Output Video with Predictions")
            st.video(output_video_bytes)

            st.write("### Total Inference Time")
            st.write(f"Total inference time: {total_inference_time:.3f} seconds")

            st.write("### Detailed Frame Predictions")
            if predictions:
                ai_count = sum(1 for p in predictions if p[1] == "AI-generated")
                real_count = len(predictions) - ai_count
                final_label = "Real" if real_count > ai_count else "AI-generated"
                avg_confidence = np.mean([p[2] for p in predictions])
                # Store video predictions in session state
                st.session_state.last_video_predictions = (predictions, total_inference_time, num_frames_processed, final_label, avg_confidence, real_count, ai_count)

                with st.expander("Show all frame predictions", expanded=False):
                    for frame_num, label, confidence in predictions:
                        color = "green" if label == "Real" else "red"
                        st.markdown(f"<p style='color:{color};'>Frame {frame_num}: {label} (Confidence: {confidence:.2f})</p>", unsafe_allow_html=True)

                st.markdown(f"<b>Overall Result: {final_label} (Average Confidence: {avg_confidence:.2f})</b>", unsafe_allow_html=True)
                st.write(f"**Frame Analysis**: {real_count} Real frames, {ai_count} AI-generated frames")

            st.download_button(
                label="📥 Download Processed Video",
                data=output_video_bytes,
                file_name=f"processed_{uploaded_file.name}",
                mime="video/mp4"
            )
        else:
            st.error("Error processing video. Please ensure the file is valid.")
            # Display previous video predictions if available
            if st.session_state.last_video_predictions:
                st.write("### Previous Video Results")
                predictions, total_inference_time, num_frames_processed, final_label, avg_confidence, real_count, ai_count = st.session_state.last_video_predictions
                st.write(f"Total inference time: {total_inference_time:.3f} seconds")
                with st.expander("Show all frame predictions", expanded=False):
                    for frame_num, label, confidence in predictions:
                        color = "green" if label == "Real" else "red"
                        st.markdown(f"<p style='color:{color};'>Frame {frame_num}: {label} (Confidence: {confidence:.2f})</p>", unsafe_allow_html=True)
                st.markdown(f"<b>Overall Result: {final_label} (Average Confidence: {avg_confidence:.2f})</b>", unsafe_allow_html=True)
                st.write(f"**Frame Analysis**: {real_count} Real frames, {ai_count} AI-generated frames")

        if os.path.exists(video_path):
            os.remove(video_path)

    else:
        st.error("Unsupported file type. Please upload a JPEG, PNG, MP4, AVI, or MOV file.")
        # Display previous results if available
        if st.session_state.last_prediction:
            st.write("### Previous Image Prediction")
            label, confidence, inference_time = st.session_state.last_prediction
            st.write(f"Total inference time: {inference_time:.3f} seconds")
            color = "green" if label == "Real" else "red"
            st.markdown(f"<p style='color:{color};'>Prediction: {label} (Confidence: {confidence:.2f})</p>", unsafe_allow_html=True)
