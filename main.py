import cv2
import torch
import numpy as np

try:
    model_type = "MiDaS_small"
    # Load the MiDaS model from PyTorch Hub
    midas = torch.hub.load("intel-isl/MiDaS", model_type)

    # Use CUDA if available, otherwise use CPU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()

    # Load the appropriate transform based on the model type
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform if model_type == "MiDaS_small" else midas_transforms.dpt_transform

    # Open the default camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while(True):
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Transform the image and move it to the device
        input_batch = transform(img).to(device)

        with torch.no_grad():
            # Make a prediction with the MiDaS model
            prediction = midas(input_batch)

            # Resize the prediction to the original image size
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        # Move the prediction to the CPU and convert it to a NumPy array
        depth_map = prediction.cpu().numpy()

        # Normalize the depth map to the range 0-255 and convert it to an 8-bit grayscale image
        output_display = cv2.normalize(depth_map, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8U)
        output_display = cv2.cvtColor(output_display, cv2.COLOR_GRAY2BGR)

        # Concatenate the original frame and the depth map horizontally
        concatenated_output = np.concatenate((frame, output_display), axis=1)

        # Display the concatenated output
        cv2.imshow('Camera Feed | MiDaS Output', concatenated_output)

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and destroy all windows
    cap.release()
    cv2.destroyAllWindows()

except Exception as e:
    print(f"An error occurred: {e}")