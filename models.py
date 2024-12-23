import torch
from torch import nn
import torch.nn.functional as F
from tool.torch_utils import *
from tool.yolo_layer import YoloLayer
import cv2
from tool import darknet2pytorch
import pickle
from tool.utils import load_class_names, plot_boxes_cv2
from tool.torch_utils import do_detect


# load weights from darknet format
model = darknet2pytorch.Darknet('/content/drive/MyDrive/snipBack/yolov4-basketball.cfg', inference=True)
model.load_weights('/content/drive/MyDrive/snipBack/yolov4-basketball.weights')

# save weights to pytorch format
torch.save(model.state_dict(), 'yolov4_orig.pth')

# # reload weights from pytorch format
model = darknet2pytorch.Darknet('/content/drive/MyDrive/snipBack/yolov4-basketball.cfg', inference=True)
model.load_state_dict(torch.load('/content/pytorch-YOLOv4/yolov4_orig.pth'))





if __name__ == "__main__":







    use_cuda = True
    if use_cuda:
        model.cuda()



    # Constants
    FRAME_STEP = 5  # Perform inference on every 5th frame

    # Set up video capture
    video_path = '/content/drive/MyDrive/videos.mp4'  # Update with your video path
    output_path = 'output_video.mp4'
    capture = cv2.VideoCapture(video_path)

    # Initialize video writer for saving output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(total_frames)
    print(fps)
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_results_list = []  # List to store results for all frames
    frame_idx = 0  # To track frame index

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break

        frame_idx += 1    
        print(frame_idx)
        # Skip frames based on FRAME_STEP
        if frame_idx % FRAME_STEP != 0:
            frame_results_list.append([])  # Append empty list for skipped frames
          
            continue

        # Preprocess the frame
        sized = cv2.resize(frame, (1920, 1088))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        # Perform detection
        boxes = do_detect(model, sized, 0.5, 0.5, use_cuda)
        preds = boxes[0]
        frame_results = []

        for i in range(len(preds)):
            box = preds[i]
            NewBox = []
            x1 = int(box[0] * frame.shape[1])
            y1 = int(box[1] * frame.shape[0])
            x2 = int(box[2] * frame.shape[1])
            y2 = int(box[3] * frame.shape[0])
            NewBox.append(x1)
            NewBox.append(y1)
            NewBox.append(x2)
            NewBox.append(y2)
            NewBox.append(box[4])
            NewBox.append(box[6])

            # Save the updated box
            preds[i] = NewBox
            print(NewBox)
            frame_results.append(NewBox)

        # Append the results of this frame to the main list
        frame_results_list.append(frame_results)

        # Draw bounding boxes on the frame
        for box in preds:
            x1, y1, x2, y2, confidence, class_id = map(int, box[:6])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Class {class_id}: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write the frame to the output video
        video_writer.write(frame)
        # frame_idx += 1

    # Release resources
    capture.release()
    video_writer.release()

    # Save results to a pickle file
    with open('detection_results.pkl', 'wb') as f:
        pickle.dump(frame_results_list, f)

    print("Inference completed. Results saved in 'detection_results.pkl' and video saved in", output_path)

