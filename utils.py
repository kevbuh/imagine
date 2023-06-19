import cv2

def split_frames(video_path, output_path):
    """
    split frames of video and put into output_path folder
    """
    frame_id = 0

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video file")
        return
    
    while cap.isOpened():

        # Read the video file frame by frame
        ret, frame = cap.read()

        # If 'ret' is True then 'frame' contains a valid image
        if ret:
            # You can use the 'frame' for further processing like face recognition, object detection etc.

            # Save frame as image
            cv2.imwrite(f'{output_path}/frame_{frame_id}.png', frame)
            frame_id += 1

            # Press Q on keyboard to exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break


        # Break the loop if no more frames are available
        else: 
            print('no more frames!')
            break

    # Release the video capture object and close windows
    cap.release()
    cv2.destroyAllWindows()

    return frame_id