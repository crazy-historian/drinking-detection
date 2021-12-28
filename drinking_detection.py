import cv2 as cv
import numpy as np
from math import sqrt, pow

np.random.seed(42)
CONFIDENCES = 0.5
THRESHOLD = 0.4
MINIMAL_DISTANCE = 100
OBJECT_TYPE_NAMES = open('./yolo3/coco.names').read().strip().split('\n')
EXPECTED_OBJECT_TYPES = ["person", "bottle", "wine glass", "cup"]
COLORS = np.random.randint(0, 255, size=(len(OBJECT_TYPE_NAMES), 3), dtype='uint8')


def get_blob_from_image(image):
    return cv.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)


def through_the_net(network, layers, blob):
    network.setInput(blob)
    return network.forward(layers)


def euclidean_distance(x_a, y_a, x_b, y_b):
    return sqrt(pow((x_a - x_b), 2) + pow((y_a - y_b), 2))


if __name__ == "__main__":

    camera_capture = cv.VideoCapture(0)
    network = cv.dnn.readNetFromDarknet(cfgFile='./yolo3/yolov3.cfg', darknetModel='./yolo3/yolov3.weights')
    network.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)

    layer_names = network.getLayerNames()
    layer_names = [layer_names[i - 1] for i in network.getUnconnectedOutLayers()]

    while True:
        ret, frame = camera_capture.read()
        H, W = frame.shape[:2]
        blob = get_blob_from_image(frame)

        layer_outputs = through_the_net(network, layer_names, blob)

        boxes = list()
        confidences = list()
        object_type_indices = list()
        person_box = None
        cup_box = None

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                object_type_index = np.argmax(scores)
                confidence = scores[object_type_index]
                object_type_name = OBJECT_TYPE_NAMES[object_type_index]

                if confidence > 0.5 and object_type_name in EXPECTED_OBJECT_TYPES:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    object_type_indices.append(object_type_index)

        box_indices = cv.dnn.NMSBoxes(boxes, confidences, CONFIDENCES, THRESHOLD)
        if len(box_indices) > 0:
            for box_index in box_indices:
                x, y = (boxes[box_index][0], boxes[box_index][1])
                w, h = (boxes[box_index][2], boxes[box_index][3])
                centerX, centerY = x + w // 2, y + h // 2

                object_type_name = OBJECT_TYPE_NAMES[object_type_indices[box_index]]
                if object_type_name == EXPECTED_OBJECT_TYPES[0]:
                    person_box = [centerX, centerY, w, h]
                elif object_type_name in EXPECTED_OBJECT_TYPES[1:]:
                    cup_box = [centerX, centerY, w, h]

                color = [int(c) for c in COLORS[object_type_indices[box_index]]]
                cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv.putText(frame, f"{OBJECT_TYPE_NAMES[object_type_indices[box_index]]}: {confidences[box_index]:.4f}",
                           (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                if person_box and cup_box:
                    person_bottom = person_box[0], person_box[1] + person_box[3] // 2
                    person_top = person_box[0], person_box[1] - person_box[3] // 2
                    cup_bottom = cup_box[0], cup_box[1] + cup_box[3] // 2
                    cup_top = cup_box[0], cup_box[1] - cup_box[3] // 2

                    cv.line(frame, person_box[:2], cup_box[:2], (255, 0, 0), 3)
                    cv.line(frame, person_bottom, cup_bottom, (0, 255, 0), 3)
                    cv.line(frame, person_top, cup_top, (0, 0, 255), 3)

                    distance = euclidean_distance(*person_box[:2], *cup_box[:2])
                    if distance < MINIMAL_DISTANCE:
                        cv.putText(frame, "I AM DRINKING", (H // 3, 40), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    else:
                        cv.putText(frame, "I AM NOT DRINKING", (H // 3, 40), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255),
                                   2)
        cv.imshow('frame', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    camera_capture.release()
    cv.destroyAllWindows()
