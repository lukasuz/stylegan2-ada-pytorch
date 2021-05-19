import dlib, cv2
import numpy as np
import warnings

class FacialLandmarksExtractor:
    def __init__(self, model_path):
        self.frontalFaceDetector = dlib.get_frontal_face_detector()
        self.faceLandmarkDetector = dlib.shape_predictor(model_path)

        self.landmarks_dict = {
            'jaw': (0, 16),           
            'left_eyebrow': (17, 21), 
            'right_eyebrow': (22, 26),
            'nose_bridge': (27, 30),  
            'lower_nose':(30, 35),    
            'left_eye': (36, 41),     
            'right_eye': (42, 47),    
            'outer_lip':(48, 59),     
            'inner_lip':(60, 67)      
        }

        # define lower part of the nose bridge as center
        self.center = self.landmarks_dict['nose_bridge'][0] 
     
    def read_and_extract(self, path):

        img = cv2.imread(path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img, self.extract(img_rgb)

    def extract(self, img):

        face_bounding_boxes = self.frontalFaceDetector(img, 0)
        if len(face_bounding_boxes) > 0:
            warnings.warn("Warning: two faces detected. Omitting the second one.")
        
        if len(face_bounding_boxes) == 0:
            raise ValueError("No face detected.")

        face_bb = face_bounding_boxes[0]

        face_rectangle = dlib.rectangle(
            int(face_bb.left()),
            int(face_bb.top()),
            int(face_bb.right()),
            int(face_bb.bottom())
            )
        landmarks = self.faceLandmarkDetector(img, face_rectangle)

        if len(landmarks.parts()) != 68:
            raise ValueError("Wrong number of landmarks detected, detected:", len(landmarks.parts()))
        
        landmarks = self._dlib_points_to_np_array(landmarks.parts())

        return landmarks

    def _drawPoints(self, img, landmarks_np, point_range, col=(255, 200, 0), closed=False):
        points = []
        start_point, end_point = point_range
        for i in range(start_point, end_point+1):
            point = [landmarks_np[i, 0], landmarks_np[i, 1]]
            points.append(point)

        points = np.array(points, dtype=np.int32)
        cv2.polylines(img, [points], closed, col, thickness=2, lineType=cv2.LINE_8)


    def _draw_landmarks_on_img(self, img, landmarks, col=(255, 200, 0)):
        if type(landmarks) != np.ndarray:
            landmarks = self._dlib_points_to_np_array(landmarks.parts())
        assert(len(landmarks) == 68)

        img_copy = img.copy()
        self._drawPoints(img_copy, landmarks, self.landmarks_dict['jaw'], col)         
        self._drawPoints(img_copy, landmarks, self.landmarks_dict['left_eyebrow'], col)        
        self._drawPoints(img_copy, landmarks, self.landmarks_dict['right_eyebrow'], col)        
        self._drawPoints(img_copy, landmarks, self.landmarks_dict['nose_bridge'], col)        
        self._drawPoints(img_copy, landmarks, self.landmarks_dict['lower_nose'], col, True)  
        self._drawPoints(img_copy, landmarks, self.landmarks_dict['left_eye'], col, True)  
        self._drawPoints(img_copy, landmarks, self.landmarks_dict['right_eye'], col, True)  
        self._drawPoints(img_copy, landmarks, self.landmarks_dict['outer_lip'], col, True)  
        self._drawPoints(img_copy, landmarks, self.landmarks_dict['inner_lip'], col, True)

        return img_copy
    
    def display_landmarks_img(self, img, landmarks):
        landmarks_img = self._draw_landmarks_on_img(img, landmarks)
        cv2.imshow("Landmark image", landmarks_img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_landmarks_img(self, img, landmarks, save_path="output.png"):
        landmarks_img = self._draw_landmarks_on_img(img, landmarks)

        cv2.imwrite(save_path, landmarks_img)

    def _dlib_points_to_np_array(self, points):
        return np.array([[p.x, p.y] for p in points]).astype('float32')

    def _drop_features(self, arr, drop_features):
        remaining_features = list(self.landmarks_dict.keys())
        remaining_points_amount = 0

        for drop_feature in drop_features:
            remaining_features.remove(drop_feature)

        for remaining_feature in remaining_features:
            start, end = self.landmarks_dict[remaining_feature]
            remaining_points_amount += end - start + 1

        new_arr = np.zeros((remaining_points_amount, 2))
        count = 0
        for remaining_feature in remaining_features:
            start, end = self.landmarks_dict[remaining_feature]
            diff = end - start + 1
            new_arr[count : count + diff, :] = arr[start:end+1, :]
            count += diff
        
        return arr

    def _project_landmarks(self, landmarks1, landmarks2, partial_features=False):
        landmarks1_np = landmarks1.reshape(-1,1,2)
        landmarks2_np = landmarks2.reshape(-1,1,2)
        mask = np.ones((68), dtype=bool)
        if partial_features:
            mask = np.zeros((68), dtype=bool)
            mask[self.landmarks_dict['jaw'][0]:self.landmarks_dict['jaw'][1]] = True
            mask[self.landmarks_dict['nose_bridge'][0]:self.landmarks_dict['nose_bridge'][1]] = 1
            mask[self.landmarks_dict['lower_nose'][0]:self.landmarks_dict['lower_nose'][1]] = 1
        
        H, mask = cv2.findHomography(landmarks2_np[mask], landmarks1_np[mask], 0)

        return cv2.perspectiveTransform(landmarks2_np, H).squeeze()

    def display_langmark_projection(self, img1, landmarks1, landmarks2):
        landmarks2_np_projected = FLE._project_landmarks(landmarks1, landmarks2)
        temp_img = self._draw_landmarks_on_img(img1, landmarks2_np_projected, col=(0, 255, 200))
        self.display_landmarks_img(temp_img, landmarks1)

    def landmarks_distance(self, landmarks1, landmarks2, normalise=True, drop_features=[], no_calc=False):
        for drop_feature in drop_features:
            if drop_feature not in self.landmarks_dict.keys():
                raise ValueError("Can not drop feature {0}. Feature that can be dropped are {1}."\
                    .format(drop_feature, self.landmarks_dict.keys()))

        if normalise: # project landmarks2 onto landmarks 1
            landmarks2 = self._project_landmarks(landmarks1, landmarks2)
        else: # just center
            landmarks1 -= landmarks1[self.center]
            landmarks2 -= landmarks2[self.center]

        if len(drop_features) > 0:
            landmarks1 = self._drop_features(landmarks1, drop_features)
            landmarks2 = self._drop_features(landmarks2, drop_features)

        if no_calc:
            return landmarks1, landmarks2

        return np.sum(np.sqrt((landmarks1 - landmarks2)**2))

model_path = "./shape_predictor_68_face_landmarks.dat"
path1 = "./obama.webp"
path2 = "2.jpg"

FLE = FacialLandmarksExtractor(model_path)
img1, landmarks1 = FLE.read_and_extract(path1)
img2, landmarks2 = FLE.read_and_extract(path2)

face_distance = FLE.landmarks_distance(landmarks1, landmarks2)
print("Face Distance:", face_distance)

FLE.display_landmarks_img(img2, landmarks2)
FLE.display_langmark_projection(img1, landmarks1, landmarks2)

