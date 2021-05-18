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
        
        return landmarks

    def _drawPoints(self, img, landmarks, point_range, closed=False):
        points = []
        start_point, end_point = point_range
        for i in range(start_point, end_point+1):
            point = [landmarks.part(i).x, landmarks.part(i).y]
            points.append(point)

        points = np.array(points, dtype=np.int32)
        cv2.polylines(img, [points], closed, (255, 200, 0), thickness=2, lineType=cv2.LINE_8)


    def _draw_landmarks_on_img(self, img, landmarks):
        assert(landmarks.num_parts == 68)

        img_copy = img.copy()
        self._drawPoints(img_copy, landmarks, self.landmarks_dict['jaw'])         
        self._drawPoints(img_copy, landmarks, self.landmarks_dict['left_eyebrow'])        
        self._drawPoints(img_copy, landmarks, self.landmarks_dict['right_eyebrow'])        
        self._drawPoints(img_copy, landmarks, self.landmarks_dict['nose_bridge'])        
        self._drawPoints(img_copy, landmarks, self.landmarks_dict['lower_nose'], True)  
        self._drawPoints(img_copy, landmarks, self.landmarks_dict['left_eye'], True)  
        self._drawPoints(img_copy, landmarks, self.landmarks_dict['right_eye'], True)  
        self._drawPoints(img_copy, landmarks, self.landmarks_dict['outer_lip'], True)  
        self._drawPoints(img_copy, landmarks, self.landmarks_dict['inner_lip'], True)

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
        return np.array([[p.x, p.y] for p in points])

    def _to_polar_landmarks(self, landmarks, center=30, eps=1e-6):
        
        # Create numpy array
        arr = landmarks.parts()
        arr = self._dlib_points_to_np_array(arr)
        
        # Center points around nose = 30
        arr -= arr[center]

        distances = np.sum(np.abs(arr), axis=-1)
        angles = np.arctan(arr[:, 1] / (arr[:, 0] + eps))

        return np.array([distances, angles]).T

    def landmarks_distance(self, landmarks1, landmarks2, normalize_dist=True, drop_features=['jaw', 'nose_bridge']):
        for drop_feature in drop_features:
            if drop_feature not in self.landmarks_dict.keys():
                raise ValueError("Can not drop feature {0}. Feature that can be dropped are {1}."\
                    .format(drop_feature, self.landmarks_dict.keys()))

        polar1 = self._to_polar_landmarks(landmarks1)
        polar2 = self._to_polar_landmarks(landmarks2)

        if len(drop_features) > 0:
            remaining_features = list(self.landmarks_dict.keys())
            remaining_points_amount = 0

            for drop_feature in drop_features:
                remaining_features.remove(drop_feature)

            for remaining_feature in remaining_features:
                start, end = self.landmarks_dict[remaining_feature]
                remaining_points_amount += end - start + 1

            new_polar1 = np.zeros((remaining_points_amount, 2))
            new_polar2 = np.zeros((remaining_points_amount, 2))

            count = 0
            for remaining_feature in remaining_features:
                start, end = self.landmarks_dict[remaining_feature]
                diff = end - start + 1
                new_polar1[count : count + diff, :] = polar1[start:end+1, :]
                new_polar2[count : count + diff, :] = polar2[start:end+1, :]
                count += diff
            
            polar1 = new_polar1
            polar2 = new_polar2

        if normalize_dist:
            max1 = np.max(polar1[:, 0])
            max2 = np.max(polar2[:, 0])
            polar1[:, 0] /= max1
            polar2[:, 0] /= max2

        return np.sum(np.sqrt((polar1 - polar2)**2))