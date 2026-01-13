import os
import numpy as np
import cv2
import argparse
import pyclipper
import bpu_infer_lib
import matplotlib.pyplot as plt
import collections
import imquality.brisque as brisque
import PIL.Image 
import pandas as pd
from pathlib import Path

class strLabelConverter:
    """Convert between string & label for OCR tasks.
    Args:
        alphabet (str): Set of possible characters.
        ignore_case(bool, default=True): Whether to ignore case
    """

    def __init__(self, alphabet, ignore_case=False):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-' 
        self.dict = {}
        for i, char in enumerate(alphabet):
            self.dict[char] = i+1
    
    def encode(self, text):
        """
        Encode a string or a list of strings into a sequence of integers
        Args:
            text (str or list of str): The text(s) to convert
        Returns:
            np.array: Encoded text as an array of indices
            np.array: Array of lenghts for each text
        """
        if isinstance(text, str):
            text = [
                self.dict[char.lower() if self._ignore_case else char]
                for char in text
            ]
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return np.array(text, dtype=np.int32), np.array(length, dtype=np.int32)

    def decode(self, t, length, raw=False):
        """
        Decode a sequence of indices back into a string
        Args:
            t (np.array): Encoded text as an array of indices
            length (np.array): Array of lengths for each text

        Raises:
            AssertionError: If the length of the text & the provided lenght do not match
        Returns:
            str or list of str: Decoded text
        """
        if len(length) == 1:
            length = length[0]
            assert len(t) == length, f'text with length: {len(t)} does not match declared length: {length}'
            if raw:
                return ''.join([self.alphabet[i-1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i]!=0 and (not (i>0 and t[i-1]==t[i])):
                        char_list.append(self.alphabet[t[i]-1])
                return ''.join(char_list)
        else:
            assert len(t)==length.sum(), f'texts with length: {len(t)} does not match with declared length: {length.sum()}'
            texts = []
            index = 0
            for i in range(length.size):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index: index+l], np.array([l]), raw=raw
                    )
                )
                index += 1
            return texts

class rec_model:

    def __init__(self, model_path, converter, input_size = (48, 320), output_size = (40, 97)):
        self.model = bpu_infer_lib.Infer(False)
        if not self.model.load_model(model_path):
            raise RuntimeError(f'Failed to load model from {model_path}')
        self.converter = converter
        self.output_size = output_size
        self.input_size = input_size

    def predict_float(self, img):
        """
        Perform prediction on an image
        Args:
            img (np.array): Input image
            img_path (str): Path to the image file
        Returns:
            str: Raw prediction result
            str: Simplified prediction result
        """
        image_resized = cv2.resize(img, dsize=(self.input_size[1], self.input_size[0]))
        image_resized = (image_resized / 255.0).astype(np.float32)
        input_image = np.zeros((image_resized.shape[0], image_resized.shape[1], 3), dtype = np.float32)
        input_image[:image_resized.shape[0], :image_resized.shape[1], :] = image_resized
        input_image = image_resized[:,:,[2,1,0]]
        input_image = input_image[None].transpose(0, 3, 1, 2)

        self.model.read_numpy_arr_float32(input_image, 0)
        self.model.forward(more=True)

        preds = self.model.get_infer_res_np_float32(0).reshape(1, *self.output_size)
        print('shape:', preds.shape)

        preds = np.transpose(preds, (1,0,2))
        confidences = np.max(preds, axis=2)
        preds = np.argmax(preds, axis=2)
        avg_confidence = np.mean(confidences)
        print(f'Avg sentence confidence: {avg_confidence}')
        preds = preds.transpose(1, 0).reshape(-1)
        preds_size = np.array([preds.size], dtype=np.int32)
        sim_pred = self.converter.decode(np.array(preds), np.array(preds_size), raw=False)
        print(f'Prediction: {sim_pred}')

        return sim_pred, avg_confidence

def init_args():
    parser = argparse.ArgumentParser(description='data_labelling')
    parser.add_argument('--rec_model_path', default='model/en_PP-OCRv3_rec_48x320_rgb.bin', type=str)
    parser.add_argument('--input_folder', default='output/dataset_raw/', type=str)
    parser.add_argument('--output_file', default='output/dataset_raw/dataset.csv', type=str)
    args = parser.parse_args()
    return args

def load_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f'Image file {img_path} not found')
    return img

def gather_label_info(img, labels):
    # Detect blur
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(f'Laplacian variance: {laplacian_var}')
    labels['laplacian_var'] = laplacian_var
    # Detect brightness
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    brightness = np.mean(v) # Value Channels in HSV captures intensity/brightness
    print(f'Brightness: {brightness}')
    labels['brightness'] = brightness
    # Detect contrast
    mean, std_dev = cv2.meanStdDev(gray)
    contrast = std_dev[0][0] # Std. Dev of pixel intensities is a good measure for contrast
    print(f'Contrast: {contrast}')
    labels['contrast'] = contrast
    # Detect noise level/ image quality
    #img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img_pil = PIL.Image.fromarray(img_rgb)
    #quality_score = brisque.score(img_pil)
    #print(f'Noise score: {quality_score}')
    # Detect resolution
    height, width, channels = img.shape
    print(f'Height: {height} Width: {width}')
    labels['height'] = height
    labels['width'] = width

def label_main(input_path, output_file):
    args = init_args()
    alphabet = """0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~!"#$%&'()*+,-./  """
    converter = strLabelConverter(alphabet)
    
    # Need to initialize the model outside the loop or the model gets reinitilized at every loop, the wights need to be reloaded  this could cause a cache creation that gives the same results no matter the input
    recognition_model = rec_model(args.rec_model_path, converter)
    
    directory_path = Path(input_path)
    files_list = [p for p in directory_path.iterdir() if p.is_file()]
    index = 0
    for img_path in files_list:
        img = load_image(str(img_path))
        labels = {}
        sim_pred, avg_confidence = recognition_model.predict_float(img)
        labels['file_path'] = str(img_path)
        labels['prediction'] = sim_pred
        labels['avg_confidence'] = avg_confidence
        gather_label_info(img, labels)
        print(labels)
        df = pd.DataFrame(labels, index=[index])
        index += 1
        df.to_csv(output_file, mode='a', header=not os.path.exists(output_file))

if __name__ == '__main__':
    args = init_args()
    main()

