import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
import os
import logging
import sys
from PIL import Image

from transformers import AutoModelForObjectDetection

from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch
from PIL import ImageDraw

from transformers import TableTransformerForObjectDetection

import numpy as np
import pandas as pd
import easyocr
from tqdm.auto import tqdm


os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

class Text:
    BOLD_START = '\033[1m'
    END = '\033[0m'
    UNDERLINE = '\033[4m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'

class MaxResize(object):
    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize((int(round(scale*width)), int(round(scale*height))))

        return resized_image    

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-detection", revision="no_timm")
model.to(device)

structure_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-structure-recognition-v1.1-all")
structure_model.to(device)


structure_transform = transforms.Compose([
    MaxResize(1000),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
from pdf2image import convert_from_path



reader = easyocr.Reader(['en'])
##pdf2img conversion code
def pdf_to_image(pdf_path, image_path):
    """
    Convert a specific page of a PDF to an image using pdf2image.

    Parameters:
    - pdf_path (str): Path to the PDF file.
    - page_number (int): Page number to convert (starting from 0).
    - image_path (str): Path to save the output image.

    Returns:
    - None
    # """
    # Convert the PDF page to an image
    images = convert_from_path(pdf_path)
    for i in range(len(images)):
    # Save the image
        try:
            images[i].save(image_path+'table_'+'_'+str(i)+'.png', 'PNG')
        except Exception as e:
            print('Error in pdf2img coversion',e)    

    return 'Done Conversion Successfully'

# Example usage
pdf_path = 'CombiningAbilityandHeterosisforPolygenic.pdf'
image_path = '/Table_transformer/'  # Replace with the desired output image path



detection_transform = transforms.Compose([
MaxResize(800),
transforms.ToTensor(),
transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b



def outputs_to_objects(outputs, img_size, id2label):
    m = outputs.logits.softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs['pred_boxes'].detach().cpu()[0]
    pred_bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)]

    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = id2label[int(label)]
        if not class_label == 'no object':
            objects.append({'label': class_label, 'score': float(score),
                            'bbox': [float(elem) for elem in bbox]})

    return objects


def apply_ocr(cell_coordinates,cropped_table,idx1):
    # let's OCR row by row
    data = dict()
    max_num_columns = 0
    for idx, row in enumerate(tqdm(cell_coordinates)):
      row_text = []
      for cell in row["cells"]:
        # crop cell out of image
        
        cell_image = np.array(cropped_table.crop(cell["cell"]))
        # apply OCR
        result = reader.readtext(np.array(cell_image))
        if len(result) > 0:
          # print([x[1] for x in list(result)])
          text = " ".join([x[1] for x in result])
          row_text.append(text)

      if len(row_text) > max_num_columns:
          max_num_columns = len(row_text)

      data[idx] = row_text

    print("Max number of columns:", max_num_columns)

    # pad rows which don't have max_num_columns elements
    # to make sure all rows have the same number of columns
    for row, row_data in data.copy().items():
        if len(row_data) != max_num_columns:
          row_data = row_data + ["" for _ in range(max_num_columns - len(row_data))]
        data[row] = row_data
    
    df =pd.DataFrame(data).transpose() 
    df.to_csv('table_data_'+str(idx1)+'.csv',index=False)
    return df
def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def visualize_detected_tables(img, det_tables, out_path=None):
    plt.imshow(img, interpolation="nearest")
    fig = plt.gcf()
    fig.set_size_inches(20, 20)
    ax = plt.gca()

    for det_table in det_tables:
        bbox = det_table['bbox']

        if det_table['label'] == 'table':
            facecolor = (1, 0, 0.45)
            edgecolor = (1, 0, 0.45)
            alpha = 0.3
            linewidth = 2
            hatch='//////'
        elif det_table['label'] == 'table rotated':
            facecolor = (0.95, 0.6, 0.1)
            edgecolor = (0.95, 0.6, 0.1)
            alpha = 0.3
            linewidth = 2
            hatch='//////'
        else:
            continue

        rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=linewidth,
                                    edgecolor='none',facecolor=facecolor, alpha=0.1)
        ax.add_patch(rect)
        rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=linewidth,
                                    edgecolor=edgecolor,facecolor='none',linestyle='-', alpha=alpha)
        ax.add_patch(rect)
        rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=0,
                                    edgecolor=edgecolor,facecolor='none',linestyle='-', hatch=hatch, alpha=0.2)
        ax.add_patch(rect)

    plt.xticks([], [])
    plt.yticks([], [])

    legend_elements = [Patch(facecolor=(1, 0, 0.45), edgecolor=(1, 0, 0.45),
                                label='Table', hatch='//////', alpha=0.3),
                        Patch(facecolor=(0.95, 0.6, 0.1), edgecolor=(0.95, 0.6, 0.1),
                                label='Table (rotated)', hatch='//////', alpha=0.3)]
    plt.legend(handles=legend_elements, bbox_to_anchor=(0.5, -0.02), loc='upper center', borderaxespad=0,
                    fontsize=10, ncol=2)
    plt.gcf().set_size_inches(10, 10)
    plt.axis('off')

    if out_path is not None:
      plt.savefig(out_path, bbox_inches='tight', dpi=150)

    return fig
def objects_to_crops(img, tokens, objects, class_thresholds, padding=10):
    """
    Process the bounding boxes produced by the table detection model into
    cropped table images and cropped tokens.
    """

    table_crops = []
    for obj in objects:
        if obj['score'] < class_thresholds[obj['label']]:
            continue

        cropped_table = {}

        bbox = obj['bbox']
        bbox = [bbox[0]-padding, bbox[1]-padding, bbox[2]+padding, bbox[3]+padding]

        cropped_img = img.crop(bbox)

        table_tokens = [token for token in tokens if iob(token['bbox'], bbox) >= 0.5]
        for token in table_tokens:
            token['bbox'] = [token['bbox'][0]-bbox[0],
                             token['bbox'][1]-bbox[1],
                             token['bbox'][2]-bbox[0],
                             token['bbox'][3]-bbox[1]]

        # If table is predicted to be rotated, rotate cropped image and tokens/words:
        if obj['label'] == 'table rotated':
            cropped_img = cropped_img.rotate(270, expand=True)
            for token in table_tokens:
                bbox = token['bbox']
                bbox = [cropped_img.size[0]-bbox[3]-1,
                        bbox[0],
                        cropped_img.size[0]-bbox[1]-1,
                        bbox[2]]
                token['bbox'] = bbox

        cropped_table['image'] = cropped_img
        cropped_table['tokens'] = table_tokens

        table_crops.append(cropped_table)

    return table_crops

def plot_results(cells, cropped_table,class_to_visualize):
    if class_to_visualize not in structure_model.config.id2label.values():
      raise ValueError("Class should be one of the available classes")

    plt.figure(figsize=(16,10))
    plt.imshow(cropped_table)
    ax = plt.gca()

    for cell in cells:
        score = cell["score"]
        bbox = cell["bbox"]
        label = cell["label"]

        if label == class_to_visualize:
          xmin, ymin, xmax, ymax = tuple(bbox)

          ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color="red", linewidth=3))
          text = f'{cell["label"]}: {score:0.2f}'
          ax.text(xmin, ymin, text, fontsize=15,
                  bbox=dict(facecolor='yellow', alpha=0.5))
          plt.axis('off')
def get_cell_coordinates_by_row(table_data):
    # Extract rows and columns
    rows = [entry for entry in table_data if entry['label'] == 'table row']
    columns = [entry for entry in table_data if entry['label'] == 'table column']

    # Sort rows and columns by their Y and X coordinates, respectively
    rows.sort(key=lambda x: x['bbox'][1])
    columns.sort(key=lambda x: x['bbox'][0])

    # Function to find cell coordinates
    def find_cell_coordinates(row, column):
        cell_bbox = [column['bbox'][0], row['bbox'][1], column['bbox'][2], row['bbox'][3]]
        return cell_bbox

    # Generate cell coordinates and count cells in each row
    cell_coordinates = []

    for row in rows:
        row_cells = []
        for column in columns:
            cell_bbox = find_cell_coordinates(row, column)
            row_cells.append({'column': column['bbox'], 'cell': cell_bbox})

        # Sort cells in the row by X coordinate
        row_cells.sort(key=lambda x: x['column'][0])

        # Append row information to cell_coordinates
        cell_coordinates.append({'row': row['bbox'], 'cells': row_cells, 'cell_count': len(row_cells)})

    # Sort rows from top to bottom
    cell_coordinates.sort(key=lambda x: x['row'][1])
    

    return cell_coordinates


def detect_tables(file_path):
    print(model.eval())
    print(model.config.id2label)
    #print(os.getcwd())
    image = Image.open(os.path.join(os.getcwd(),file_path)).convert("RGB")
    width, height = image.size
    img_new=image.resize((int(0.6*width), (int(0.6*height))))
    print(img_new)
    img_new.show()
    pixel_values = detection_transform(image).unsqueeze(0)
    pixel_values = pixel_values.to(device)
    print(pixel_values.shape)
    with torch.no_grad():
        outputs = model(pixel_values)
    print(outputs.logits.shape)  
    id2label = model.config.id2label
    id2label[len(model.config.id2label)] = "no object"  
    objects = outputs_to_objects(outputs, image.size, id2label)
    print(Text.BOLD_START+Text.RED +'Printing Label ,Score, BBox !!!!!!!!!!!!!!' +Text.END )
    print(objects)
    fig = visualize_detected_tables(image, objects)
    visualized_image = fig2img(fig)
    #print(objects)
    tokens = []
    detection_class_thresholds = {
        "table": 0.5,
        "table rotated": 0.5,
        "no object": 10
    }
    crop_padding = 15

    tables_crops = objects_to_crops(image, tokens, objects, detection_class_thresholds, padding=crop_padding)
    cropped_table_list=[]
    for i in range(len(tables_crops)):

        cropped_table = tables_crops[i]['image'].convert("RGB")
        cropped_table.show()
        cropped_table.save("cropped_table_"+str(i)+".jpg")
        cropped_table_list.append(cropped_table)
    print(Text.BOLD_START+Text.GREEN +'Table structure recog model will run from here #####' +Text.END )
    print(structure_model.config.id2label)
    outputs_l=[]
    for itm in cropped_table_list:
        pixel_values = structure_transform(itm).unsqueeze(0)
        # pixel_values = structure_transform(cropped_table_list[0]).unsqueeze(0)
        pixel_values = pixel_values.to(device)
        print(pixel_values.shape)
        with torch.no_grad():
            outputs = structure_model(pixel_values)
        outputs_l.append(outputs)
    structure_id2label = structure_model.config.id2label
    structure_id2label[len(structure_id2label)] = "no object"
    cell_l=[]
    for output , cropped_table in zip(outputs_l,cropped_table_list):
        cells = outputs_to_objects(output, cropped_table.size, structure_id2label)
        # cells = outputs_to_objects(outputs, cropped_table_list[0].size, structure_id2label)
        cell_l.append(cells)
    print(cell_l)    
     
    cropped_table_visualized = cropped_table_list[0].copy()
    print(Text.BOLD_START+Text.GREEN +'Cropped table :::#####' +Text.END )
    print(cropped_table_list[0])
    draw = ImageDraw.Draw(cropped_table_visualized)

    for cell in cells:
        draw.rectangle(cell["bbox"], outline="green")

    cropped_table_visualized.show()
    plot_results(cells, cropped_table_list[0],class_to_visualize="table row")
    cell_coordinates = [get_cell_coordinates_by_row(cells) for cells in cell_l]
    print(Text.BOLD_START+Text.CYAN +'Length of the cell coordinates' +Text.END )
    print(len(cell_coordinates))
    # for row in cell_coordinates:
    #     print(row["cells"])
    #print(len(cropped_table))
    print('%'*123)    
    print(cropped_table_list)    
    for idx, itm in  enumerate(cropped_table_list):
        data = apply_ocr(cell_coordinates[idx],cropped_table_list[idx],idx)
    return 'Dataframes created successfully'
    # df =pd.DataFrame(data)
    # df.to_csv('table_data.csv',index=False)

if __name__ == '__main__':
    #pass

    detect_tables('table__3.png')
    #pdf_to_image(pdf_path, page_number, image_path)
