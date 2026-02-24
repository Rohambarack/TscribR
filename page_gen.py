import io
import re
import colorsys
import random

from collections import defaultdict

import numpy as np
import pyarrow
from datasets import Dataset

from PIL import Image, ImageDraw,  ImageFilter, ImageFont

from trdg.generators import GeneratorFromStrings
from trdg.utils import mask_to_bboxes, get_text_width

from hyphen import Hyphenator
from hyphen.textwrap2 import fill

def calc_n_rows(size:int,page_height:int=2480)->int:
    """ 
    Calculates the number of rows of text based on size to roughly fit an A/5 page
    """
    return int(np.floor(page_height/size))

def count_avg_char_width(font: str, 
                         fontsize:int,
                         alphabet:str = "ABCDEFGHIJKLMNOPQRSTUVWXYZÆØÅ")-> int:
    
    """ 
    gives a rough estimate (mean) of the number of pixels a character takes in the current configuration pf parameters
    
    """
    image_font = ImageFont.truetype(font, fontsize)
    alphabet_low = alphabet.lower()
    #oversample lowercase letters
    est_set = alphabet+(alphabet_low*10) + "123456789!?'-. ¨^'*"
    text_width = np.mean([get_text_width(image_font, c) for c in est_set])

    return int(np.ceil(text_width))

def cut_text_to_rows(avg_size:int,
                     character_spacing,
                     text:str,
                     n_rows:int,
                     hyphenator,
                     page_width:int = 1754,
                     margin:int = 10) -> list[str]:
    """
    cuts up the text to rows based on font and fontsize. Cuts up words, when neccessary.
    """
    #remove newlines
    text = re.sub("\n"," ",text)
    #available space
    page_width = page_width-margin
    #consider extra spacing
    avg_size = avg_size + round(avg_size) * character_spacing
    n_chars = round((page_width/avg_size))

    text = fill(text, width=n_chars, use_hyphenator=hyphenator)
    t_list = text.split("\n")
    
    t_list = t_list[:n_rows]
    #remove edge whitespaces, it breaks the mask retrieval
    t_list = [i.strip() for i in t_list]
    #also double whitespaces
    t_list = [re.sub(' +', ' ', i) for i in t_list]
    return t_list
    
def hsv_to_hex(h,s,v):
    """
    Turn hsv color values to hex
    """
    r, g, b = (int(255 * component) for component in colorsys.hsv_to_rgb(h / 255, s / 255, v / 255))
    return f"#{r:02x}{g:02x}{b:02x}"


def pad_point(point:list[int],padding:int,dir:int)->list[int]:
    """
    moves a point by some pixels in some direction
    0: up
    1: down
    2: left
    3: right
    """
    point = list(point)
    assert int(dir) < 4, "The direction value must be 0,1,2 or 3"
    if dir == 0:
        point[1] = point[1] - padding
    if dir == 1:
        point[1] = point[1] + padding
    if dir == 2:
        point[0] = point[0] - padding
    if dir == 3:
        point[0] = point[0] + padding
    return point

def points_from_bboxes(text_mask,padding:int=5) -> list[int]:
    """creates multi pont edges around the line of text"""
    bboxes = mask_to_bboxes(text_mask)
    #points
    upper_p = []
    lower_p = []
    #polygon from bboxes
    for idx_i, i in enumerate(bboxes):
        A = i[0],i[1]
        A = pad_point(A,padding,0)
        B = i[2],i[1]
        B = pad_point(B,padding,0)
        C = i[0],i[3]
        C = pad_point(C,padding,1)
        D = i[2],i[3]
        D = pad_point(D,padding,1)
        if idx_i == 0:
            A = pad_point(A,padding,2)
            C = pad_point(C,padding,2)
        elif idx_i == len(bboxes)-1:
            B = pad_point(B,padding,3)
            D = pad_point(D,padding,3)
        upper_p.append(A)
        upper_p.append(B)
        lower_p.append(C)
        lower_p.append(D)
    points = upper_p + lower_p[::-1]
    return points

def cut_out_line(image, mask) -> Image:

    """
    Takes the mask of the text, adds an alpha channel and cuts the page so only the text and its surroundings are kept
    """
    
    og_image = image.copy()
    points = points_from_bboxes(mask)
    draw = ImageDraw.Draw(image)
    draw.polygon(points, (0,0,0), width=1)
    #make array
    a = np.array(image)
    #grayscale (keep only one col channel)
    a = np.where(a > 1, 255, a)
    a = a[:,:,0]
    #flip black and white for alpha ch use 
    where_0 = np.where(a == 0)
    where_255 = np.where(a == 255)
    a[where_0] = 255
    a[where_255] = 0
    #trim zeros
    a = np.trim_zeros(a,axis=1)
    #feather the edges a bit
    a = Image.fromarray(a)
    feather_filt = ImageFilter.GaussianBlur(1)
    a = Image.fromarray(np.where(np.array(a.filter(feather_filt)) > 1,
                                255,
                                    a))
    #apply mask back to image
    w,h = a.size
    og_image = og_image.crop([0,0,w,h])
    og_image.putalpha(a)
    
    return og_image

def separate_splits(n_lines:int,
                    splits:list[float]=[0.7,0.1,0.2]) -> list[int]:
    """ 
    Splits an integer roughly to the given split percentages
    splits = [train,validation,test]
    """
    #if n_lines less than 10, all go to train
    if n_lines<10:
        return[10,0,0]
    
    assert sum(splits) == 1, "Split ratios must add up to 1" 

    line_ids = list(range(n_lines))
    #separate train
    n_train = round(n_lines*splits[0])
    split_1 = [line_ids[i:i + n_train] for i in range(0, len(line_ids), n_train)]
    #separate validation
    n_test = round(n_lines*splits[2])
    split_2 = [split_1[1][i:i + n_test] for i in range(0, len(split_1[1]), n_test)]
    #
    return [len(split_1[0]),len(split_2[1]),len(split_2[0])]

def sample_ids_by_split(n_lines:int,
                        splits:list[int] = [0.7,0.1,0.2]) -> list[list[int]]:
    """
    Samples ids for the lines based on the number
    """
    if n_lines < 10:
        return [list(range(10)),[],[]]

    line_ids = list(range(n_lines))
    n_splits = separate_splits(n_lines,splits)
    split_ids =[]
    taken_ids = []
    for i in n_splits:
        temp_ids = [item for item in line_ids if item not in taken_ids]
        #sample the number from the ids
        ids_for_a_split = random.sample(temp_ids,i)
        #Do not to sample it twice
        [taken_ids.append(i) for i in ids_for_a_split]
        split_ids.append(ids_for_a_split)

    return split_ids
    
def generate_page(text:str,
                  font_path:str,
                  ofl:int = 0,
                  lang:str = "da_DK")->list[dict]:

    """
    Docstring for generate_page

    The function return a list of dictionaries in the example format:

    [
    {line:{
            text: "the text line which is written on the image
            im:  "the PNG bytes of the image"
        },
        train:bool,
        validation:bool,
        test:bool,
        },
    {line:{
            text: "the text line which is written on the image
            im:  "the PNG bytes of the image"
        },
        train:bool,
        validation:bool,
        test:bool,
        }...
    ]

    :param text: the text to be printed on the pages
    :type text: str

    :param font_path: the path to the .ttf file
    :type font_path: str

    :param ofl: Is the font under a license, which allows the modification of the font, if 1, the text will be modified also, not just the background
    :type ofl: int

    :param lang: which language to use for the hyphenating of the text. look up the "hyphen" package for codes to use for a language.
    :type lang: str
    """

    #initalize
    img_list = []
    mask_list = []
    post_gen_text = []
    h_dk = Hyphenator(lang)
    r_font = [font_path]
    #random color for fonts (grayish or brownish)
    r_color = np.random.randint(0,2)
    r_hsv_val = np.random.randint(0,50)
    if r_color ==0:
        r_font_color = hsv_to_hex(15,255,r_hsv_val)
    else:
        r_font_color = hsv_to_hex(0,0,r_hsv_val)
    #randomize page vals
    #background color (white/yellowish)
    rand_r = np.random.randint(250,255)
    rand_g = np.random.randint(235,255)
    rand_b = np.random.randint(200,255)
    #font size, the width of " ", and general spacing
    r_size_meta = np.random.randint(0,2)
    if r_size_meta == 0:
        rand_size = np.random.randint(30,50)
    else:
        rand_size = np.random.randint(50,80) #40,80
    rand_space_w = np.random.uniform(0.5,1.5)
    rand_spacing = np.random.randint(0,20)


    #if font license allows modification, bold, skew, blur, distort too
    if ofl==1:
        rand_stroke_w = np.random.randint(0,3)
        #randomize skew direction
        r_meta_skew = np.random.randint(0,2)
        #randomize skew degree (bigger fonts can take more)
        if rand_size > 70:
            if r_meta_skew == 0:
                r_skew = np.random.randint(0,5)
            else:
                r_skew = np.random.randint(356,361)
        else:
            if r_meta_skew == 0:
                r_skew = np.random.randint(0,3)
            else:
                r_skew = np.random.randint(358,361)
        #randomize blur, blur applied at the end, because it messes up masks 
        if rand_size > 70:
            r_blur = np.random.randint(0,3)
        else:
            r_blur = np.random.randint(0,2)      
        #randomize distortion
        r_distort = np.random.randint(0,4)
        r_distort_or = np.random.randint(0,3)
    elif ofl==0:
        rand_stroke_w = 0
        r_skew=0
        r_blur=0
        r_distort=0
        r_distort_or=0

    #calculate rows to split the text to fit roughly on an A/5 page
    n_rows = calc_n_rows(rand_size)
    avg_size = count_avg_char_width(r_font[0], rand_size)
    text_list = cut_text_to_rows(avg_size,
                                rand_space_w,
                                text,
                                n_rows,
                                h_dk)

    generator_from_str = GeneratorFromStrings(text_list,
                                            count=len(text_list),
                                            fonts=r_font,
                                            size=rand_size,
                                            background_type=3,
                                            RGB_vals=[rand_r,rand_g,rand_b],                      width=1754,  
                                            alignment=0,
                                            space_width=rand_space_w,
                                            character_spacing=rand_spacing,
                                            margins=(5,5,5,5),
                                            stroke_width=rand_stroke_w,
                                            skewing_angle=r_skew,
                                            blur=0,
                                            distorsion_type=r_distort,
                                            distorsion_orientation=r_distort_or,
                                            text_color=r_font_color,
                                            stroke_fill=r_font_color,
                                            output_mask=1,
                                            fit=True


                                            )
    for img, text in generator_from_str:
        post_gen_text.append(text)
        img_list.append(img[0])
        mask_list.append(img[1])

    #assign training labels to the data
    label_dicts = []
    id_splits = sample_ids_by_split(n_rows)
    for id_s, split in enumerate(id_splits):
        #basic setup
        split_setup = {"train":False,
                "validation":False,
                "test":False}
        if id_s == 0:
            lab = {"train":True}
        elif id_s == 1:
            lab = {"validation":True}
        elif id_s == 2:
            lab = {"test":True}
        split_setup.update(lab)
        label_dicts.append({"id":split,"split":split_setup})
        
        
    all_lines =[]
    #assemble dataset
    for idx_image, im in enumerate(img_list):
        #keep text outlines only
        im = cut_out_line(im,mask_list[idx_image])
        #blur after to not mess up the mask
        gaussian_filter = ImageFilter.GaussianBlur(
                radius=r_blur 
            )
        im = im.filter(gaussian_filter)

        #convert to bytes
        img_byte_arr = io.BytesIO()
        im.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        #find train validation test split
        if idx_image in label_dicts[0]["id"]:
            train_split = label_dicts[0]["split"]
        elif idx_image in label_dicts[1]["id"]:
            train_split = label_dicts[1]["split"]
        elif idx_image in label_dicts[2]["id"]:
            train_split = label_dicts[2]["split"]
        
        
        line_dict = {"lines":{"text":post_gen_text[idx_image],
                            "im":img_byte_arr}}
        line_dict.update(train_split)
        all_lines.append(line_dict)

    return all_lines 


def count_chars_inline(row_text,char_count:defaultdict) -> defaultdict:
    """
    Counts charachter occurences for the metadata
    """
    for i in row_text:
        char_count[i] += 1

    return char_count


def count_splits(split_column,
                 col_type:str,
                 split_count:defaultdict) -> defaultdict:
    """  
    Count the splits in the arrow table by column.
    split_column: an array of true and false values
    col_type: train, validation or test
    split_count: the dictionary holding the counts 
    """
    for line in split_column:
        if line:
            split_count[col_type] += 1
        else:
            pass

    return split_count

def gather_table_metadata(table:pyarrow.Table) -> dict:
    """  
    Gathers the metadata from the generated lines in a format, compatible with kraken.
    """
    #collect metadata
    meta_data = {}
    #gather text
    text = []
    for i in table.column(0):
        text.append(str(i.get("text")))

    #count charachters in text
    char_count = defaultdict(int)
    for i in text:
        char_count = count_chars_inline(i,char_count)

    #count splits
    split_count = defaultdict(int)

    for i,j in zip([table.column(1),table.column(2),table.column(3)],
                ["train","validation","test"]):
        split_count = count_splits(i,j,split_count)

    meta_data.update({
        "type": "kraken_recognition_baseline",
        "alphabet":dict(char_count),
        "text_type": "raw",
        "image_type": "raw",
        "splits": ["train", "eval", "test"], 
        "im_mode": "RGBA",
        "legacy_polygons": False,
        "counts": {"all": len(text)}
        })
    meta_data["counts"].update(dict(split_count))

    return meta_data
        