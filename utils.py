import cv2, time
import PIL.ImageColor as ImageColor
import numpy as np


STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]



def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color = (255,0,0),
                               thickness = 2,
                               display_str_list = (),
                               display_distance = None,
                               meta = None,
                               use_normalized_coordinates=False,
                               font_scale = 0.5,
                               font_thickness = 1):
    im_height, im_width, _ = image.shape
    if use_normalized_coordinates:
        (left, right, top, bottom) = (int(xmin * im_width), int(xmax * im_width),
                                      int(ymin * im_height), int(ymax * im_height))
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    # print((left, right, top, bottom), image.shape)
    cv2.rectangle(image, (left, top), (right, bottom), color, thickness)

    if len(display_str_list) != 0:
        display_str_heights = [cv2.getTextSize(ds, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0][1] for ds in display_str_list]
        total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

        if top > total_display_str_height:
            text_bottom = top
        else:
            text_bottom = int(bottom + total_display_str_height)

        for display_str in display_str_list[::-1]:
            (text_width, text_height), _ = cv2.getTextSize(display_str, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            margin = int(np.ceil(0.05 * text_height))

            cv2.rectangle(image, (left, text_bottom - text_height - 2 * margin),
                          (left + text_width, text_bottom), color, -1)
            cv2.putText(image, display_str,
                        (left + margin, text_bottom - text_height//2 + margin),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

            text_bottom -= text_height - 2 * margin

    if display_distance is not None:
        display_distance = "{:.2f}".format(display_distance)
        # center_x, center_y = right + (left-right)//2, bottom + (top - bottom)//2

        (text_width, text_height), _ = cv2.getTextSize(display_distance, cv2.FONT_HERSHEY_DUPLEX, 0.6, 2)
        margin = int(np.ceil(0.05 * text_height))

        cv2.rectangle(image, (right - text_width - 2*margin, bottom - text_height - 2*margin),
                      (right, bottom), color, -1)

        mid_x = (xmax + xmin) / 2
        mid_y = (ymax + ymin) / 2
        warn_indicator = (0,0,255) if (0.3 < mid_x < 0.7) else (0,0,0)

        cv2.putText(image, display_distance,
                    (right - text_width - margin, bottom - 2*margin),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, warn_indicator, 2)

    if meta is not None:
        state, score = meta["state"], "{:.2f}".format(meta["estimator"])
        display_str_meta = "{}: {}".format(state,score)

        (text_width, text_height), _ = cv2.getTextSize(display_str_meta, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
        margin = int(np.ceil(0.05 * text_height))

        cv2.rectangle(image, (left, top),
                      (left + text_width + 2*margin, top + text_height + 2*margin), color, -1)

        cv2.putText(image, display_str_meta,
                    (left + margin, top + text_height + 2*margin),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,0,0), 1)

def visualize_boxes_on_image(
    img,
    predictions,
    show_labels = True,
    show_distances = True,
    show_meta = True,
    ):
    for prediction in predictions:
        ymin, xmin, ymax, xmax = prediction['box']
        color = STANDARD_COLORS[prediction['id'] % 110]
        color = ImageColor.getrgb(color) #bgr to opencv
        display_str = [prediction['name']] if show_labels else []
        distance = prediction['distance'] if show_distances else None
        meta = prediction['meta'] if (show_meta and 'meta' in prediction) else None
        draw_bounding_box_on_image(img,
                                    ymin,
                                    xmin,
                                    ymax,
                                    xmax,
                                    display_str_list = display_str,
                                    display_distance = distance,
                                    color = color,
                                    meta = meta,
                                    thickness=2)
    return img
