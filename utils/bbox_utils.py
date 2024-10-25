def get_center_of_bbox(bbox):
    """
    Calculate the center point of a bounding box.

    Args:
        bbox (list): List or tuple of bounding box coordinates [x1, y1, x2, y2]
                    where (x1, y1) is the top-left corner and 
                    (x2, y2) is the bottom-right corner

    Returns:
        tuple: (center_x, center_y) coordinates as integers
    """
    x1, y1, x2, y2 = bbox
    return int((x1 + x2)/2), int((y1 + y2)/2)

def get_bbox_width(bbox):
    """
    Calculate the width of a bounding box.

    Args:
        bbox (list): List or tuple of bounding box coordinates [x1, y1, x2, y2]
                    where x1 is the left edge and x2 is the right edge

    Returns:
        float: Width of the bounding box
    """
    return bbox[2] - bbox[0]

def measure_distance(p1, p2):
    """
    Calculate the Euclidean distance between two points.

    Args:
        p1 (tuple): First point coordinates (x1, y1)
        p2 (tuple): Second point coordinates (x2, y2)

    Returns:
        float: Euclidean distance between the two points
    """
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

def measure_xy_distance(p1, p2):
    """
    Calculate the separate x and y distances between two points.
    Useful for measuring displacement in each direction.

    Args:
        p1 (tuple): First point coordinates (x1, y1)
        p2 (tuple): Second point coordinates (x2, y2)

    Returns:
        tuple: (x_distance, y_distance) where positive values indicate
               p1 is to the right/below p2
    """
    return p1[0] - p2[0], p1[1] - p2[1]

def get_foot_position(bbox):
    """
    Calculate the position of an object's feet/base from its bounding box.
    Uses the bottom-center point of the bounding box.

    Args:
        bbox (list): List or tuple of bounding box coordinates [x1, y1, x2, y2]
                    where (x1, y1) is the top-left corner and 
                    (x2, y2) is the bottom-right corner

    Returns:
        tuple: (x, y) coordinates of the foot position as integers,
               where x is the horizontal center and y is the bottom of the bbox
    """
    x1, y1, x2, y2 = bbox
    return int((x1 + x2)/2), int(y2)