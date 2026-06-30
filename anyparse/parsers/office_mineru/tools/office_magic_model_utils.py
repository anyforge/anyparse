# Copyright (c) 2026 MinerU Team. All rights reserved.
# === AnyParse Modifications ===
# Modified by AnyParse Team (2026)
# ==================================

import math
from typing import List, Dict, Any, Callable


def is_in(box1, box2) -> bool:
    """box1是否完全在box2里面."""
    x0_1, y0_1, x1_1, y1_1 = box1
    x0_2, y0_2, x1_2, y1_2 = box2

    return (
        x0_1 >= x0_2  # box1的左边界不在box2的左边外
        and y0_1 >= y0_2  # box1的上边界不在box2的上边外
        and x1_1 <= x1_2  # box1的右边界不在box2的右边外
        and y1_1 <= y1_2
    )  # box1的下边界不在box2的下边外


def bbox_relative_pos(bbox1, bbox2):
    """判断两个矩形框的相对位置关系.

    Args:
        bbox1: 一个四元组，表示第一个矩形框的左上角和右下角的坐标，格式为(x1, y1, x1b, y1b)
        bbox2: 一个四元组，表示第二个矩形框的左上角和右下角的坐标，格式为(x2, y2, x2b, y2b)

    Returns:
        一个四元组，表示矩形框1相对于矩形框2的位置关系，格式为(left, right, bottom, top)
        其中，left表示矩形框1是否在矩形框2的左侧，right表示矩形框1是否在矩形框2的右侧，
        bottom表示矩形框1是否在矩形框2的下方，top表示矩形框1是否在矩形框2的上方
    """
    x1, y1, x1b, y1b = bbox1
    x2, y2, x2b, y2b = bbox2

    left = x2b < x1
    right = x1b < x2
    bottom = y2b < y1
    top = y1b < y2
    return left, right, bottom, top


def bbox_distance(bbox1, bbox2):
    """计算两个矩形框的距离。

    Args:
        bbox1 (tuple): 第一个矩形框的坐标，格式为 (x1, y1, x2, y2)，其中 (x1, y1) 为左上角坐标，(x2, y2) 为右下角坐标。
        bbox2 (tuple): 第二个矩形框的坐标，格式为 (x1, y1, x2, y2)，其中 (x1, y1) 为左上角坐标，(x2, y2) 为右下角坐标。

    Returns:
        float: 矩形框之间的距离。
    """

    def dist(point1, point2):
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    x1, y1, x1b, y1b = bbox1
    x2, y2, x2b, y2b = bbox2

    left, right, bottom, top = bbox_relative_pos(bbox1, bbox2)

    if top and left:
        return dist((x1, y1b), (x2b, y2))
    elif left and bottom:
        return dist((x1, y1), (x2b, y2b))
    elif bottom and right:
        return dist((x1b, y1), (x2, y2b))
    elif right and top:
        return dist((x1b, y1b), (x2, y2))
    elif left:
        return x1 - x2b
    elif right:
        return x2 - x1b
    elif bottom:
        return y1 - y2b
    elif top:
        return y2 - y1b
    return 0.0


def bbox_center_distance(bbox1, bbox2):
    """计算两个矩形框中心点之间的欧氏距离。

    Args:
        bbox1 (tuple): 第一个矩形框的坐标，格式为 (x1, y1, x2, y2)
        bbox2 (tuple): 第二个矩形框的坐标，格式为 (x1, y1, x2, y2)

    Returns:
        float: 两个矩形框中心点之间的距离
    """
    x1, y1, x1b, y1b = bbox1
    x2, y2, x2b, y2b = bbox2

    # 计算中心点
    center1_x = (x1 + x1b) / 2
    center1_y = (y1 + y1b) / 2
    center2_x = (x2 + x2b) / 2
    center2_y = (y2 + y2b) / 2

    # 计算欧氏距离
    return math.sqrt((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2)


def get_minbox_if_overlap_by_ratio(bbox1, bbox2, ratio):
    """通过calculate_overlap_area_2_minbox_area_ratio计算两个bbox重叠的面积占最小面积的box的比例
    如果比例大于ratio，则返回小的那个bbox, 否则返回None."""
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    overlap_ratio = calculate_overlap_area_2_minbox_area_ratio(bbox1, bbox2)
    if overlap_ratio > ratio:
        if area1 <= area2:
            return bbox1
        else:
            return bbox2
    else:
        return None


def calculate_overlap_area_2_minbox_area_ratio(bbox1, bbox2):
    """计算box1和box2的重叠面积占最小面积的box的比例."""
    # Determine the coordinates of the intersection rectangle
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The area of overlap area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    min_box_area = min([(bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]),
                        (bbox2[3] - bbox2[1]) * (bbox2[2] - bbox2[0])])
    if min_box_area == 0:
        return 0
    else:
        return intersection_area / min_box_area


def calculate_iou(bbox1, bbox2):
    """计算两个边界框的交并比(IOU)。

    Args:
        bbox1 (list[float]): 第一个边界框的坐标，格式为 [x1, y1, x2, y2]，其中 (x1, y1) 为左上角坐标，(x2, y2) 为右下角坐标。
        bbox2 (list[float]): 第二个边界框的坐标，格式与 `bbox1` 相同。

    Returns:
        float: 两个边界框的交并比(IOU)，取值范围为 [0, 1]。
    """
    # Determine the coordinates of the intersection rectangle
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The area of overlap area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # The area of both rectangles
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    if any([bbox1_area == 0, bbox2_area == 0]):
        return 0

    # Compute the intersection over union by taking the intersection area
    # and dividing it by the sum of both areas minus the intersection area
    iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)

    return iou


def calculate_overlap_area_in_bbox1_area_ratio(bbox1, bbox2):
    """计算box1和box2的重叠面积占bbox1的比例."""
    # Determine the coordinates of the intersection rectangle
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The area of overlap area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    if bbox1_area == 0:
        return 0
    else:
        return intersection_area / bbox1_area


def calculate_vertical_projection_overlap_ratio(block1, block2):
    """
    Calculate the proportion of the x-axis covered by the vertical projection of two blocks.

    Args:
        block1 (tuple): Coordinates of the first block (x0, y0, x1, y1).
        block2 (tuple): Coordinates of the second block (x0, y0, x1, y1).

    Returns:
        float: The proportion of the x-axis covered by the vertical projection of the two blocks.
    """
    x0_1, _, x1_1, _ = block1
    x0_2, _, x1_2, _ = block2

    # Calculate the intersection of the x-coordinates
    x_left = max(x0_1, x0_2)
    x_right = min(x1_1, x1_2)

    if x_right < x_left:
        return 0.0

    # Length of the intersection
    intersection_length = x_right - x_left

    # Length of the x-axis projection of the first block
    block1_length = x1_1 - x0_1

    if block1_length == 0:
        return 0.0

    # Proportion of the x-axis covered by the intersection
    # logger.info(f"intersection_length: {intersection_length}, block1_length: {block1_length}")
    return intersection_length / block1_length


def reduct_overlap(bboxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    去除重叠的bbox，保留不被其他bbox包含的bbox

    Args:
        bboxes: 包含bbox信息的字典列表

    Returns:
        去重后的bbox列表
    """
    N = len(bboxes)
    keep = [True] * N
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if is_in(bboxes[i]['bbox'], bboxes[j]['bbox']):
                keep[i] = False
    return [bboxes[i] for i in range(N) if keep[i]]


def tie_up_category_by_index(
        get_subjects_func: Callable,
        get_objects_func: Callable,
        extract_subject_func: Callable = None,
        extract_object_func: Callable = None,
        object_block_type: str = "object",
        include_bbox: bool = True,
):
    """
    基于index的类别关联方法，用于将主体对象与客体对象进行关联
    客体优先匹配给index最接近的主体，匹配优先级为：
    1. index差值（最高优先级）
    2. bbox边缘距离（相邻边距离）
    3. bbox中心点距离（最低优先级，作为最终tiebreaker）

    参数:
        get_subjects_func: 函数，提取主体对象
        get_objects_func: 函数，提取客体对象
        extract_subject_func: 函数，自定义提取主体属性（默认使用bbox和其他属性）
        extract_object_func: 函数，自定义提取客体属性（默认使用bbox和其他属性）

    返回:
        关联后的对象列表，按主体index升序排列
    """
    subjects = get_subjects_func()
    objects = get_objects_func()

    # 如果没有提供自定义提取函数，使用默认函数
    if extract_subject_func is None:
        extract_subject_func = lambda x: x
    if extract_object_func is None:
        extract_object_func = lambda x: x

    # 初始化结果字典，key为主体索引，value为关联信息
    result_dict = {}

    # 初始化所有主体
    for i, subject in enumerate(subjects):
        result_dict[i] = {
            "sub_bbox": extract_subject_func(subject),
            "obj_bboxes": [],
            "sub_idx": i,
        }

    # 提取所有客体的index集合，用于计算有效index差值
    object_indices = set(obj["index"] for obj in objects)

    def calc_effective_index_diff(obj_index: int, sub_index: int) -> int:
        """
        计算有效的index差值
        有效差值 = 绝对差值 - 区间内其他客体的数量
        即：如果obj_index和sub_index之间的差值是由其他客体造成的，则应该扣除这部分差值
        """
        if obj_index == sub_index:
            return 0

        start, end = min(obj_index, sub_index), max(obj_index, sub_index)
        abs_diff = end - start

        # 计算区间(start, end)内有多少个其他客体的index
        other_objects_count = 0
        for idx in range(start + 1, end):
            if idx in object_indices:
                other_objects_count += 1

        return abs_diff - other_objects_count

    # 为每个客体找到最匹配的主体
    for obj in objects:
        if len(subjects) == 0:
            # 如果没有主体，跳过客体
            continue

        obj_index = obj["index"]
        min_index_diff = float("inf")
        best_subject_indices = []

        # 找出有效index差值最小的所有主体
        for i, subject in enumerate(subjects):
            sub_index = subject["index"]
            index_diff = calc_effective_index_diff(obj_index, sub_index)

            if index_diff < min_index_diff:
                min_index_diff = index_diff
                best_subject_indices = [i]
            elif index_diff == min_index_diff:
                best_subject_indices.append(i)

        if len(best_subject_indices) == 1:
            best_subject_idx = best_subject_indices[0]
        # 如果有多个主体的index差值相同（最多两个），根据边缘距离进行筛选
        elif len(best_subject_indices) == 2:
            # 只有在包含bbox信息时才进行边缘距离的计算和比较，否则直接匹配第一个主体
            if include_bbox:
                # 计算所有候选主体的边缘距离
                edge_distances = [(idx, bbox_distance(obj["bbox"], subjects[idx]["bbox"])) for idx in best_subject_indices]
                edge_dist_diff = abs(edge_distances[0][1] - edge_distances[1][1])
                if edge_dist_diff > 2:
                    # 边缘距离差值大于2，匹配边缘距离更小的主体
                    best_subject_idx = min(edge_distances, key=lambda x: x[1])[0]
                
                elif object_block_type == "table_caption":
                    # 边缘距离差值<=2且为table_caption，匹配index更大的主体
                    best_subject_idx = max(best_subject_indices, key=lambda idx: subjects[idx]["index"])

                elif object_block_type.endswith("footnote"):
                    # 边缘距离差值<=2且为footnote，匹配index更小的主体
                    best_subject_idx = min(best_subject_indices, key=lambda idx: subjects[idx]["index"])

                else:
                    # 边缘距离差值<=2 且不适用特殊匹配规则，使用中心点距离匹配
                    center_distances = [(idx, bbox_center_distance(obj["bbox"], subjects[idx]["bbox"])) for idx in best_subject_indices]
                    best_subject_idx = min(center_distances, key=lambda x: x[1])[0]
            else:
                best_subject_idx = best_subject_indices[0]
        else:
            raise ValueError("More than two subjects have the same minimal index difference, which is unexpected.")

        # 将客体添加到最佳主体的obj_bboxes中
        result_dict[best_subject_idx]["obj_bboxes"].append(extract_object_func(obj))

    # 转换为列表并按主体index排序
    ret = list(result_dict.values())
    ret.sort(key=lambda x: x["sub_idx"])

    return ret
