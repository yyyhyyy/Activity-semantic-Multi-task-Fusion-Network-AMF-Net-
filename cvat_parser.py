# -*- coding: utf-8 -*-
"""Parser for CVAT for video 1.1 XML annotation files.

Notes:
- Scene labels such as desk layout, teaching mode, and teacher interaction object are frame-level annotations.
- Teacher location is stored as an attribute of teacher-action shapes/tracks in CVAT and is parsed from the same shape.
- Student view is stored as an attribute of student-action shapes/tracks in CVAT and is parsed from the same shape.
"""
from pathlib import Path
from collections import defaultdict

try:
    from lxml import etree
    HAS_LXML = True
except ImportError:
    import xml.etree.ElementTree as etree
    HAS_LXML = False

from config import TASKS, ALIAS_MAP


def _normalize_label_value(val):
    """Normalize a label value and apply annotation aliases."""
    if not val:
        return None
    val = val.strip()
    return ALIAS_MAP.get(val, val)


def _task_for_attribute(attr_name):
    """Map an XML attribute name or label value to a task name."""
    attr_lower = attr_name.strip().lower()
    if attr_lower in ("scene_desk", "desk", "桌椅排列"):
        return "scene_desk"
    if attr_lower in ("scene_method", "method", "scne_mothod", "教学模式"):
        return "scene_method"
    if attr_lower in ("scene_inte", "inte", "scne_inte", "教师交互对象"):
        return "scene_inte"
    if attr_lower in ("teacher_act", "teacher_act_exp", "teacher_act_ques", "教师行为"):
        return "teacher_act"
    if attr_lower in ("location", "教师位置", "位置"):
        return "location"
    if attr_lower in ("stu_act", "student_act", "学生行为"):
        return "stu_act"
    if attr_lower in ("view", "学生视线", "视线"):
        return "view"
    for task_name, (labels, _) in TASKS.items():
        if attr_name in labels or _normalize_label_value(attr_name) in labels:
            return task_name
    return None


def _get_attr(el, name, default=None):
    """Read an XML element attribute."""
    if hasattr(el, "get"):
        return el.get(name, default)
    return getattr(el, "attrib", {}).get(name, default)


def _iter_children(parent, tag=None):
    """Iterate over child nodes, optionally filtering by tag."""
    for c in parent:
        if hasattr(c, "tag"):
            if tag is None or c.tag == tag or (isinstance(c.tag, str) and c.tag.endswith(tag)):
                yield c


def parse_cvat_video_xml(xml_path, frames_dir=None):
    """Parse a CVAT video annotations.xml file.

    Returns a list of dictionaries with frame id, image path, parsed labels, and a bbox flag.
    """
    xml_path = Path(xml_path)
    if not xml_path.exists():
        raise FileNotFoundError(f"annotation file does not exist: {xml_path}")

    with open(xml_path, "rb") as f:
        tree = etree.parse(f)
    root = tree.getroot()

    def _local_tag(el):
        t = getattr(el, "tag", None)
        if isinstance(t, str) and "}" in t:
            return t.split("}", 1)[1]
        return t

    def find_all(node, tag):
        out = node.findall(f".//{tag}")
        if not out:
            out = [el for el in node.iter() if _local_tag(el) == tag]
        return out

    frames = []

    # Mode 1: one <image> element per frame, common when CVAT exports with images.
    for img_el in find_all(root, "image"):
        img_id = _get_attr(img_el, "id")
        name = _get_attr(img_el, "name")
        if not name:
            continue
        frame_labels = _extract_labels_from_element(img_el)
        image_path = Path(name).name if frames_dir else name
        frames.append({
            "frame_id": img_id,
            "image_path": image_path,
            "labels": frame_labels,
            "has_bbox": _has_bbox(img_el),
        })

    # Mode 2: track-based annotations that need to be expanded by frame.
    if not frames:
        frames = _parse_tracks(root, find_all, frames_dir)

    return frames


def _extract_labels_from_element(container_el):
    """Extract task-to-label mappings from an image or shape container."""
    out = {}
    for attr in _iter_children(container_el, "attribute"):
        name = _get_attr(attr, "name")
        val = _get_attr(attr, "value") or (getattr(attr, "text") and attr.text and attr.text.strip())
        if not val or not name:
            continue
        val = _normalize_label_value(val)
        task = _task_for_attribute(name)
        if task and task in TASKS:
            out[task] = val

    for shape in _iter_children(container_el):
        label_type = _get_attr(shape, "label") or (getattr(shape, "tag", "") or "")
        if isinstance(label_type, str) and "}" in label_type:
            label_type = label_type.split("}")[-1]
        for attr in _iter_children(shape, "attribute"):
            name = _get_attr(attr, "name")
            val = _get_attr(attr, "value") or (getattr(attr, "text") and attr.text and attr.text.strip())
            if not val or not name:
                continue
            val = _normalize_label_value(val)
            task = _task_for_attribute(name)
            if task and task in TASKS:
                out[task] = val
        if label_type:
            task = _task_for_attribute(label_type)
            if task and task in TASKS and task not in out:
                out[task] = _normalize_label_value(label_type)
    return out


def _has_bbox(container_el):
    """Return whether the container includes any spatial annotation shape."""
    for shape in _iter_children(container_el):
        if shape.tag in ("box", "polygon", "polyline", "points"):
            return True
    return False


def _parse_tracks(root, find_all, frames_dir):
    """Parse <track> and <tag> nodes into frame-level labels."""
    frame_data = defaultdict(lambda: {"labels": {}, "has_bbox": False})

    def _apply_tag_to_frame(frame_id: int, tag_el, inherited_label: str | None = None):
        """Apply a frame-level <tag> annotation to one frame."""
        label = _get_attr(tag_el, "label") or inherited_label or ""
        if isinstance(label, str) and "}" in label:
            label = label.split("}")[-1]

        for attr in _iter_children(tag_el, "attribute"):
            name = _get_attr(attr, "name")
            val = _get_attr(attr, "value") or (attr.text and attr.text.strip())
            if not name or not val:
                continue
            val = _normalize_label_value(val)
            task = _task_for_attribute(name)
            if task and task in TASKS:
                frame_data[frame_id]["labels"][task] = val

        if label:
            task = _task_for_attribute(label)
            if task and task in TASKS and task not in frame_data[frame_id]["labels"]:
                frame_data[frame_id]["labels"][task] = _normalize_label_value(label)

    for tag_el in find_all(root, "tag"):
        frame_num = _get_attr(tag_el, "frame")
        if frame_num is None:
            continue
        _apply_tag_to_frame(int(frame_num), tag_el, inherited_label=None)

    for track in find_all(root, "track"):
        label_type = _get_attr(track, "label")
        for child in track:
            child_tag = getattr(child, "tag", "")
            if isinstance(child_tag, str) and "}" in child_tag:
                child_tag = child_tag.split("}")[-1]

            if child_tag == "box":
                frame_num = _get_attr(child, "frame")
                if frame_num is None:
                    continue
                frame_id = int(frame_num)
                frame_data[frame_id]["has_bbox"] = True
                for attr in _iter_children(child, "attribute"):
                    name = _get_attr(attr, "name")
                    val = _get_attr(attr, "value") or (attr.text and attr.text.strip())
                    if not val:
                        continue
                    val = _normalize_label_value(val)
                    task = _task_for_attribute(name or label_type or "")
                    if task and task in TASKS:
                        frame_data[frame_id]["labels"][task] = val
                if label_type:
                    task = _task_for_attribute(label_type)
                    if task and task in TASKS and task not in frame_data[frame_id]["labels"]:
                        frame_data[frame_id]["labels"][task] = _normalize_label_value(label_type)

            elif child_tag == "tag":
                frame_num = _get_attr(child, "frame")
                if frame_num is None:
                    continue
                _apply_tag_to_frame(int(frame_num), child, inherited_label=label_type)

    if not frame_data:
        size = None
        try:
            meta = root.find(".//meta")
            if meta is not None:
                task = meta.find(".//task")
                if task is not None and task.find("size") is not None:
                    size = int(task.find("size").text)
        except Exception:
            size = None
        if size is not None:
            for fid in range(size):
                frame_data[fid]

    frames = []
    for frame_id in sorted(frame_data.keys()):
        info = frame_data[frame_id]
        if frames_dir:
            for fmt in (
                "frame_{:06d}.jpg",
                "{:06d}.jpg",
                "frame_{:06d}.JPG",
                "{:06d}.JPG",
                "frame_{:06d}.png",
                "{:06d}.png",
                "frame_{:06d}.PNG",
                "{:06d}.PNG",
            ):
                candidate = Path(frames_dir) / fmt.format(frame_id)
                if candidate.exists():
                    image_path = str(candidate)
                    break
            else:
                image_path = str(Path(frames_dir) / f"frame_{frame_id:06d}.jpg")
        else:
            image_path = f"frame_{frame_id:06d}.jpg"
        frames.append({
            "frame_id": frame_id,
            "image_path": image_path,
            "labels": info["labels"],
            "has_bbox": info["has_bbox"],
        })
    return frames


def get_task_indices_and_masks(frames, data_root):
    """Convert parsed frame labels into per-task class indices and valid masks."""
    rows = []
    for f in frames:
        row = {"frame_id": f["frame_id"], "image_path": f["image_path"], "labels_raw": f["labels"]}
        for task_name, (labels, to_idx) in TASKS.items():
            val = f["labels"].get(task_name)
            idx = to_idx.get(val) if val else -1
            row[f"{task_name}_idx"] = idx
            row[f"{task_name}_valid"] = idx >= 0
        rows.append(row)
    return rows
