"""Rebuild namespaced Panda MJCF without changing or copying the source meshes.

Run from any directory with Python. Runtime geometry overrides live in config.
"""
from __future__ import annotations

import copy
from pathlib import Path
import xml.etree.ElementTree as ET


def build_scene() -> Path:
    directory = Path(__file__).resolve().parent
    source = ET.parse(directory.parent / "franka_emika_panda" / "panda.xml").getroot()
    root = ET.Element("mujoco", model="task3_bimanual_pick_carry_place")
    ET.SubElement(root, "compiler", angle="radian", meshdir="../franka_emika_panda/assets", autolimits="true")
    ET.SubElement(root, "option", timestep="0.002", integrator="implicitfast", iterations="80", noslip_iterations="5")
    ET.SubElement(root, "size", njmax="3000", nconmax="500")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="640", offheight="480")
    ET.SubElement(visual, "headlight", ambient="0.35 0.35 0.35", diffuse="0.7 0.7 0.7")
    root.append(copy.deepcopy(source.find("default")))
    root.append(copy.deepcopy(source.find("asset")))
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", name="task_light", pos="0 0 2", dir="0 0 -1", directional="true")
    ET.SubElement(world, "geom", name="floor", type="plane", size="2 2 0.05", rgba="0.18 0.22 0.28 1")
    table = ET.SubElement(world, "body", name="table", pos="0 0 0.175")
    ET.SubElement(table, "geom", name="table_geom", type="box", size="0.5 0.55 0.175", rgba="0.46 0.32 0.23 1", friction="1.5 0.1 0.02")
    for side in ("left", "right"):
        sign = -1 if side == "left" else 1
        pedestal = ET.SubElement(world, "body", name=f"{side}_pedestal", pos=f"0 {sign * 0.75} 0.175")
        ET.SubElement(pedestal, "geom", name=f"{side}_pedestal_geom", type="box", size="0.15 0.15 0.175", rgba="0.35 0.4 0.45 1")
        arm = copy.deepcopy(source.find("worldbody/body"))
        for index, node in enumerate(arm.iter()):
            if "name" in node.attrib:
                node.set("name", f"{side}_{node.get('name')}")
            elif node.tag == "geom":
                node.set("name", f"{side}_geom_{index}")
                if node.get("class", "").startswith("fingertip") or (node.get("class") == "collision" and node.get("mesh") == "finger_0"):
                    node.set("friction", "4 0.1 0.02")
                    node.set("condim", "6")
        arm.set("pos", f"0 {sign * 0.75} 0.35")
        arm.set("quat", f"0.70710678 0 0 {-sign * 0.70710678}")
        hand = next(node for node in arm.iter("body") if node.get("name") == f"{side}_hand")
        ET.SubElement(hand, "site", name=f"{side}_ee", pos="0 0 0.103", size="0.006", rgba="0 0.7 1 0.6")
        world.append(arm)
    obj = ET.SubElement(world, "body", name="cooperative_object", pos="0 0 0.372")
    ET.SubElement(obj, "freejoint", name="object_freejoint")
    ET.SubElement(obj, "geom", name="object_geom", type="box", size="0.018 0.27 0.022", mass="0.35", friction="3 0.1 0.02", condim="6", rgba="0.85 0.22 0.16 1")
    for side, sign in (("left", -1), ("right", 1)):
        ET.SubElement(obj, "site", name=f"{side}_grasp_site", pos=f"0 {sign * 0.22} 0", size="0.004", rgba="1 0.9 0 1")
    target = ET.SubElement(world, "body", name="target", pos="0.15 0 0.3505")
    ET.SubElement(target, "geom", name="target_geom", type="box", size="0.045 0.3 0.0004", rgba="0.2 0.85 0.3 0.65", contype="0", conaffinity="0")
    # Top/front show dimensions without depth-dependent scale. The oblique
    # third view separates the two grippers for contact inspection.
    ET.SubElement(world, "camera", name="top", pos="0.04 0 3.35", xyaxes="0 1 0 -1 0 0", projection="orthographic", fovy="0.62")
    for name, position, look_at in (("front", "3 0 0.53", "0 0 0.53"),
                                    ("side", "2.2 -3 1.65", "0.06 0 0.48")):
        ET.SubElement(world, "body", name=f"{name}_camera_target", pos=look_at)
        ET.SubElement(world, "camera", name=name, pos=position, mode="targetbody", target=f"{name}_camera_target", projection="orthographic" if name == "front" else "perspective", fovy="0.62" if name == "front" else "11.5")
    for section in ("tendon", "equality", "actuator", "contact"):
        target_section = ET.SubElement(root, section)
        for side in ("left", "right"):
            for index, element in enumerate(source.find(section)):
                clone = copy.deepcopy(element)
                for node in clone.iter():
                    for key in ("name", "joint", "joint1", "joint2", "tendon", "body1", "body2"):
                        if key in node.attrib:
                            node.set(key, f"{side}_{node.get(key)}")
                if section == "equality" and not clone.get("name"):
                    clone.set("name", f"{side}_finger_equality_{index}")
                target_section.append(clone)
    ET.indent(root)
    destination = directory / "dual_arm_scene.xml"
    ET.ElementTree(root).write(destination, encoding="utf-8", xml_declaration=True)
    return destination


if __name__ == "__main__":
    print(build_scene())
