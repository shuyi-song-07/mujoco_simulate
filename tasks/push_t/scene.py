"""Compile source-frozen Push-T geometry; names are the only robot XML adaptation."""
from pathlib import Path
import copy
import math
import xml.etree.ElementTree as ET
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[2]


def initial_state(config, seed):
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    p = config['scene']['push_t']; rng = np.random.default_rng(seed)
    radius = np.hypot(.15, .15); goal_radius = np.hypot(.155, .155)
    separation = max(p['object_min_separation_m'], radius + goal_radius + .005)
    def safe(xy, half):
        return (.34 <= np.linalg.norm(xy) <= p['safe_reach_radius_m']
                and .1 + half <= xy[0] <= .9 - half and abs(xy[1]) <= .5 - half)
    for _ in range(1000):
        obj = rng.uniform([p['object_x_range'][0], p['object_y_range'][0]], [p['object_x_range'][1], p['object_y_range'][1]])
        target = rng.uniform([p['target_x_range'][0], p['target_y_range'][0]], [p['target_x_range'][1], p['target_y_range'][1]])
        if safe(obj, radius) and safe(target, goal_radius) and np.linalg.norm(obj-target) >= separation:
            break
    else:
        raise ValueError('Push-T randomization cannot satisfy source separation/reach constraints')
    yaw = float(np.deg2rad(rng.uniform(*p['yaw_degrees'])))
    return {'seed': seed, 'object_position': [*obj, .415 + p['initial_object_clearance_m']],
            'object_yaw_rad': yaw, 'object_quaternion_wxyz': [math.cos(yaw/2), 0, 0, math.sin(yaw/2)],
            'target_position': [*target, .401], 'target_yaw_rad': 0.}


def configure_xml(root, config):
    """Add aliases and auxiliary cameras without changing source physics."""
    world = root.find('worldbody')
    cube = world.find("body[@name='cube']")
    if cube is not None:
        cube.set('name', 'cooperative_object')
        cube.find('freejoint').set('name', 'object_freejoint')
        cube.find("geom[@name='push_t_bar']").set('name', 'object_geom')
    target = world.find("body[@name='target_plate']")
    if target is not None: target.set('name', 'target')
    world.find("body[@name='table']/geom").set('name', 'table_geom')
    hand = root.find(".//body[@name='left_hand']")
    if hand is not None and hand.find("site[@name='left_ee']") is None:
        ET.SubElement(hand, 'site', name='left_ee', pos='0 0 0', size='.002', rgba='0 0 0 0')
    for name in config['scene']['cameras']:
        if world.find(f"camera[@name='{name}']") is None:
            ET.SubElement(world, 'camera', name=name, pos='0 0 2', fovy='45')


def create_model(config):
    scene = ET.parse(ROOT/'tasks/push_t/assets/push_t_scene.xml').getroot()
    panda = ET.parse(ROOT/'tasks/push_t/assets/panda.xml').getroot()
    # Namespace body/joint/actuator/tendon names and every corresponding reference.
    names = {node.get('name') for tag in ('body', 'joint', 'general', 'position', 'fixed')
             for node in panda.iter(tag) if node.get('name')}
    refs = {'name', 'joint', 'joint1', 'joint2', 'body', 'body1', 'body2', 'tendon', 'target'}
    for node in panda.iter():
        for key in refs:
            if node.get(key) in names: node.set(key, 'left_'+node.get(key))
    root = ET.Element('mujoco', model='single Panda Push-T')
    merge = {'asset', 'worldbody', 'visual', 'default', 'contact', 'equality', 'tendon', 'actuator'}
    for source in (panda, scene):
        for child in source:
            if child.tag in {'include', 'keyframe'}: continue
            existing = root.find(child.tag)
            if existing is not None and child.tag in merge:
                for entry in child: existing.append(copy.deepcopy(entry))
            elif existing is not None:
                existing.attrib.update(child.attrib)
            else: root.append(copy.deepcopy(child))
    root.find('compiler').set('meshdir', '')
    visual = root.find('visual')
    global_node = visual.find('global')
    if global_node is None: global_node = ET.SubElement(visual, 'global')
    global_node.set('offwidth', str(config['recording']['width']))
    global_node.set('offheight', str(config['recording']['height']))
    configure_xml(root, config)
    mesh_dir = ROOT/'simulation/mujoco/assets/franka_emika_panda/assets'
    assets = {node.get('file'): (mesh_dir/node.get('file')).read_bytes() for node in root.findall('asset/mesh')}
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'), assets=assets)
    # Convert MuJoCo's own free-camera convention into fixed auxiliary cameras.
    # This avoids guessing azimuth signs or using a different perspective model.
    data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    view = mujoco.MjvScene(model, maxgeom=2000); option = mujoco.MjvOption()
    for name, spec in config['scene']['cameras'].items():
        camera = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(camera)
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = spec['lookat']; camera.distance = spec['distance']
        camera.azimuth = spec['azimuth']; camera.elevation = spec['elevation']
        mujoco.mjv_updateScene(model, data, option, None, camera, mujoco.mjtCatBit.mjCAT_ALL, view)
        gl = view.camera[0]; cid = model.camera(name).id
        position = .5*(view.camera[0].pos + view.camera[1].pos)
        forward = np.asarray(gl.forward); up = np.asarray(gl.up)
        right = np.cross(forward, up); rotation = np.column_stack((right, up, -forward))
        model.cam_pos[cid] = position
        mujoco.mju_mat2Quat(model.cam_quat[cid], rotation.reshape(-1))
        model.cam_fovy[cid] = model.vis.global_.fovy
    return model


def reset(controller, object_position, yaw):
    return []


def preflight(config, seeds, demonstration_seeds=(), engineering_seeds=()):
    from simulation.mujoco.single_arm.task_controller import SingleArmTaskController
    controller = SingleArmTaskController(config)
    cases=[]
    for seed in seeds:
        controller.reset(int(seed)); controller.step(.4)
        status=controller.get_status()
        cases.append({'seed': int(seed), 'finite': bool(np.isfinite(controller.data.qpos).all()),
                      'initial_state': controller.initial_state, 'coverage': status['push_t_coverage']})
    return {'cases': cases, 'summary': {'count': len(cases), 'all_finite': all(c['finite'] for c in cases)}}
