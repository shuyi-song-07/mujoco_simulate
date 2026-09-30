#!/usr/bin/env python
"""Run a local LeRobot ACT policy in the Panda pick-and-place MuJoCo scene."""
from __future__ import annotations
import argparse, time
from pathlib import Path
import mujoco
import mujoco.viewer
import numpy as np
import torch
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

DEFAULT_SCENE = Path(__file__).parent / "assets/franka_emika_panda/scene.xml"
DEFAULT_POLICY = Path.home() / "桌面/outputs/train/act_50eps_pyav/checkpoints/last/pretrained_model"
CAMERAS = {"overview": 145.0, "camera_2": 25.0, "camera_3": 265.0}
INITIAL_QPOS = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8])


def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    r, p, y = rpy
    cr, cp, cy = np.cos([r, p, y]); sr, sp, sy = np.sin([r, p, y])
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                     [-sp, cp*sr, cp*cr]])


class Panda:
    def __init__(self, scene: Path, seed: int, width=640, height=480):
        self.model = mujoco.MjModel.from_xml_path(str(scene)); self.data = mujoco.MjData(self.model)
        self.ik = mujoco.MjData(self.model); self.rng = np.random.default_rng(seed)
        self.base = self.model.body("link0").id; self.hand = self.model.body("hand").id
        self.cube = self.model.body("cube").id; self.plate = self.model.body("target_plate").id
        self.jids = [self.model.joint(f"joint{i}").id for i in range(1, 8)]
        self.qadr = np.array([self.model.jnt_qposadr[j] for j in self.jids]); self.dadr = np.array([self.model.jnt_dofadr[j] for j in self.jids])
        self.fingers = [self.model.jnt_qposadr[self.model.joint(f"finger_joint{i}").id] for i in (1, 2)]
        self.cube_adr = self.model.jnt_qposadr[self.model.body_jntadr[self.cube]]
        self.jp = np.zeros((3, self.model.nv)); self.jr = np.zeros_like(self.jp)
        self.cameras = {}
        for name, azimuth in CAMERAS.items():
            cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam); cam.lookat[:] = [.48, 0, .48]; cam.distance = 1.45; cam.azimuth = azimuth; cam.elevation = -15
            self.cameras[name] = cam
        self.width, self.height = width, height; self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        for _ in range(1000):
            cube = self.rng.uniform([.44, -.07], [.51, .07]); plate = self.rng.uniform([.54, -.13], [.62, .09])
            if np.linalg.norm(cube - plate) >= .16: break
        self.data.qpos[self.cube_adr:self.cube_adr+2] = cube; self.model.body_pos[self.plate, :2] = plate
        self.data.qpos[self.qadr] = INITIAL_QPOS; self.data.qpos[self.fingers] = .04; self.data.ctrl[:7] = INITIAL_QPOS; self.data.ctrl[7] = 255.
        mujoco.mj_forward(self.model, self.data)

    def observation(self, renderer):
        state = np.r_[self.data.qpos[self.qadr], self.data.qpos[self.fingers].sum()].astype(np.float32)
        obs = {"observation.state": torch.from_numpy(state)}
        for name, cam in self.cameras.items():
            renderer.update_scene(self.data, camera=cam); rgb = renderer.render().copy()
            obs[f"observation.images.{name}"] = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        return obs

    def apply_action(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.size != 7 or not np.isfinite(action).all(): raise ValueError(f"ACT action must be 7 finite values, got {action}")
        self.ik.qpos[:] = self.data.qpos; self.ik.qpos[self.qadr] = self.data.ctrl[:7]; mujoco.mj_forward(self.model, self.ik)
        base_r = self.ik.xmat[self.base].reshape(3, 3); target_p = self.ik.xpos[self.base] + base_r @ action[:3]; target_r = base_r @ rpy_to_matrix(action[3:6])
        for _ in range(20):
            cur_r = self.ik.xmat[self.hand].reshape(3, 3)
            err = np.r_[target_p - self.ik.xpos[self.hand], .5 * sum(np.cross(cur_r[:, i], target_r[:, i]) for i in range(3))]
            if np.linalg.norm(err) < 1e-4: break
            mujoco.mj_jacBody(self.model, self.ik, self.jp, self.jr, self.hand); jac = np.vstack((self.jp[:, self.dadr], self.jr[:, self.dadr]))
            dq = jac.T @ np.linalg.solve(jac @ jac.T + .03**2 * np.eye(6), err); self.ik.qpos[self.qadr] += np.clip(dq, -.08, .08)
            limits = self.model.jnt_range[self.jids]; self.ik.qpos[self.qadr] = np.clip(self.ik.qpos[self.qadr], limits[:, 0], limits[:, 1]); mujoco.mj_forward(self.model, self.ik)
        lead = self.ik.qpos[self.qadr] - self.data.qpos[self.qadr]; lead *= min(1., .12 / max(float(np.max(np.abs(lead))), 1e-12))
        limits = self.model.actuator_ctrlrange[:7]; self.data.ctrl[:7] = np.clip(self.data.qpos[self.qadr] + lead, limits[:, 0], limits[:, 1]); self.data.ctrl[7] = np.clip(action[6], 0., 255.)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY); ap.add_argument("--scene", type=Path, default=DEFAULT_SCENE); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=("cuda", "cpu", "mps")); ap.add_argument("--episodes", type=int, default=1); ap.add_argument("--steps", type=int, default=1800); ap.add_argument("--seed", type=int, default=None, help="Random seed; omit for a new layout each run"); ap.add_argument("--headless", action="store_true"); args = ap.parse_args()
    path = args.policy_path.expanduser().resolve()
    for f in ("config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"):
        if not (path / f).is_file(): ap.error(f"missing {path / f}")
    policy = ACTPolicy.from_pretrained(path, strict=True).eval().to(args.device)
    pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=str(path), preprocessor_overrides={"device_processor": {"device": args.device}})
    env = Panda(args.scene, args.seed); renderer = mujoco.Renderer(env.model, height=480, width=640)
    viewer = None
    try:
        if not args.headless:
            viewer = mujoco.viewer.launch_passive(env.model, env.data); viewer.cam.lookat[:] = [.48, 0, .48]; viewer.cam.distance = 1.45; viewer.cam.azimuth = 145; viewer.cam.elevation = -15
        successes = 0
        for ep in range(args.episodes):
            env.reset(); policy.reset(); pre.reset(); post.reset()
            for step in range(args.steps):
                if viewer is not None and not viewer.is_running(): return
                t = time.monotonic()
                with torch.inference_mode(): action = post(policy.select_action(pre(env.observation(renderer))))
                env.apply_action(action.detach().cpu().numpy()); deadline = (step + 1) / 30.0
                while env.data.time < deadline: mujoco.mj_step(env.model, env.data)
                if viewer is not None: viewer.sync(); time.sleep(max(0, 1/30 - (time.monotonic() - t)))
                if step % 100 == 0: print(f"episode={ep+1} step={step} cube-target={np.linalg.norm(env.data.xpos[env.cube,:2]-env.data.xpos[env.plate,:2]):.3f}m", flush=True)
            xy_distance = float(np.linalg.norm(env.data.xpos[env.cube, :2] - env.data.xpos[env.plate, :2]))
            cube_z = float(env.data.xpos[env.cube, 2])
            success = xy_distance <= 0.05 and 0.42 <= cube_z <= 0.49
            successes += int(success)
            print(f"Episode {ep + 1}: {'SUCCESS' if success else 'FAIL'} "
                  f"(xy={xy_distance:.3f}m, cube_z={cube_z:.3f}m)", flush=True)
        print(f"Success rate: {successes}/{args.episodes} = {successes / args.episodes:.1%}", flush=True)
    finally:
        renderer.close()
        if viewer is not None: viewer.close()

if __name__ == "__main__": main()
