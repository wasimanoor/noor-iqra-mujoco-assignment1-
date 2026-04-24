# -*- coding: utf-8 -*-
"""
pickandplace.py — FULL UPDATED (Robust Re-Perception + Retry + No Flip + Voice waits for task)

Key fixes vs your last run:
✅ Re-assess object pose (and yaw) EVERY attempt
✅ Also re-assess DURING descend (tracks live XY so “pushed/slipped” doesn’t miss)
✅ After failed grasp: retreats to safe hover + waits a bit (prevents retrying same bad spot)
✅ Yaw alignment WITHOUT flipping (keeps stable “home/down” roll/pitch, applies only yaw delta)
✅ No NumPy ctrl DeprecationWarning (uses ctrl array by actuator id)
✅ Drop-during-carry detection (abort place/stack and optionally retry)
✅ Voice listener pauses while a motion task is running (no spam “couldn’t understand” during motion)

Requires:
- world.xml, panda.xml in same folder
- nl_interface.py in same folder (parse_command)
- Voice mode:
    py -m pip install SpeechRecognition PyAudio

Run:
  py pickandplace.py --viewer-console
  py pickandplace.py --terminal
  py pickandplace.py --voice-terminal
  py pickandplace.py --task pick_place_xy --obj red_box --x 0.55 --y -0.25 --no-gui
"""

from __future__ import annotations

import time
import threading
from threading import Thread
import tkinter as tk
from tkinter import ttk, messagebox

import glfw
import mujoco
import numpy as np

from nl_interface import parse_command

# ------------------ VOICE ------------------
import speech_recognition as sr
# ------------------------------------------


def _now() -> float:
    return time.time()


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class Demo:
    # home joint pose (teleport at init/reset only)
    qpos0 = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]

    # Cartesian impedance gains (pos xyz, rot xyz)
    K = np.array([900.0, 900.0, 900.0, 40.0, 40.0, 40.0])

    height, width = 480, 640
    fps = 30

    # ---------------- motion tuning ----------------
    ctrl_hz = 400  # target update rate
    hold_hz = 500  # mj_step rate in hold loop

    # adaptive speed (seconds): duration = max(min_dur, dist / max_speed)
    max_speed_xy = 0.30  # m/s
    max_speed_z = 0.25   # m/s
    min_move_dur = 0.25  # s
    max_move_dur = 2.50  # s

    # grasp approach heights (meters)
    hover_clear = 0.15
    pre_clear = 0.020
    near_clear = 0.010
    touch_clear = 0.002

    # grasp verification lift
    verify_lift = 0.06       # m
    verify_min_rise = 0.02   # m
    verify_max_dxy = 0.07    # m

    # retry
    max_grasp_attempts = 3

    # yaw align option
    align_yaw_to_object = True

    # live tracking during descend
    track_object_during_descend = True

    # after failed attempt: retreat and settle
    retreat_z_min = 0.22
    retreat_settle_s = 0.25

    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_path("world.xml")
        self.data = mujoco.MjData(self.model)

        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.cam.fixedcamid = 0
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)

        self.run = True
        self.stop_flag = threading.Event()

        # motion busy flag (used by voice listener + console)
        self.motion_busy = threading.Event()

        # ---- actuator id cache (fix NumPy ctrl warning) ----
        self._act_id_cache: dict[str, int] = {}

        # ---- init pose ----
        self.gripper(True)
        for i in range(1, 8):
            self.data.joint(f"panda_joint{i}").qpos = self.qpos0[i - 1]
        mujoco.mj_forward(self.model, self.data)

        hand = self.data.body("panda_hand")
        self.target_pos = hand.xpos.copy()
        self.target_quat = hand.xquat.copy()
        self._hold_running = True

        # store home pose (task space) for smooth return
        self.home_pos = hand.xpos.copy()
        self.home_quat = hand.xquat.copy()

        # bookkeeping held object name
        self.held_obj: str | None = None

        # ----- viewer console state -----
        self.viewer_console_enabled = False
        self.console_input = ""
        self.console_history: list[str] = []
        self.console_status = "Ready"
        self._console_lock = threading.Lock()
        self._console_busy = False  # block parallel motion tasks

        # after each motion task -> return to home (task-space)
        self.go_home_after_motion_task = True

        # voice thread guard
        self._voice_running = False

    # ---------------- actuator helpers (NO NumPy ctrl warning) ----------------
    def _act_id(self, name: str) -> int:
        if name in self._act_id_cache:
            return self._act_id_cache[name]
        aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            raise KeyError(f"Actuator '{name}' not found in model.")
        self._act_id_cache[name] = int(aid)
        return int(aid)

    def _set_act(self, name: str, value: float) -> None:
        self.data.ctrl[self._act_id(name)] = float(value)

    # ---------------- quaternion helpers ----------------
    @staticmethod
    def _quat_conj(q: np.ndarray) -> np.ndarray:
        return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)

    @staticmethod
    def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ], dtype=float)

    def _quat_rotate_vec(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        # v' = q * (0,v) * q_conj
        qv = np.array([0.0, v[0], v[1], v[2]], dtype=float)
        return self._quat_mul(self._quat_mul(q, qv), self._quat_conj(q))[1:]

    def _yaw_from_quat(self, q: np.ndarray) -> float:
        # rotate world X axis; yaw = atan2(y,x)
        vx = self._quat_rotate_vec(q, np.array([1.0, 0.0, 0.0], dtype=float))
        return float(np.arctan2(vx[1], vx[0]))

    @staticmethod
    def _wrap_pi(a: float) -> float:
        while a > np.pi:
            a -= 2*np.pi
        while a < -np.pi:
            a += 2*np.pi
        return float(a)

    @staticmethod
    def _quat_from_yaw(yaw: float) -> np.ndarray:
        half = 0.5 * yaw
        return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=float)

    # ---------------- basic helpers ----------------
    def _valid_object_names(self) -> list[str]:
        skip_keywords = ("panda", "floor", "world", "table", "ground", "base")
        out: list[str] = []
        for bid in range(self.model.nbody):
            nm = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if not nm:
                continue
            if any(k in nm.lower() for k in skip_keywords):
                continue
            # keep your convention: boxes only
            if "box" in nm.lower():
                out.append(nm)
        return sorted(set(out))

    def _require_body(self, name: str) -> str:
        try:
            _ = self.data.body(name)
            return name
        except KeyError:
            valid = self._valid_object_names()
            raise KeyError(f"Invalid name '{name}'. Valid names: {valid}")

    def _site_xy(self, site_name: str) -> np.ndarray:
        try:
            return self.data.site(site_name).xpos[:2].copy()
        except Exception:
            raise KeyError(f"Site '{site_name}' not found. Try: bin_center, zone_left, zone_right")

    def _body_xy(self, body_name: str) -> np.ndarray:
        return self.data.body(body_name).xpos[:2].copy()

    def _body_top_z(self, body_name: str) -> float:
        """Estimate top Z from body z + geom half-height for first geom."""
        b = self.data.body(body_name)
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)

        z_top = float(b.xpos[2])
        for g in range(self.model.ngeom):
            if self.model.geom_bodyid[g] != bid:
                continue
            size = np.array(self.model.geom_size[g])
            gtype = int(self.model.geom_type[g])
            if gtype == mujoco.mjtGeom.mjGEOM_BOX and size.size >= 3:
                return float(b.xpos[2] + size[2])
            if gtype == mujoco.mjtGeom.mjGEOM_CYLINDER and size.size >= 2:
                return float(b.xpos[2] + size[1])
        return z_top

    def _body_yaw(self, body_name: str) -> float:
        """Yaw about world Z from body's rotation matrix."""
        b = self.data.body(body_name)
        R = np.array(b.xmat).reshape(3, 3)
        return float(np.arctan2(R[1, 0], R[0, 0]))

    # ---------------- controller ----------------
    def gripper(self, open: bool = True) -> None:
        v = 0.04 if open else 0.0
        self._set_act("pos_panda_finger_joint1", v)
        self._set_act("pos_panda_finger_joint2", v)

    def control(self, xpos_d: np.ndarray, xquat_d: np.ndarray) -> None:
        xpos = self.data.body("panda_hand").xpos
        xquat = self.data.body("panda_hand").xquat

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        bodyid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "panda_hand")
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, bodyid)

        error = np.zeros(6)
        error[:3] = xpos_d - xpos

        res = np.zeros(3)
        mujoco.mju_subQuat(res, xquat, xquat_d)
        mujoco.mju_rotVecQuat(res, res, xquat)
        error[3:] = -res

        J = np.concatenate((jacp, jacr))
        v = J @ self.data.qvel

        Kp = np.diag(self.K)
        Kd = np.diag(2.0 * np.sqrt(self.K))

        # torque ctrl (actuators panda_joint1..7 must be torque-type)
        for i in range(1, 8):
            dofadr = self.model.joint(f"panda_joint{i}").dofadr
            self.data.actuator(f"panda_joint{i}").ctrl = self.data.joint(f"panda_joint{i}").qfrc_bias
            self.data.actuator(f"panda_joint{i}").ctrl += (J[:, dofadr].T @ Kp @ error)
            self.data.actuator(f"panda_joint{i}").ctrl -= (J[:, dofadr].T @ Kd @ v)

    # --------------- hold loop ----------------
    def _hold_loop(self) -> None:
        dt = 1.0 / float(self.hold_hz)
        while self.run and self._hold_running:
            self.control(self.target_pos, self.target_quat)
            mujoco.mj_step(self.model, self.data)
            time.sleep(dt)

    # ---------------- motion primitives ----------------
    @staticmethod
    def _quat_err(q: np.ndarray, r: np.ndarray) -> float:
        dot = abs(float(np.dot(q, r)))
        dot = max(min(dot, 1.0), -1.0)
        return 2.0 * np.arccos(dot)

    def _reach_pose(
        self,
        pos_goal: np.ndarray,
        quat_goal: np.ndarray,
        pos_tol: float = 0.004,
        ang_tol: float = 0.06,
        timeout: float = 2.2,
    ) -> bool:
        t0 = _now()
        self.target_pos = pos_goal.copy()
        self.target_quat = quat_goal.copy()
        dt = 1.0 / float(self.ctrl_hz)

        while _now() - t0 < timeout and not self.stop_flag.is_set():
            hand = self.data.body("panda_hand")
            p_err = float(np.linalg.norm(pos_goal - hand.xpos))
            a_err = float(self._quat_err(quat_goal, hand.xquat))
            if p_err < pos_tol and a_err < ang_tol:
                return True
            time.sleep(dt)
        return False

    def _adaptive_duration(self, start: np.ndarray, goal: np.ndarray) -> float:
        d = goal - start
        dxy = float(np.linalg.norm(d[:2]))
        dz = abs(float(d[2]))
        t_xy = dxy / max(self.max_speed_xy, 1e-6)
        t_z = dz / max(self.max_speed_z, 1e-6)
        t = max(t_xy, t_z, self.min_move_dur)
        return float(_clamp(t, self.min_move_dur, self.max_move_dur))

    def _move_linear(self, target_pos: np.ndarray, xquat_ref: np.ndarray, duration_s: float | None = None) -> None:
        start = self.data.body("panda_hand").xpos.copy()
        if duration_s is None:
            duration_s = self._adaptive_duration(start, target_pos)

        steps = max(1, int(duration_s * self.ctrl_hz))
        self.target_quat = xquat_ref
        dt = 1.0 / float(self.ctrl_hz)

        for k in range(steps):
            if self.stop_flag.is_set():
                return
            a = (k + 1) / steps
            self.target_pos = (1.0 - a) * start + a * target_pos
            time.sleep(dt)

    def _descend_with_xy_lock(
        self,
        xy_ref,  # np.array([x,y]) OR callable -> np.array([x,y])
        z_from: float,
        z_to: float,
        xquat_ref: np.ndarray,
        duration_s: float,
        xy_alpha: float = 0.45,
    ) -> None:
        steps = max(1, int(duration_s * self.ctrl_hz))
        dt = 1.0 / float(self.ctrl_hz)

        def _xy():
            return xy_ref() if callable(xy_ref) else xy_ref

        for k in range(steps):
            if self.stop_flag.is_set():
                return
            a = (k + 1) / steps
            z = (1.0 - a) * z_from + a * z_to

            xy = np.asarray(_xy(), dtype=float)

            cur = self.target_pos.copy()
            cur_xy = cur[:2]
            new_xy = (1.0 - xy_alpha) * cur_xy + xy_alpha * xy

            self.target_pos = np.array([new_xy[0], new_xy[1], z], dtype=float)
            self.target_quat = xquat_ref
            time.sleep(dt)

    def wait(self, seconds: float) -> None:
        t0 = _now()
        dt = 1.0 / float(self.ctrl_hz)
        while _now() - t0 < seconds:
            if self.stop_flag.is_set():
                return
            time.sleep(dt)

    # ---------------- grasp verification & tracking ----------------
    def _holding_object_now(self, obj_name: str) -> bool:
        hand = self.data.body("panda_hand")
        obj = self.data.body(obj_name)
        dxy = float(np.linalg.norm(obj.xpos[:2] - hand.xpos[:2]))
        return dxy < self.verify_max_dxy

    def _lift_verify_grasp(self, obj_name: str, xquat_ref: np.ndarray) -> bool:
        hand = self.data.body("panda_hand")
        obj = self.data.body(obj_name)

        obj_z0 = float(obj.xpos[2])

        lift = hand.xpos.copy()
        lift[2] += self.verify_lift
        self._move_linear(lift, xquat_ref, duration_s=0.9)
        self._reach_pose(lift, xquat_ref, pos_tol=0.012, ang_tol=0.12, timeout=2.0)

        obj_z1 = float(obj.xpos[2])
        dxy = float(np.linalg.norm(obj.xpos[:2] - hand.xpos[:2]))

        lifted_enough = (obj_z1 - obj_z0) > self.verify_min_rise
        near_hand = dxy < self.verify_max_dxy
        return bool(lifted_enough and near_hand)

    # ---------------- home / reset ----------------
    def reset_home(self) -> None:
        self.stop_flag.clear()
        self.held_obj = None

        self.gripper(True)
        for i in range(1, 8):
            self.data.joint(f"panda_joint{i}").qpos = self.qpos0[i - 1]
        mujoco.mj_forward(self.model, self.data)

        hand = self.data.body("panda_hand")
        self.target_pos = hand.xpos.copy()
        self.target_quat = hand.xquat.copy()
        self.home_pos = hand.xpos.copy()
        self.home_quat = hand.xquat.copy()

    def return_home_smooth(self, duration_s: float = 1.8) -> None:
        self.stop_flag.clear()
        hand = self.data.body("panda_hand")
        xquat_ref = self.home_quat.copy()
        start = hand.xpos.copy()

        steps = max(1, int(duration_s * self.ctrl_hz))
        dt = 1.0 / float(self.ctrl_hz)

        for k in range(steps):
            if self.stop_flag.is_set():
                return
            a = (k + 1) / steps
            self.target_pos = (1.0 - a) * start + a * self.home_pos
            self.target_quat = xquat_ref
            time.sleep(dt)

        self._reach_pose(self.home_pos, self.home_quat, pos_tol=0.006, ang_tol=0.08, timeout=2.2)

    def _retreat_safe(self) -> None:
        """After a failed attempt, lift to safe Z and let physics settle."""
        hand = self.data.body("panda_hand")
        pos = hand.xpos.copy()
        pos[2] = max(float(pos[2]), float(self.retreat_z_min))
        self._move_linear(pos, self.home_quat, duration_s=0.6)
        self._reach_pose(pos, self.home_quat, pos_tol=0.02, ang_tol=0.25, timeout=1.6)
        self.wait(self.retreat_settle_s)

    # ---------------- robust pick ----------------
    def _make_pick_orientation(self, obj_name: str) -> np.ndarray:
        """
        Stable orientation:
        base = home_quat (your stable down orientation),
        then optionally apply ONLY yaw delta around world Z to align to object yaw.
        This avoids the flip you saw.
        """
        base_q = self.home_quat.copy()
        if not self.align_yaw_to_object:
            return base_q

        try:
            obj_yaw = float(self._body_yaw(obj_name))
            base_yaw = float(self._yaw_from_quat(base_q))
            dyaw = float(self._wrap_pi(obj_yaw - base_yaw))
            qz = self._quat_from_yaw(dyaw)
            return self._quat_mul(qz, base_q)  # yaw_delta * base
        except Exception:
            return base_q

    def pick_only_once(self, obj_name: str) -> bool:
        """
        One pick attempt:
        - re-read LIVE object pose (works even if pushed/slipped)
        - stable no-flip orientation
        - optional yaw align (yaw only)
        - descend tracks LIVE object XY (best for “it moved while descending”)
        - close
        - lift-verify
        """
        self.stop_flag.clear()
        obj = self._require_body(obj_name)

        # pose read (attempt start)
        xy0 = self._body_xy(obj)
        z_top0 = self._body_top_z(obj)
        xquat_ref = self._make_pick_orientation(obj)

        hover = np.array([xy0[0], xy0[1], z_top0 + self.hover_clear], dtype=float)
        pregrasp = np.array([xy0[0], xy0[1], z_top0 + self.pre_clear], dtype=float)

        self.gripper(True)
        self.wait(0.10)

        # approach hover/pregrasp (re-read again right before pregrasp to avoid stale approach)
        self._move_linear(hover, xquat_ref)
        self._reach_pose(hover, xquat_ref, 0.010, 0.12, 2.4)

        # refresh pose before getting close (box may have moved already)
        xy1 = self._body_xy(obj)
        z_top1 = self._body_top_z(obj)
        pregrasp = np.array([xy1[0], xy1[1], z_top1 + self.pre_clear], dtype=float)

        self._move_linear(pregrasp, xquat_ref)
        self._reach_pose(pregrasp, xquat_ref, 0.010, 0.12, 2.4)

        # build near/touch using latest pose
        xy2 = self._body_xy(obj)
        z_top2 = self._body_top_z(obj)
        near = np.array([xy2[0], xy2[1], z_top2 + self.near_clear], dtype=float)
        touch = np.array([xy2[0], xy2[1], z_top2 + self.touch_clear], dtype=float)

        # if enabled, track object during descend
        if self.track_object_during_descend:
            xy_live = lambda: self._body_xy(obj)
        else:
            xy_live = xy2

        # descend to near
        self._descend_with_xy_lock(
            xy_live, z_from=pregrasp[2], z_to=near[2],
            xquat_ref=xquat_ref, duration_s=0.8, xy_alpha=0.45
        )
        self._reach_pose(near, xquat_ref, 0.012, 0.14, 2.2)

        # refresh z_top before final touch (object may have tilted/changed)
        z_top3 = self._body_top_z(obj)
        touch = np.array([self._body_xy(obj)[0], self._body_xy(obj)[1], z_top3 + self.touch_clear], dtype=float)

        # slow final descend
        self._descend_with_xy_lock(
            xy_live, z_from=float(self.target_pos[2]), z_to=touch[2],
            xquat_ref=xquat_ref, duration_s=1.2, xy_alpha=0.60
        )
        self._reach_pose(touch, xquat_ref, 0.014, 0.16, 2.2)

        self.wait(0.10)

        # close
        self.gripper(False)
        self.wait(0.28)

        ok = self._lift_verify_grasp(obj, xquat_ref)

        if not ok:
            # failed: reopen and back off
            self.gripper(True)
            self.wait(0.18)
            self._retreat_safe()
            return False

        # success
        self.held_obj = obj

        # go back to safe carry hover using current object XY (not stale)
        xy_c = self._body_xy(obj)
        z_top_c = self._body_top_z(obj)
        carry = np.array([xy_c[0], xy_c[1], z_top_c + self.hover_clear], dtype=float)
        self._move_linear(carry, xquat_ref, duration_s=1.0)
        self._reach_pose(carry, xquat_ref, 0.014, 0.16, 2.0)
        return True

    def pick_only(self, target_hint: str = "box", attempts: int | None = None) -> bool:
        """
        Robust pick with retries.
        IMPORTANT: re-reads object pose each attempt, and retreats between attempts.
        """
        obj = self._require_body(target_hint)
        if attempts is None:
            attempts = int(self.max_grasp_attempts)

        for k in range(attempts):
            if self.stop_flag.is_set():
                return False

            # show status
            with self._console_lock:
                self.console_status = f"Picking {obj} (attempt {k+1}/{attempts})"

            ok = self.pick_only_once(obj)
            if ok:
                with self._console_lock:
                    self.console_status = f"✅ Grasped {obj}"
                return True

            # IMPORTANT: wait & re-settle BEFORE next attempt (object may still be moving)
            self.wait(0.15)

        with self._console_lock:
            self.console_status = f"❌ Grasp failed after {attempts} attempts: {obj}"
        return False

    # ---------------- place / stack ----------------
    def place_xy(self, x: float, y: float) -> bool:
        """
        Place at XY on table with careful descent.
        Returns False if it detects drop during carry.
        """
        self.stop_flag.clear()
        hand = self.data.body("panda_hand")
        xquat_ref = hand.xquat.copy()

        z_place = 0.03
        z_hover = z_place + 0.15

        hover = np.array([x, y, z_hover], dtype=float)
        place = np.array([x, y, z_place], dtype=float)

        # travel
        self._move_linear(hover, xquat_ref)
        self._reach_pose(hover, xquat_ref, 0.010, 0.12, 2.4)

        # drop check
        if self.held_obj is not None and not self._holding_object_now(self.held_obj):
            with self._console_lock:
                self.console_status = f"⚠️ Dropped {self.held_obj} during carry (abort place)"
            self.held_obj = None
            return False

        # descend
        self._descend_with_xy_lock(np.array([x, y]), z_from=hover[2], z_to=place[2], xquat_ref=xquat_ref, duration_s=1.1, xy_alpha=0.55)
        self._reach_pose(place, xquat_ref, 0.012, 0.14, 2.2)

        # open
        self.wait(0.10)
        self.gripper(True)
        self.wait(0.22)
        self.held_obj = None

        # back up
        self._move_linear(hover, xquat_ref, duration_s=1.0)
        self._reach_pose(hover, xquat_ref, 0.012, 0.14, 2.0)
        return True

    def place_on_top_of_body(self, base_name: str) -> bool:
        """Place on top of base (stack). Returns False if drop detected."""
        self.stop_flag.clear()
        base = self._require_body(base_name)

        base_xy = self._body_xy(base)
        base_top = self._body_top_z(base)

        hand = self.data.body("panda_hand")
        xquat_ref = hand.xquat.copy()

        z_place = base_top + 0.03
        z_hover = z_place + 0.15

        hover = np.array([base_xy[0], base_xy[1], z_hover], dtype=float)
        place = np.array([base_xy[0], base_xy[1], z_place], dtype=float)

        self._move_linear(hover, xquat_ref)
        self._reach_pose(hover, xquat_ref, 0.012, 0.14, 2.4)

        if self.held_obj is not None and not self._holding_object_now(self.held_obj):
            with self._console_lock:
                self.console_status = f"⚠️ Dropped {self.held_obj} during carry (abort stack)"
            self.held_obj = None
            return False

        # live-follow base XY during descent
        duration = 1.2
        steps = max(1, int(duration * self.ctrl_hz))
        dt = 1.0 / float(self.ctrl_hz)
        z_from = hover[2]
        z_to = place[2]
        for k in range(steps):
            if self.stop_flag.is_set():
                return False
            a = (k + 1) / steps
            z = (1 - a) * z_from + a * z_to
            base_xy_live = self._body_xy(base)
            self.target_pos = np.array([base_xy_live[0], base_xy_live[1], z], dtype=float)
            self.target_quat = xquat_ref
            time.sleep(dt)

        self._reach_pose(place, xquat_ref, 0.016, 0.18, 2.2)

        self.wait(0.12)
        self.gripper(True)
        self.wait(0.22)
        self.held_obj = None

        self._move_linear(hover, xquat_ref, duration_s=1.0)
        self._reach_pose(hover, xquat_ref, 0.014, 0.16, 2.0)
        return True

    # ---------------- placement target selection ----------------
    def _plan_free_xy_near_site(
        self,
        site_name: str,
        grid_dx: float = 0.085,
        grid_dy: float = 0.085,
        grid_n: int = 3,
        clearance: float = 0.075,
    ) -> tuple[float, float]:
        base = self._site_xy(site_name)
        candidates: list[tuple[float, float]] = []

        for iy in range(-(grid_n // 2), grid_n // 2 + 1):
            for ix in range(-(grid_n // 2), grid_n // 2 + 1):
                x = float(base[0] + ix * grid_dx)
                y = float(base[1] + iy * grid_dy)
                candidates.append((x, y))

        candidates.sort(key=lambda p: (p[0] - base[0]) ** 2 + (p[1] - base[1]) ** 2)

        objs = self._valid_object_names()
        for (cx, cy) in candidates:
            free = True
            for nm in objs:
                b = self.data.body(nm)
                ox, oy = float(b.xpos[0]), float(b.xpos[1])
                if (cx - ox) ** 2 + (cy - oy) ** 2 < (clearance ** 2):
                    free = False
                    break
            if free:
                return (cx, cy)

        return (float(base[0]), float(base[1]))

    # ---------------- composite tasks ----------------
    def pick_place_to_site(self, obj: str, target_site: str) -> None:
        if not self.pick_only(target_hint=obj):
            return

        x, y = self._plan_free_xy_near_site(target_site)
        ok_place = self.place_xy(x, y)

        # if it dropped during carry, optionally retry
        if not ok_place:
            with self._console_lock:
                self.console_status = "Retrying after drop..."
            self.wait(0.3)
            if self.pick_only(target_hint=obj, attempts=2):
                self.place_xy(x, y)

    def pick_place_xy(self, obj: str, x: float, y: float) -> None:
        if not self.pick_only(target_hint=obj):
            return
        ok_place = self.place_xy(float(x), float(y))
        if not ok_place:
            with self._console_lock:
                self.console_status = "Retrying after drop..."
            self.wait(0.3)
            if self.pick_only(target_hint=obj, attempts=2):
                self.place_xy(float(x), float(y))

    def stack(self, obj: str, base: str) -> None:
        if not self.pick_only(target_hint=obj):
            return
        ok_place = self.place_on_top_of_body(base)
        if not ok_place:
            with self._console_lock:
                self.console_status = "Retrying after drop..."
            self.wait(0.3)
            if self.pick_only(target_hint=obj, attempts=2):
                self.place_on_top_of_body(base)

    def sort_all(self) -> None:
        mapping = {
            "red_box": "zone_left",
            "green_box": "zone_right",
            "blue_box": "bin_center",
            "yellow_box": "bin_center",
            "box": "bin_center",
        }
        objs = self._valid_object_names()
        for obj in objs:
            if self.stop_flag.is_set():
                return
            target = mapping.get(obj, "bin_center")
            self.pick_place_to_site(obj, target)

    def tower(self) -> None:
        objs = self._valid_object_names()
        wanted = ["blue_box", "green_box", "red_box"]
        present = [o for o in wanted if o in objs]
        if len(present) < 2:
            raise RuntimeError(f"Need at least 2 boxes for tower. Found: {objs}")

        base = present[0]
        for obj in present[1:]:
            if self.stop_flag.is_set():
                return
            self.stack(obj, base)
            base = obj

    # ---------------- perception ----------------
    def list_objects(self) -> str:
        objs = self._valid_object_names()
        lines = ["Objects:"]
        hand = self.data.body("panda_hand")
        hx, hy, hz = hand.xpos
        for nm in objs:
            b = self.data.body(nm)
            x, y, z = b.xpos
            d = float(np.linalg.norm(np.array([x - hx, y - hy, z - hz])))
            lines.append(f"  - {nm:10s} pos=({x:+.3f},{y:+.3f},{z:+.3f})  dist={d:.3f}")
        return "\n".join(lines)

    def where_is(self, obj: str) -> str:
        nm = self._require_body(obj)
        b = self.data.body(nm)
        x, y, z = b.xpos
        return f"{nm} at ({x:+.3f},{y:+.3f},{z:+.3f})"

    # ---------------- Terminal NL control ----------------
    def start_terminal_control(self) -> None:
        def loop():
            print("\n--- Natural Language Control (Terminal) ---")
            print("Examples:")
            print("  list objects")
            print("  where is red_box")
            print("  pick red_box place bin")
            print("  pick green_box place left")
            print("  stack red_box on green_box")
            print("  sort all")
            print("  make a tower")
            print("  open gripper | close gripper")
            print("  reset | quit\n")

            while self.run:
                try:
                    text = input("NL> ").strip()
                except (EOFError, KeyboardInterrupt):
                    text = "quit"
                cmd = parse_command(text)
                self._execute_parsed_command(cmd, raw=text)

        Thread(target=loop, daemon=True).start()

    # ---------------- Voice NL control ----------------
    def start_voice_terminal_control(self) -> None:
        def loop():
            if self._voice_running:
                return
            self._voice_running = True

            r = sr.Recognizer()
            r.dynamic_energy_threshold = True
            r.pause_threshold = 0.6
            r.phrase_threshold = 0.25
            r.non_speaking_duration = 0.3

            print("\n--- Natural Language Control (Voice) ---")
            print("Speak commands like:")
            print("  list objects")
            print("  where is red box")
            print("  pick red box place bin")
            print("  stack red on green")
            print("  sort all")
            print("  make a tower")
            print("  open gripper | close gripper")
            print("  reset | quit\n")
            print("NOTE: Voice will PAUSE while robot is moving.\n")

            try:
                mic = sr.Microphone()
            except Exception as e:
                print("VOICE> Could not open microphone:", e)
                self._voice_running = False
                return

            # calibrate once
            try:
                with mic as source:
                    r.adjust_for_ambient_noise(source, duration=0.8)
            except Exception as e:
                print("VOICE> Ambient noise calibration failed:", e)

            while self.run:
                # ✅ IMPORTANT: do not listen while motion task is running
                if self.motion_busy.is_set():
                    time.sleep(0.12)
                    continue

                try:
                    print("VOICE> listening...")
                    with mic as source:
                        audio = r.listen(source, timeout=6, phrase_time_limit=5.5)

                    text = r.recognize_google(audio).strip()
                    if not text:
                        continue
                    text_l = text.lower()
                    print("VOICE>", text_l)

                    cmd = parse_command(text_l)
                    self._execute_parsed_command(cmd, raw=text_l)

                except sr.WaitTimeoutError:
                    continue
                except sr.UnknownValueError:
                    print("VOICE> (couldn't understand)")
                except sr.RequestError as e:
                    print("VOICE> Speech service error:", e)
                except Exception as e:
                    print("VOICE> Error:", e)

            self._voice_running = False

        Thread(target=loop, daemon=True).start()

    # ---------------- command execution ----------------
    def _execute_parsed_command(self, cmd: dict, raw: str | None = None) -> None:
        if raw:
            with self._console_lock:
                self.console_history.append(raw)
                self.console_history = self.console_history[-10:]

        task = cmd.get("task", "unknown")

        with self._console_lock:
            if self._console_busy and task in ("pick_place", "pick_place_xy", "stack", "place_on", "sort_all", "tower"):
                self.console_status = "Busy: wait for current motion to finish"
                return

        def run():
            try:
                with self._console_lock:
                    self.console_status = f"Parsed: {task}"

                if task == "help":
                    with self._console_lock:
                        self.console_status = (
                            "Help: list objects | where is red_box | pick red_box place bin/left/right/x..y.. | "
                            "stack red_box on green_box | sort all | make a tower | reset | quit"
                        )
                    return

                if task == "unknown":
                    with self._console_lock:
                        self.console_status = "❌ Unknown command"
                    return

                if task == "quit":
                    with self._console_lock:
                        self.console_status = "Stopping..."
                    self.run = False
                    self._hold_running = False
                    self.stop_flag.set()
                    return

                if task == "reset":
                    with self._console_lock:
                        self.console_status = "Resetting..."
                    self.reset_home()
                    with self._console_lock:
                        self.console_status = "✅ reset done"
                    return

                if task == "gripper":
                    mode = cmd.get("mode", "open")
                    self.gripper(open=(mode == "open"))
                    with self._console_lock:
                        self.console_status = f"✅ gripper {mode}"
                    return

                if task == "list_objects":
                    s = self.list_objects()
                    with self._console_lock:
                        self.console_status = "✅ Listed objects (see terminal)"
                    print(s)
                    return

                if task == "where":
                    s = self.where_is(cmd.get("obj", "box"))
                    with self._console_lock:
                        self.console_status = s
                    print(s)
                    return

                # ----- motion tasks -----
                with self._console_lock:
                    self._console_busy = True
                self.motion_busy.set()

                if task == "pick_place":
                    obj = cmd.get("obj", "box")
                    target = cmd.get("target", "bin_center")
                    with self._console_lock:
                        self.console_status = f"Pick {obj} -> {target}"
                    self.pick_place_to_site(obj, target)

                elif task == "pick_place_xy":
                    obj = cmd.get("obj", "box")
                    x = float(cmd["x"])
                    y = float(cmd["y"])
                    with self._console_lock:
                        self.console_status = f"Pick {obj} -> ({x:.2f},{y:.2f})"
                    self.pick_place_xy(obj, x, y)

                elif task == "stack":
                    obj = cmd.get("obj", "box")
                    base = cmd.get("base", "box")
                    with self._console_lock:
                        self.console_status = f"Stack {obj} on {base}"
                    self.stack(obj, base)

                elif task == "place_on":
                    obj = cmd.get("obj", "box")
                    base = cmd.get("base", "box")
                    with self._console_lock:
                        self.console_status = f"Place {obj} on {base}"
                    self.stack(obj, base)

                elif task == "sort_all":
                    with self._console_lock:
                        self.console_status = "Sorting all..."
                    self.sort_all()

                elif task == "tower":
                    with self._console_lock:
                        self.console_status = "Building tower..."
                    self.tower()

                else:
                    with self._console_lock:
                        self.console_status = f"Unhandled task: {task}"

                if self.go_home_after_motion_task and task in ("pick_place", "pick_place_xy", "stack", "place_on", "sort_all", "tower"):
                    with self._console_lock:
                        self.console_status = "Returning home..."
                    self.return_home_smooth()

                with self._console_lock:
                    self._console_busy = False
                    self.console_status = "✅ Done"
                self.motion_busy.clear()

            except Exception as e:
                self.motion_busy.clear()
                with self._console_lock:
                    self._console_busy = False
                    self.console_status = f"Error: {e}"

        Thread(target=run, daemon=True).start()

    # ---------------- viewer ----------------
    def render(self) -> None:
        glfw.init()
        glfw.window_hint(glfw.SAMPLES, 8)
        window = glfw.create_window(self.width, self.height, "Panda Demo", None, None)
        glfw.make_context_current(window)

        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_100)
        opt = mujoco.MjvOption()
        pert = mujoco.MjvPerturb()
        viewport = mujoco.MjrRect(0, 0, self.width, self.height)

        def _clipboard_get_str() -> str:
            clip = glfw.get_clipboard_string(window)
            if clip is None:
                return ""
            if isinstance(clip, (bytes, bytearray)):
                try:
                    return clip.decode("utf-8", errors="ignore")
                except Exception:
                    return ""
            return str(clip)

        def _clipboard_set_str(s: str) -> None:
            try:
                glfw.set_clipboard_string(window, s)
            except Exception:
                pass

        def on_char(win, codepoint):
            if not self.viewer_console_enabled:
                return
            ch = chr(codepoint)
            if ch.isprintable():
                with self._console_lock:
                    self.console_input += ch

        def on_key(win, key, scancode, action, mods):
            if not self.viewer_console_enabled:
                return
            if action not in (glfw.PRESS, glfw.REPEAT):
                return

            ctrl = (mods & glfw.MOD_CONTROL) != 0

            if ctrl and key == glfw.KEY_V:
                clip = _clipboard_get_str()
                if clip:
                    with self._console_lock:
                        self.console_input += clip
                return

            if ctrl and key == glfw.KEY_C:
                with self._console_lock:
                    s = self.console_input
                _clipboard_set_str(s)
                with self._console_lock:
                    self.console_status = "Copied input to clipboard"
                return

            if key == glfw.KEY_BACKSPACE:
                with self._console_lock:
                    self.console_input = self.console_input[:-1]
            elif key == glfw.KEY_ESCAPE:
                with self._console_lock:
                    self.console_input = ""
                    self.console_status = "Cleared"
            elif key == glfw.KEY_ENTER or key == glfw.KEY_KP_ENTER:
                with self._console_lock:
                    text = self.console_input.strip()
                    self.console_input = ""
                if text:
                    cmd = parse_command(text)
                    self._execute_parsed_command(cmd, raw=text)

        glfw.set_char_callback(window, on_char)
        glfw.set_key_callback(window, on_key)

        while not glfw.window_should_close(window):
            w, h = glfw.get_framebuffer_size(window)
            viewport.width, viewport.height = w, h

            mujoco.mjv_updateScene(
                self.model,
                self.data,
                opt,
                pert,
                self.cam,
                mujoco.mjtCatBit.mjCAT_ALL,
                self.scene,
            )
            mujoco.mjr_render(viewport, self.scene, self.context)

            if self.viewer_console_enabled:
                with self._console_lock:
                    lines = []
                    lines.append("Console (Viewer): Enter=run | Esc=clear | Backspace=del | Ctrl+V Paste | Ctrl+C Copy")
                    lines.append(f"Status: {self.console_status}")
                    lines.append("History:")
                    for hline in self.console_history[-6:]:
                        lines.append(f"  {hline}")
                    lines.append("")
                    lines.append(f"> {self.console_input}_")
                    overlay_text = "\n".join(lines)

                mujoco.mjr_overlay(
                    mujoco.mjtFontScale.mjFONTSCALE_100,
                    mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    viewport,
                    overlay_text,
                    "",
                    self.context,
                )

            time.sleep(1.0 / self.fps)
            glfw.swap_buffers(window)
            glfw.poll_events()

        self.run = False
        self._hold_running = False
        self.stop_flag.set()
        glfw.terminate()

    def start(self) -> None:
        Thread(target=self._hold_loop, daemon=True).start()
        self.render()


# ---------------- GUI ----------------
def launch_gui(demo: Demo) -> None:
    root = tk.Tk()
    root.title("Panda: Pick & Place")
    root.geometry("590x240")

    target_name = tk.StringVar(value="box")
    ttk.Label(root, text="Target body:").grid(row=0, column=0, padx=6, sticky="e")
    ttk.Entry(root, width=18, textvariable=target_name).grid(row=0, column=1, padx=4, sticky="w")

    place_x = tk.StringVar(value="0.55")
    place_y = tk.StringVar(value="-0.25")
    ttk.Label(root, text="Place X:").grid(row=1, column=0, sticky="e", padx=6)
    ttk.Entry(root, width=10, textvariable=place_x).grid(row=1, column=1, padx=4, sticky="w")
    ttk.Label(root, text="Place Y:").grid(row=1, column=2, sticky="e", padx=6)
    ttk.Entry(root, width=10, textvariable=place_y).grid(row=1, column=3, padx=4, sticky="w")

    attempts_var = tk.StringVar(value=str(demo.max_grasp_attempts))
    ttk.Label(root, text="Grasp attempts:").grid(row=2, column=0, sticky="e", padx=6)
    ttk.Entry(root, width=10, textvariable=attempts_var).grid(row=2, column=1, padx=4, sticky="w")

    yaw_align = tk.BooleanVar(value=demo.align_yaw_to_object)
    ttk.Checkbutton(root, text="Align gripper yaw to object", variable=yaw_align).grid(row=3, column=0, columnspan=2, padx=8, sticky="w")

    track_desc = tk.BooleanVar(value=demo.track_object_during_descend)
    ttk.Checkbutton(root, text="Track object during descend (recommended)", variable=track_desc).grid(row=4, column=0, columnspan=3, padx=8, sticky="w")

    def run_pickplace_now():
        try:
            demo.max_grasp_attempts = int(attempts_var.get())
            demo.align_yaw_to_object = bool(yaw_align.get())
            demo.track_object_during_descend = bool(track_desc.get())

            obj = target_name.get().strip()
            x = float(place_x.get())
            y = float(place_y.get())

            demo.pick_place_xy(obj, x, y)
            demo.return_home_smooth()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    ttk.Button(root, text="Run Pick & Place XY", command=run_pickplace_now).grid(row=5, column=0, padx=8, pady=12, sticky="w")
    ttk.Label(root, text="Viewer console NL: run with --viewer-console").grid(row=6, column=0, columnspan=4, padx=8, sticky="w")
    ttk.Label(root, text="Voice NL: run with --voice-terminal").grid(row=7, column=0, columnspan=4, padx=8, sticky="w")

    root.mainloop()


# ---------------- main ----------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["pick_place", "pick_place_xy"], default=None)
    parser.add_argument("--obj", type=str, default="box")
    parser.add_argument("--target", type=str, default="bin_center")
    parser.add_argument("--x", type=float, default=None)
    parser.add_argument("--y", type=float, default=None)

    parser.add_argument("--gripper", choices=["open", "close"], default=None)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--no-gui", action="store_true")

    parser.add_argument("--terminal", action="store_true")
    parser.add_argument("--viewer-console", action="store_true")
    parser.add_argument("--voice-terminal", action="store_true",
                        help="Use microphone voice commands (like --terminal, but spoken).")

    # optional tuning
    parser.add_argument("--attempts", type=int, default=None, help="Max grasp attempts (default 3).")
    parser.add_argument("--no-yaw-align", action="store_true", help="Disable yaw alignment to object.")
    parser.add_argument("--no-track-descend", action="store_true", help="Disable tracking object during descend.")

    args = parser.parse_args()

    demo = Demo()
    demo.viewer_console_enabled = bool(args.viewer_console)

    if args.attempts is not None:
        demo.max_grasp_attempts = int(args.attempts)
    if args.no_yaw_align:
        demo.align_yaw_to_object = False
    if args.no_track_descend:
        demo.track_object_during_descend = False

    if args.viewer_console:
        demo.start()
        raise SystemExit(0)

    if args.terminal:
        demo.start_terminal_control()
        demo.start()
        raise SystemExit(0)

    if args.voice_terminal:
        demo.start_voice_terminal_control()
        demo.start()
        raise SystemExit(0)

    cli_mode = (args.task is not None) or (args.gripper is not None) or args.reset or args.no_gui
    if cli_mode:

        def run_sequence():
            if args.reset:
                demo.reset_home()
                time.sleep(0.1)

            if args.gripper is not None:
                demo.gripper(open=(args.gripper == "open"))
                time.sleep(0.1)

            if args.task == "pick_place":
                demo.pick_place_to_site(args.obj, args.target)
                demo.return_home_smooth()

            elif args.task == "pick_place_xy":
                if args.x is None or args.y is None:
                    raise ValueError("pick_place_xy requires --x and --y")
                demo.pick_place_xy(args.obj, float(args.x), float(args.y))
                demo.return_home_smooth()

        Thread(target=run_sequence, daemon=True).start()
        demo.start()

    else:
        Thread(target=launch_gui, args=(demo,), daemon=True).start()
        demo.start()
