"""Desktop AI weapon detection console.

This is the primary application entry point. It wraps a trained Ultralytics
YOLO model with a reviewer-friendly Tkinter interface for live webcam and
sample-image inference.
"""

from __future__ import annotations

import argparse
import csv
import queue
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox, ttk
from ultralytics import YOLO


APP_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = APP_DIR / "best.pt"
DEFAULT_SAMPLE_IMAGE = APP_DIR / "gun.png"


@dataclass(frozen=True)
class Detection:
    label: str
    category: str
    confidence: float
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class FrameResult:
    frame: object
    detections: list[Detection]
    inference_ms: float
    fps: float
    source_name: str
    timestamp: datetime


class WeaponDetectionApp:
    """Tkinter UI for live and still-image YOLO inference."""

    CATEGORY_KEYWORDS = {
        "handgun": ("handgun", "pistol", "revolver", "gun", "firearm"),
        "rifle": ("rifle", "shotgun", "sniper", "ak47", "m16", "assault"),
        "knife": ("knife", "blade", "dagger", "switchblade", "sword"),
        "explosive": ("grenade", "bomb", "explosive", "mine", "c4"),
        "other": ("weapon",),
    }

    CATEGORY_LABELS = {
        "handgun": "Handguns",
        "rifle": "Rifles / Shotguns",
        "knife": "Knives / Blades",
        "explosive": "Explosives",
        "other": "Other Weapons",
    }

    CATEGORY_COLORS = {
        "handgun": (58, 185, 118),
        "rifle": (255, 169, 77),
        "knife": (221, 91, 132),
        "explosive": (239, 83, 80),
        "other": (156, 163, 175),
    }

    UI_COLORS = {
        "bg": "#101418",
        "panel": "#171d24",
        "panel_alt": "#1f2730",
        "line": "#2e3a46",
        "text": "#f4f7fb",
        "muted": "#9aa7b4",
        "accent": "#55c2ff",
        "success": "#3ab976",
        "warning": "#ef5350",
        "amber": "#ffa94d",
    }

    def __init__(self, root: tk.Tk, model_path: Path, camera_index: int = 0) -> None:
        self.root = root
        self.model_path = model_path
        self.camera_index = tk.IntVar(value=camera_index)
        self.confidence_threshold = tk.DoubleVar(value=0.35)
        self.status_text = tk.StringVar(value="Ready. Load the webcam or analyze a sample image.")

        self.model = YOLO(str(model_path))
        self.model_names = self._extract_model_names()

        self.cap: cv2.VideoCapture | None = None
        self.running = False
        self.worker: threading.Thread | None = None
        self.result_queue: queue.Queue[FrameResult] = queue.Queue(maxsize=2)
        self.last_frame_bgr = None
        self.session_events: list[dict[str, str]] = []
        self.recent_fps: deque[float] = deque(maxlen=20)
        self.session_started_at: datetime | None = None

        self.root.title("AI Weapon Detection Console")
        self.root.geometry("1440x860")
        self.root.minsize(1120, 720)
        self.root.configure(bg=self.UI_COLORS["bg"])
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self._configure_styles()
        self._build_layout()
        self._show_placeholder()
        self._poll_results()

    def _extract_model_names(self) -> dict[int, str]:
        names = getattr(self.model, "names", {})
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        return {index: str(name) for index, name in enumerate(names)}

    def _configure_styles(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Console.Horizontal.TScale",
            background=self.UI_COLORS["panel"],
            troughcolor=self.UI_COLORS["panel_alt"],
        )
        style.configure(
            "Console.TSpinbox",
            fieldbackground=self.UI_COLORS["panel_alt"],
            background=self.UI_COLORS["panel_alt"],
            foreground=self.UI_COLORS["text"],
            arrowcolor=self.UI_COLORS["accent"],
            bordercolor=self.UI_COLORS["line"],
        )

    def _build_layout(self) -> None:
        main = tk.Frame(self.root, bg=self.UI_COLORS["bg"])
        main.pack(fill=tk.BOTH, expand=True)

        sidebar = tk.Frame(main, bg=self.UI_COLORS["panel"], width=390)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)

        self.video_area = tk.Frame(main, bg=self.UI_COLORS["bg"])
        self.video_area.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._build_sidebar(sidebar)
        self._build_video_area()

    def _build_sidebar(self, parent: tk.Frame) -> None:
        body = tk.Frame(parent, bg=self.UI_COLORS["panel"])
        body.pack(fill=tk.BOTH, expand=True, padx=22, pady=22)

        self._label(
            body,
            "AI Weapon Detection Console",
            size=20,
            weight="bold",
            color=self.UI_COLORS["accent"],
        ).pack(anchor=tk.W)
        self._label(
            body,
            "Real-time YOLO inference with confidence gating, class breakdown, and exportable evidence logs.",
            size=10,
            color=self.UI_COLORS["muted"],
            wraplength=330,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(6, 18))

        model_text = f"Model: {self.model_path.name}\nClasses: {', '.join(self.model_names.values()) or 'Unavailable'}"
        self._info_panel(body, "Model Readiness", model_text).pack(fill=tk.X, pady=(0, 14))

        kpi_grid = tk.Frame(body, bg=self.UI_COLORS["panel"])
        kpi_grid.pack(fill=tk.X, pady=(0, 14))
        kpi_grid.columnconfigure((0, 1), weight=1, uniform="kpi")

        self.on_screen_var = tk.StringVar(value="0")
        self.total_var = tk.StringVar(value="0")
        self.fps_var = tk.StringVar(value="0.0")
        self.inference_var = tk.StringVar(value="0 ms")

        self._kpi(kpi_grid, "On Screen", self.on_screen_var, self.UI_COLORS["success"]).grid(
            row=0, column=0, sticky="nsew", padx=(0, 6), pady=(0, 8)
        )
        self._kpi(kpi_grid, "Session Hits", self.total_var, self.UI_COLORS["accent"]).grid(
            row=0, column=1, sticky="nsew", padx=(6, 0), pady=(0, 8)
        )
        self._kpi(kpi_grid, "FPS", self.fps_var, self.UI_COLORS["amber"]).grid(
            row=1, column=0, sticky="nsew", padx=(0, 6)
        )
        self._kpi(kpi_grid, "Latency", self.inference_var, self.UI_COLORS["muted"]).grid(
            row=1, column=1, sticky="nsew", padx=(6, 0)
        )

        controls = tk.Frame(body, bg=self.UI_COLORS["panel"])
        controls.pack(fill=tk.X, pady=(4, 16))
        self._label(controls, "Confidence Threshold", size=11, weight="bold").pack(anchor=tk.W)

        threshold_row = tk.Frame(controls, bg=self.UI_COLORS["panel"])
        threshold_row.pack(fill=tk.X, pady=(6, 10))
        self.threshold_value = tk.StringVar(value="35%")
        threshold = ttk.Scale(
            threshold_row,
            from_=0.10,
            to=0.90,
            variable=self.confidence_threshold,
            command=self._on_threshold_change,
            style="Console.Horizontal.TScale",
        )
        threshold.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._label(
            threshold_row,
            "",
            textvariable=self.threshold_value,
            size=10,
            color=self.UI_COLORS["accent"],
            width=5,
        ).pack(side=tk.RIGHT, padx=(10, 0))

        camera_row = tk.Frame(controls, bg=self.UI_COLORS["panel"])
        camera_row.pack(fill=tk.X, pady=(2, 12))
        self._label(camera_row, "Camera", size=10, color=self.UI_COLORS["muted"]).pack(side=tk.LEFT)
        ttk.Spinbox(
            camera_row,
            from_=0,
            to=5,
            width=5,
            textvariable=self.camera_index,
            style="Console.TSpinbox",
        ).pack(side=tk.RIGHT)

        self.start_btn = self._button(controls, "Start Live Detection", self.start_detection, self.UI_COLORS["success"])
        self.start_btn.pack(fill=tk.X, pady=(0, 8))
        self.stop_btn = self._button(controls, "Stop Detection", self.stop_detection, self.UI_COLORS["warning"])
        self.stop_btn.configure(state=tk.DISABLED)
        self.stop_btn.pack(fill=tk.X, pady=(0, 8))

        secondary = tk.Frame(controls, bg=self.UI_COLORS["panel"])
        secondary.pack(fill=tk.X)
        secondary.columnconfigure((0, 1), weight=1, uniform="actions")
        self._button(secondary, "Analyze Image", self.analyze_image, self.UI_COLORS["panel_alt"]).grid(
            row=0, column=0, sticky="ew", padx=(0, 4)
        )
        self._button(secondary, "Save Snapshot", self.save_snapshot, self.UI_COLORS["panel_alt"]).grid(
            row=0, column=1, sticky="ew", padx=(4, 0)
        )

        self._label(body, "Detection Breakdown", size=12, weight="bold").pack(anchor=tk.W, pady=(4, 8))
        self.category_vars: dict[str, tk.StringVar] = {}
        for category, display in self.CATEGORY_LABELS.items():
            self.category_vars[category] = tk.StringVar(value="0")
            self._category_row(body, display, category).pack(fill=tk.X, pady=3)

        self._label(body, "Current Frame Evidence", size=12, weight="bold").pack(anchor=tk.W, pady=(18, 8))
        self.detection_list = tk.Listbox(
            body,
            height=8,
            bg=self.UI_COLORS["panel_alt"],
            fg=self.UI_COLORS["text"],
            selectbackground=self.UI_COLORS["accent"],
            activestyle="none",
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground=self.UI_COLORS["line"],
            font=("Segoe UI", 10),
        )
        self.detection_list.pack(fill=tk.BOTH, expand=True)
        self.detection_list.insert(tk.END, "  No detections yet")

        self._button(body, "Export Session CSV", self.export_session_csv, self.UI_COLORS["accent"]).pack(
            fill=tk.X, pady=(14, 8)
        )
        self._label(
            body,
            "",
            textvariable=self.status_text,
            size=10,
            color=self.UI_COLORS["muted"],
            wraplength=330,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(2, 0))

    def _build_video_area(self) -> None:
        header = tk.Frame(self.video_area, bg=self.UI_COLORS["bg"])
        header.pack(fill=tk.X, padx=24, pady=(18, 10))
        self.source_var = tk.StringVar(value="Source: idle")
        self._label(header, "Live Inference View", size=18, weight="bold").pack(side=tk.LEFT)
        self._label(header, "", textvariable=self.source_var, size=10, color=self.UI_COLORS["muted"]).pack(
            side=tk.RIGHT
        )

        frame_shell = tk.Frame(self.video_area, bg=self.UI_COLORS["line"])
        frame_shell.pack(fill=tk.BOTH, expand=True, padx=24, pady=(0, 18))
        self.video_label = tk.Label(frame_shell, bg="#05070a")
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)

    def _label(
        self,
        parent: tk.Widget,
        text: str,
        *,
        textvariable: tk.StringVar | None = None,
        size: int = 10,
        weight: str = "normal",
        color: str | None = None,
        **kwargs,
    ) -> tk.Label:
        return tk.Label(
            parent,
            text=text,
            textvariable=textvariable,
            bg=parent.cget("bg"),
            fg=color or self.UI_COLORS["text"],
            font=("Segoe UI", size, weight),
            **kwargs,
        )

    def _button(self, parent: tk.Widget, text: str, command, color: str) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=color,
            fg=self.UI_COLORS["text"],
            activebackground=color,
            activeforeground=self.UI_COLORS["text"],
            relief=tk.FLAT,
            bd=0,
            cursor="hand2",
            font=("Segoe UI", 10, "bold"),
            height=2,
        )

    def _info_panel(self, parent: tk.Widget, title: str, body: str) -> tk.Frame:
        panel = tk.Frame(parent, bg=self.UI_COLORS["panel_alt"], highlightthickness=1, highlightbackground=self.UI_COLORS["line"])
        inner = tk.Frame(panel, bg=self.UI_COLORS["panel_alt"])
        inner.pack(fill=tk.X, padx=14, pady=12)
        self._label(inner, title, size=11, weight="bold").pack(anchor=tk.W)
        self._label(inner, body, size=9, color=self.UI_COLORS["muted"], wraplength=310, justify=tk.LEFT).pack(
            anchor=tk.W, pady=(4, 0)
        )
        return panel

    def _kpi(self, parent: tk.Widget, title: str, value_var: tk.StringVar, color: str) -> tk.Frame:
        card = tk.Frame(parent, bg=self.UI_COLORS["panel_alt"], highlightthickness=1, highlightbackground=self.UI_COLORS["line"])
        self._label(card, title, size=9, color=self.UI_COLORS["muted"]).pack(anchor=tk.W, padx=12, pady=(10, 0))
        self._label(card, "", textvariable=value_var, size=24, weight="bold", color=color).pack(
            anchor=tk.W, padx=12, pady=(0, 10)
        )
        return card

    def _category_row(self, parent: tk.Widget, display: str, category: str) -> tk.Frame:
        row = tk.Frame(parent, bg=self.UI_COLORS["panel"])
        color = self._hex_color(self.CATEGORY_COLORS[category])
        marker = tk.Frame(row, width=8, height=22, bg=color)
        marker.pack(side=tk.LEFT, padx=(0, 9))
        marker.pack_propagate(False)
        self._label(row, display, size=10).pack(side=tk.LEFT)
        self._label(row, "", textvariable=self.category_vars[category], size=11, weight="bold", color=color).pack(
            side=tk.RIGHT
        )
        return row

    def _on_threshold_change(self, _value: str) -> None:
        self.threshold_value.set(f"{int(self.confidence_threshold.get() * 100)}%")

    def start_detection(self) -> None:
        if self.running:
            return

        self.cap = cv2.VideoCapture(int(self.camera_index.get()))
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = None
            self.status_text.set("Camera could not be opened. Try another camera index or use Analyze Image.")
            messagebox.showerror("Camera unavailable", "Unable to access the selected webcam.")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.session_started_at = datetime.now()
        self.running = True
        self.start_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        self.source_var.set(f"Source: camera {self.camera_index.get()}")
        self.status_text.set("Live detection is running.")

        self.worker = threading.Thread(target=self._camera_loop, name="weapon-detector", daemon=True)
        self.worker.start()

    def stop_detection(self) -> None:
        if not self.running and self.cap is None:
            return

        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.start_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        self.source_var.set("Source: idle")
        self.status_text.set("Detection stopped.")

    def analyze_image(self) -> None:
        if self.running:
            self.stop_detection()

        initial_file = DEFAULT_SAMPLE_IMAGE if DEFAULT_SAMPLE_IMAGE.exists() else APP_DIR
        path = filedialog.askopenfilename(
            title="Choose image for inference",
            initialdir=str(initial_file.parent if initial_file.is_file() else initial_file),
            filetypes=(("Images", "*.png *.jpg *.jpeg *.bmp *.webp"), ("All files", "*.*")),
        )
        if not path and DEFAULT_SAMPLE_IMAGE.exists():
            path = str(DEFAULT_SAMPLE_IMAGE)
        if not path:
            return

        frame = cv2.imread(path)
        if frame is None:
            messagebox.showerror("Invalid image", "The selected file could not be read as an image.")
            return

        self.status_text.set(f"Analyzing {Path(path).name}...")
        self.session_started_at = self.session_started_at or datetime.now()
        threading.Thread(
            target=self._single_image_inference,
            args=(frame, Path(path).name),
            daemon=True,
        ).start()

    def save_snapshot(self) -> None:
        if self.last_frame_bgr is None:
            messagebox.showinfo("No frame", "Run detection or analyze an image before saving a snapshot.")
            return

        output_path = APP_DIR / f"detection_snapshot_{datetime.now():%Y%m%d_%H%M%S}.jpg"
        cv2.imwrite(str(output_path), self.last_frame_bgr)
        self.status_text.set(f"Snapshot saved: {output_path.name}")

    def export_session_csv(self) -> None:
        if not self.session_events:
            messagebox.showinfo("No detections", "There are no session detections to export yet.")
            return

        output_path = APP_DIR / f"detection_session_{datetime.now():%Y%m%d_%H%M%S}.csv"
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("timestamp", "source", "label", "category", "confidence", "bbox"),
            )
            writer.writeheader()
            writer.writerows(self.session_events)

        self.status_text.set(f"Session CSV exported: {output_path.name}")

    def _camera_loop(self) -> None:
        previous_time = time.perf_counter()
        while self.running and self.cap is not None and self.cap.isOpened():
            ok, frame = self.cap.read()
            if not ok:
                self._set_status_from_worker("Camera frame could not be read. Detection stopped.")
                break

            now = time.perf_counter()
            fps = 1.0 / max(now - previous_time, 1e-6)
            previous_time = now
            result = self._run_inference(frame, source_name=f"camera {self.camera_index.get()}", fps=fps)
            self._push_result(result)

        self.running = False

    def _single_image_inference(self, frame, source_name: str) -> None:
        result = self._run_inference(frame, source_name=source_name, fps=0.0)
        self._push_result(result)

    def _run_inference(self, frame, source_name: str, fps: float) -> FrameResult:
        started = time.perf_counter()
        results = self.model(frame, verbose=False)
        inference_ms = (time.perf_counter() - started) * 1000
        detections = self._parse_detections(results)
        annotated = self._annotate_frame(frame.copy(), detections, inference_ms, fps)
        return FrameResult(
            frame=annotated,
            detections=detections,
            inference_ms=inference_ms,
            fps=fps,
            source_name=source_name,
            timestamp=datetime.now(),
        )

    def _parse_detections(self, results) -> list[Detection]:
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        threshold = self.confidence_threshold.get()
        detections: list[Detection] = []
        for box in boxes:
            confidence = float(box.conf[0])
            if confidence < threshold:
                continue

            class_id = int(box.cls[0])
            label = self.model_names.get(class_id, "weapon")
            category = self._categorize(label)
            x1, y1, x2, y2 = (int(value) for value in box.xyxy[0])
            detections.append(
                Detection(
                    label=label,
                    category=category,
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                )
            )
        return detections

    def _categorize(self, label: str) -> str:
        lower = label.lower()
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            if any(keyword in lower for keyword in keywords):
                return category
        return "other"

    def _annotate_frame(self, frame, detections: Iterable[Detection], inference_ms: float, fps: float):
        detections = list(detections)
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            color = self.CATEGORY_COLORS[detection.category]
            label = f"{detection.label} | {detection.confidence:.0%}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
            label_y = max(24, y1)
            cv2.rectangle(frame, (x1, label_y - 24), (x1 + label_size[0] + 12, label_y), color, -1)
            cv2.putText(
                frame,
                label,
                (x1 + 6, label_y - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        banner = f"Detections: {len(detections)} | Threshold: {self.confidence_threshold.get():.0%} | Latency: {inference_ms:.0f} ms"
        if fps:
            banner += f" | FPS: {fps:.1f}"
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 38), (8, 12, 16), -1)
        cv2.putText(frame, banner, (14, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (244, 247, 251), 2, cv2.LINE_AA)
        return frame

    def _push_result(self, result: FrameResult) -> None:
        if self.result_queue.full():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                pass
        self.result_queue.put(result)

    def _poll_results(self) -> None:
        try:
            while True:
                result = self.result_queue.get_nowait()
                self._render_result(result)
        except queue.Empty:
            pass
        self.root.after(30, self._poll_results)

    def _render_result(self, result: FrameResult) -> None:
        self.last_frame_bgr = result.frame.copy()
        self.recent_fps.append(result.fps)
        self._display_frame(result.frame)
        self._update_metrics(result)
        self._append_session_events(result)
        self.source_var.set(f"Source: {result.source_name}")
        self.status_text.set(
            f"Last update: {result.timestamp:%H:%M:%S} | {len(result.detections)} detection(s) above threshold."
        )

    def _update_metrics(self, result: FrameResult) -> None:
        category_counts = defaultdict(int)
        for detection in result.detections:
            category_counts[detection.category] += 1

        self.on_screen_var.set(str(len(result.detections)))
        self.total_var.set(str(len(self.session_events) + len(result.detections)))
        active_fps = [fps for fps in self.recent_fps if fps > 0]
        self.fps_var.set(f"{(sum(active_fps) / len(active_fps)):.1f}" if active_fps else "-")
        self.inference_var.set(f"{result.inference_ms:.0f} ms")

        for category, variable in self.category_vars.items():
            variable.set(str(category_counts.get(category, 0)))

        self.detection_list.delete(0, tk.END)
        if not result.detections:
            self.detection_list.insert(tk.END, "  No weapons detected above threshold")
            return

        for detection in sorted(result.detections, key=lambda item: item.confidence, reverse=True):
            display = self.CATEGORY_LABELS[detection.category]
            self.detection_list.insert(
                tk.END,
                f"  {display}: {detection.label} ({detection.confidence:.0%})",
            )

    def _append_session_events(self, result: FrameResult) -> None:
        for detection in result.detections:
            self.session_events.append(
                {
                    "timestamp": result.timestamp.isoformat(timespec="seconds"),
                    "source": result.source_name,
                    "label": detection.label,
                    "category": detection.category,
                    "confidence": f"{detection.confidence:.4f}",
                    "bbox": ",".join(str(value) for value in detection.bbox),
                }
            )

    def _display_frame(self, frame) -> None:
        frame_height = max(self.video_label.winfo_height(), 1)
        frame_width = max(self.video_label.winfo_width(), 1)
        height, width = frame.shape[:2]
        scale = min(frame_width / width, frame_height / height)
        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))
        resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

        canvas = cv2.copyMakeBorder(
            resized,
            (frame_height - new_height) // 2,
            frame_height - new_height - (frame_height - new_height) // 2,
            (frame_width - new_width) // 2,
            frame_width - new_width - (frame_width - new_width) // 2,
            cv2.BORDER_CONSTANT,
            value=(5, 7, 10),
        )
        image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        photo = ImageTk.PhotoImage(image=image)
        self.video_label.configure(image=photo)
        self.video_label.image = photo

    def _show_placeholder(self) -> None:
        placeholder = cv2.imread(str(DEFAULT_SAMPLE_IMAGE)) if DEFAULT_SAMPLE_IMAGE.exists() else None
        if placeholder is None:
            placeholder = self._blank_frame("Start live detection or analyze an image")
        else:
            placeholder = self._annotate_frame(placeholder, [], 0, 0)
        self.last_frame_bgr = placeholder.copy()
        self.root.after(100, lambda: self._display_frame(placeholder))

    def _blank_frame(self, text: str):
        frame = cv2.UMat(720, 1280, cv2.CV_8UC3).get()
        frame[:] = (8, 12, 16)
        cv2.putText(frame, text, (330, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (154, 167, 180), 2, cv2.LINE_AA)
        return frame

    def _set_status_from_worker(self, message: str) -> None:
        self.root.after(0, lambda: self.status_text.set(message))

    @staticmethod
    def _hex_color(color: tuple[int, int, int]) -> str:
        return f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"

    def on_closing(self) -> None:
        self.stop_detection()
        self.root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the AI weapon detection desktop console.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Path to the trained YOLO .pt model.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index to use for live detection.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    root = tk.Tk()
    try:
        WeaponDetectionApp(root, model_path=model_path, camera_index=args.camera)
        root.mainloop()
    except Exception as exc:
        messagebox.showerror("Application error", str(exc))
        root.destroy()
        raise


if __name__ == "__main__":
    main()
