from __future__ import annotations

import argparse
import csv
import json
import math
import re
import socket
import time
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple
from urllib.parse import urlparse

import monitor_progress as monitor


REPO_ROOT = Path(__file__).resolve().parent


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Experiment Progress</title>
  <style>
    :root {
      --bg: #070808;
      --bg-2: #111315;
      --surface: #151719;
      --surface-2: #1d2024;
      --surface-3: #0d0f11;
      --text: #f4f4ef;
      --muted: #a7adb2;
      --line: #3f4449;
      --line-strong: #6f767d;
      --nasa-red: #c8463c;
      --nasa-blue: #3f6f9f;
      --cyan: #b7c6cc;
      --green: #9fbd9a;
      --amber: #cfc06d;
      --ink: #070808;
      --ok-bg: rgba(159, 189, 154, 0.14);
      --warn-bg: rgba(207, 192, 109, 0.16);
      --wait-bg: rgba(167, 173, 178, 0.14);
      --bad-bg: rgba(200, 70, 60, 0.16);
      --shadow: 0 18px 44px rgba(0, 0, 0, 0.38);
      --titanium-a: #303438;
      --titanium-b: #111315;
      --titanium-c: #747a80;
      --titanium-highlight: rgba(244, 244, 239, 0.16);
      --font-body: "Bahnschrift", "Segoe UI", Arial, sans-serif;
      --font-display: "Agency FB", "Bahnschrift SemiBold", "Segoe UI", Arial, sans-serif;
      --font-readout: "Cascadia Mono", "Consolas", "Courier New", monospace;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      background:
        radial-gradient(circle at 12% 20%, rgba(244, 244, 239, 0.08) 1px, transparent 1.8px) 0 0 / 82px 82px,
        radial-gradient(circle at 68% 32%, rgba(183, 198, 204, 0.07) 1px, transparent 1.8px) 0 0 / 132px 132px,
        linear-gradient(90deg, rgba(200, 70, 60, 0.045), transparent 25%, rgba(63, 111, 159, 0.055) 75%, transparent),
        linear-gradient(180deg, var(--bg) 0%, var(--bg-2) 52%, #070808 100%);
      color: var(--text);
      font-family: var(--font-body);
      line-height: 1.45;
      min-height: 100vh;
    }

    main {
      width: min(1260px, calc(100% - 32px));
      margin: 0 auto;
      padding: 16px 0 34px;
    }

    header {
      display: flex;
      justify-content: space-between;
      gap: 18px;
      align-items: center;
      margin-bottom: 12px;
      min-height: 82px;
      padding: 12px 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background:
        linear-gradient(90deg, rgba(200, 70, 60, 0.12), transparent 26%, rgba(63, 111, 159, 0.08) 76%, transparent),
        linear-gradient(180deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.01)),
        var(--surface);
      box-shadow: var(--shadow);
      position: relative;
      overflow: hidden;
    }

    header::before {
      content: "";
      position: absolute;
      left: 0;
      right: 0;
      top: 0;
      height: 4px;
      background: linear-gradient(90deg, var(--nasa-red) 0 34%, #ffffff 34% 42%, var(--nasa-blue) 42% 100%);
    }

    .mission-brand {
      display: flex;
      align-items: center;
      gap: 14px;
      min-width: 0;
    }

    .nasa-mark {
      width: 58px;
      height: 58px;
      object-fit: contain;
      flex: 0 0 auto;
      filter: drop-shadow(0 8px 18px rgba(0, 0, 0, 0.42));
    }

    .kicker {
      color: var(--cyan);
      font-size: 12px;
      font-weight: 800;
      letter-spacing: 0;
      text-transform: uppercase;
      font-family: var(--font-readout);
      text-shadow: 0 0 12px rgba(183, 198, 204, 0.36);
    }

    h1 {
      margin: 0;
      font-size: 30px;
      line-height: 1.05;
      font-weight: 800;
      font-family: var(--font-display);
      text-transform: uppercase;
      text-shadow: 0 0 16px rgba(183, 198, 204, 0.18);
    }

    h2 {
      margin: 0 0 10px;
      font-size: 16px;
      font-weight: 760;
      display: flex;
      align-items: center;
      gap: 10px;
      font-family: var(--font-display);
      text-transform: uppercase;
      text-shadow: 0 0 14px rgba(183, 198, 204, 0.18);
    }

    .subline {
      color: var(--muted);
      margin-top: 5px;
      max-width: 780px;
      overflow-wrap: anywhere;
      font-family: var(--font-readout);
      font-size: 13px;
    }

    .status-pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      min-height: 30px;
      padding: 6px 9px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(5, 7, 11, 0.64);
      color: var(--muted);
      white-space: nowrap;
      box-shadow: var(--shadow);
      font-variant-numeric: tabular-nums;
      font-family: var(--font-readout);
      text-transform: uppercase;
    }

    .dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: var(--amber);
    }

    .dot.fresh { background: var(--green); box-shadow: 0 0 12px rgba(159, 189, 154, 0.62); }
    .dot.stale { background: var(--nasa-red); box-shadow: 0 0 12px rgba(200, 70, 60, 0.62); }

    .ui-icon {
      width: 24px;
      height: 24px;
      border: 1px solid var(--line-strong);
      border-radius: 8px;
      display: inline-grid;
      place-items: center;
      background:
        linear-gradient(180deg, rgba(244, 244, 239, 0.11), rgba(17, 19, 21, 0.22)),
        linear-gradient(135deg, rgba(183, 198, 204, 0.08), rgba(63, 111, 159, 0.06));
      position: relative;
      flex: 0 0 auto;
    }

    .ui-icon::before,
    .ui-icon::after {
      content: "";
      position: absolute;
      display: block;
    }

    .ui-complete::before {
      width: 13px;
      height: 7px;
      border-left: 2px solid var(--green);
      border-bottom: 2px solid var(--green);
      transform: rotate(-45deg);
      top: 8px;
      left: 7px;
    }

    .ui-target::before {
      width: 13px;
      height: 13px;
      border: 2px solid var(--nasa-red);
      border-radius: 50%;
    }

    .ui-target::after {
      width: 4px;
      height: 4px;
      border-radius: 50%;
      background: var(--cyan);
    }

    .ui-stage::before {
      width: 14px;
      height: 14px;
      border: 2px solid var(--cyan);
      border-left-color: transparent;
      border-radius: 50%;
      transform: rotate(-30deg);
    }

    .ui-speed::before {
      width: 15px;
      height: 8px;
      border: 2px solid var(--amber);
      border-bottom: 0;
      border-radius: 14px 14px 0 0;
      top: 8px;
    }

    .ui-speed::after {
      width: 9px;
      height: 2px;
      background: var(--amber);
      transform: rotate(-28deg);
      transform-origin: left center;
      top: 15px;
      left: 13px;
    }

    .ui-progress::before {
      width: 16px;
      height: 4px;
      border: 1px solid var(--cyan);
      border-radius: 4px;
    }

    .ui-progress::after {
      width: 9px;
      height: 4px;
      border-radius: 4px 0 0 4px;
      background: var(--cyan);
      left: 6px;
    }

    .ui-setup::before {
      width: 14px;
      height: 14px;
      border: 2px solid var(--nasa-blue);
      transform: rotate(45deg);
    }

    .ui-metrics::before {
      width: 3px;
      height: 14px;
      background: var(--green);
      box-shadow: 6px -4px 0 var(--cyan), 12px 3px 0 var(--amber);
      bottom: 6px;
      left: 6px;
    }

    .ui-config::before {
      width: 14px;
      height: 10px;
      border: 2px solid var(--cyan);
      border-radius: 3px;
    }

    .ui-config::after {
      width: 10px;
      height: 2px;
      background: var(--cyan);
      top: 13px;
    }

    .ui-log::before {
      width: 15px;
      height: 12px;
      border: 2px solid var(--amber);
      border-radius: 2px;
    }

    .ui-log::after {
      width: 9px;
      height: 2px;
      background: var(--amber);
      box-shadow: 0 4px 0 var(--amber);
      left: 9px;
      top: 9px;
    }

    .layout {
      display: grid;
      grid-template-columns: minmax(0, 1.4fr) minmax(300px, 0.8fr);
      gap: 12px;
    }

    .panel {
      background:
        linear-gradient(126deg, transparent 0 18%, var(--titanium-highlight) 19%, transparent 24% 100%),
        repeating-linear-gradient(112deg, rgba(255, 255, 255, 0.045) 0 1px, rgba(0, 0, 0, 0.035) 1px 4px),
        linear-gradient(145deg, var(--titanium-c), var(--titanium-a) 40%, var(--titanium-b));
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      box-shadow: var(--shadow);
      position: relative;
      overflow: hidden;
    }

    .panel::before {
      content: "";
      position: absolute;
      left: 0;
      top: 0;
      bottom: 0;
      width: 3px;
      background: var(--nasa-blue);
    }

    .span-2 { grid-column: 1 / -1; }

    .metrics {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 12px;
    }

    .top-metrics {
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }

    .eval-metrics {
      grid-template-columns: repeat(5, minmax(0, 1fr));
    }

    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 11px;
      background:
        linear-gradient(118deg, rgba(244, 244, 239, 0.12), transparent 18% 100%),
        linear-gradient(302deg, rgba(183, 198, 204, 0.05), transparent 42%),
        repeating-linear-gradient(110deg, rgba(255, 255, 255, 0.045) 0 1px, rgba(0, 0, 0, 0.055) 1px 5px),
        linear-gradient(145deg, #555b61, #24282d 46%, #111315);
      min-width: 0;
      box-shadow: 0 10px 26px rgba(0, 0, 0, 0.22);
    }

    .metric-top {
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 6px;
    }

    .metric-label {
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 0;
      text-transform: uppercase;
      font-weight: 700;
      font-family: var(--font-readout);
    }

    .metric-value {
      font-size: 24px;
      font-weight: 760;
      overflow-wrap: anywhere;
      font-family: var(--font-readout);
      color: #f8fbff;
      text-shadow: 0 0 12px rgba(244, 244, 239, 0.14);
    }

    .metric-note {
      margin-top: 3px;
      color: var(--muted);
      font-size: 13px;
      overflow-wrap: anywhere;
      font-family: var(--font-readout);
    }

    .flight-speed-panel {
      grid-column: 1 / -1;
      padding: 14px;
      background:
        linear-gradient(135deg, rgba(183, 198, 204, 0.08), transparent 28%),
        linear-gradient(315deg, rgba(200, 70, 60, 0.045), transparent 30%),
        linear-gradient(116deg, rgba(244, 244, 239, 0.14), transparent 20% 100%),
        repeating-linear-gradient(112deg, rgba(255, 255, 255, 0.045) 0 1px, rgba(0, 0, 0, 0.05) 1px 4px),
        linear-gradient(145deg, #6f7478, #303438 44%, #111315);
    }

    .flight-speed-panel::before {
      background: linear-gradient(180deg, #c4c8c8, var(--nasa-blue));
    }

    .flight-dashboard {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      align-items: stretch;
    }

    .instrument-panel {
      --gauge-angle: -130deg;
      min-height: 316px;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background:
        linear-gradient(120deg, rgba(244, 244, 239, 0.16), transparent 22% 100%),
        linear-gradient(300deg, rgba(183, 198, 204, 0.06), transparent 46%),
        repeating-linear-gradient(112deg, rgba(255, 255, 255, 0.052) 0 1px, rgba(0, 0, 0, 0.070) 1px 5px),
        linear-gradient(145deg, #62676c, #2a2f34 46%, #151719);
      box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, 0.14),
        inset 0 -20px 30px rgba(0, 0, 0, 0.18),
        0 12px 28px rgba(0, 0, 0, 0.24);
      display: grid;
      grid-template-rows: auto auto 1fr auto;
      gap: 7px;
      position: relative;
      overflow: hidden;
    }

    .instrument-panel::before {
      content: "";
      position: absolute;
      inset: 8px;
      pointer-events: none;
      background:
        radial-gradient(circle at 0 0, #050606 0 3px, #70777d 3.5px 5px, transparent 5.5px),
        radial-gradient(circle at 100% 0, #050606 0 3px, #70777d 3.5px 5px, transparent 5.5px),
        radial-gradient(circle at 0 100%, #050606 0 3px, #70777d 3.5px 5px, transparent 5.5px),
        radial-gradient(circle at 100% 100%, #050606 0 3px, #70777d 3.5px 5px, transparent 5.5px);
      opacity: 0.88;
    }

    .instrument-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      color: var(--muted);
      font-family: var(--font-readout);
      font-size: 12px;
      text-transform: uppercase;
      position: relative;
      z-index: 1;
    }

    .instrument-code {
      color: var(--cyan);
    }

    .annunciator-row {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 5px;
      position: relative;
      z-index: 1;
    }

    .annunciator-row span {
      min-height: 31px;
      display: grid;
      place-items: center;
      border: 1px solid rgba(244, 244, 239, 0.16);
      border-radius: 4px;
      background:
        linear-gradient(180deg, rgba(244, 244, 239, 0.09), rgba(7, 8, 8, 0.42)),
        #14171a;
      color: #d9dedc;
      font-family: var(--font-readout);
      font-size: 10px;
      text-transform: uppercase;
      box-shadow: inset 0 0 10px rgba(0, 0, 0, 0.42);
      overflow: hidden;
    }

    .annunciator-row span:nth-child(1) { color: var(--green); }
    .annunciator-row span:nth-child(4) { color: var(--amber); }

    .annunciator-row b {
      display: block;
      max-width: 100%;
      color: var(--text);
      font-size: 11px;
      line-height: 1.05;
      font-weight: 800;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .gauge-face {
      position: relative;
      align-self: center;
      justify-self: center;
      width: min(236px, 100%);
      aspect-ratio: 1;
      border-radius: 50%;
      background:
        radial-gradient(circle at center, #101214 0 43%, transparent 44%),
        repeating-conic-gradient(from -132deg, rgba(244, 244, 239, 0.92) 0deg 1.4deg, transparent 1.4deg 6deg),
        repeating-conic-gradient(from -132deg, rgba(244, 244, 239, 0.45) 0deg 0.8deg, transparent 0.8deg 2deg),
        conic-gradient(from -135deg, rgba(244, 244, 239, 0.62) 0deg 270deg, transparent 270deg 360deg),
        radial-gradient(circle at 50% 54%, #050606 0 59%, #555b60 60% 64%, #090a0c 65% 100%);
      border: 8px solid #121416;
      box-shadow:
        inset 0 0 22px rgba(0, 0, 0, 0.82),
        inset 0 0 0 1px rgba(255, 255, 255, 0.16),
        0 0 0 1px rgba(244, 244, 239, 0.16),
        0 18px 32px rgba(0, 0, 0, 0.36);
      overflow: hidden;
    }

    .gauge-face::before {
      content: "";
      position: absolute;
      inset: 24px;
      border-radius: 50%;
      background:
        radial-gradient(circle, transparent 0 44%, rgba(244, 244, 239, 0.18) 44.5% 45.5%, transparent 46%),
        radial-gradient(circle, transparent 0 68%, rgba(244, 244, 239, 0.10) 68.5% 69.5%, transparent 70%),
        linear-gradient(90deg, transparent 0 48%, rgba(244, 244, 239, 0.18) 49% 51%, transparent 52%),
        linear-gradient(0deg, transparent 0 48%, rgba(244, 244, 239, 0.14) 49% 51%, transparent 52%);
      opacity: 0.58;
    }

    .gauge-face::after {
      content: "";
      position: absolute;
      inset: 0;
      border-radius: 50%;
      background:
        radial-gradient(circle at 35% 20%, rgba(244, 244, 239, 0.16), transparent 28%),
        linear-gradient(130deg, rgba(244, 244, 239, 0.10), transparent 34% 100%);
      pointer-events: none;
    }

    .gauge-scale {
      position: absolute;
      inset: 19px;
      border-radius: 50%;
      z-index: 1;
      font-family: var(--font-readout);
      font-size: 12px;
      color: rgba(244, 244, 239, 0.86);
      pointer-events: none;
    }

    .gauge-scale span {
      position: absolute;
      transform: translate(-50%, -50%);
      text-shadow: 0 1px 3px rgba(0, 0, 0, 0.9);
    }

    .gauge-scale span:nth-child(1) { left: 50%; top: 94%; }
    .gauge-scale span:nth-child(2) { left: 19%; top: 77%; }
    .gauge-scale span:nth-child(3) { left: 11%; top: 43%; }
    .gauge-scale span:nth-child(4) { left: 34%; top: 16%; }
    .gauge-scale span:nth-child(5) { left: 67%; top: 16%; }
    .gauge-scale span:nth-child(6) { left: 90%; top: 43%; }

    .gauge-needle {
      position: absolute;
      left: 50%;
      top: 50%;
      width: 4px;
      height: 42%;
      background: linear-gradient(180deg, #f7f5e8, #b8b9ad);
      border-radius: 6px;
      transform-origin: 50% 100%;
      transform: translate(-50%, -100%) rotate(var(--gauge-angle));
      box-shadow: 0 0 10px rgba(244, 244, 239, 0.42);
      z-index: 2;
    }

    .gauge-hub {
      position: absolute;
      left: 50%;
      top: 50%;
      width: 28px;
      height: 28px;
      border-radius: 50%;
      background: radial-gradient(circle, #f4f4ef 0 18%, #b8bdbf 19% 42%, #202327 43% 100%);
      transform: translate(-50%, -50%);
      box-shadow: 0 0 14px rgba(244, 244, 239, 0.26);
      z-index: 3;
    }

    .gauge-readout {
      position: absolute;
      left: 50%;
      top: 57%;
      transform: translateX(-50%);
      display: grid;
      place-items: center;
      gap: 3px;
      z-index: 4;
      text-align: center;
      width: 74%;
    }

    .gauge-value {
      color: #ffffff;
      font-family: var(--font-readout);
      font-size: 31px;
      font-weight: 800;
      line-height: 1;
      text-shadow:
        0 0 14px rgba(244, 244, 239, 0.28),
        0 0 28px rgba(183, 198, 204, 0.16);
    }

    .gauge-caption,
    .instrument-note {
      color: var(--muted);
      font-family: var(--font-readout);
      font-size: 12px;
      text-transform: uppercase;
      overflow-wrap: anywhere;
    }

    .instrument-readouts {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 7px;
      position: relative;
      z-index: 1;
    }

    .details-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin-top: 14px;
    }

    .detail {
      background:
        linear-gradient(120deg, rgba(244, 244, 239, 0.10), transparent 18% 100%),
        linear-gradient(306deg, rgba(183, 198, 204, 0.05), transparent 46%),
        repeating-linear-gradient(110deg, rgba(255, 255, 255, 0.04) 0 1px, rgba(0, 0, 0, 0.055) 1px 5px),
        linear-gradient(145deg, #4e5357, #24282c 46%, #111315);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      min-width: 0;
    }

    .detail span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 3px;
      font-family: var(--font-readout);
      text-transform: uppercase;
    }

    .detail strong {
      display: block;
      overflow-wrap: anywhere;
      font-family: var(--font-readout);
      color: #f8fbff;
    }

    .detail small {
      display: block;
      margin-top: 4px;
      color: var(--muted);
      font-family: var(--font-readout);
      font-size: 12px;
      text-transform: uppercase;
      overflow-wrap: anywhere;
    }

    .throughput-mini {
      grid-column: 1 / -1;
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 8px 12px;
      align-items: end;
      background:
        linear-gradient(90deg, rgba(207, 192, 109, 0.10), transparent 34%),
        linear-gradient(120deg, rgba(244, 244, 239, 0.12), transparent 20% 100%),
        repeating-linear-gradient(110deg, rgba(255, 255, 255, 0.045) 0 1px, rgba(0, 0, 0, 0.060) 1px 5px),
        linear-gradient(145deg, #5d6469, #292e33 46%, #111315);
    }

    .throughput-mini span {
      grid-column: 1 / -1;
    }

    .throughput-mini strong {
      color: #f4f4ef;
      font-size: 22px;
      line-height: 1.05;
      text-shadow: 0 0 12px rgba(244, 244, 239, 0.24);
    }

    .throughput-mini small {
      margin-top: 0;
      text-align: right;
      color: var(--amber);
    }

    .intel-grid {
      display: grid;
      grid-template-columns: minmax(0, 0.9fr) minmax(0, 1.1fr);
      gap: 10px;
    }

    .micro-panel {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background:
        linear-gradient(118deg, rgba(244, 244, 239, 0.10), transparent 18% 100%),
        repeating-linear-gradient(110deg, rgba(255, 255, 255, 0.035) 0 1px, rgba(0, 0, 0, 0.055) 1px 5px),
        linear-gradient(145deg, #4f5458, #22262a 48%, #101214);
      min-width: 0;
    }

    .micro-panel h3 {
      margin: 0 0 8px;
      font-family: var(--font-display);
      font-size: 15px;
      text-transform: uppercase;
    }

    .fact-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 7px;
    }

    .fact {
      min-width: 0;
      padding: 8px;
      border: 1px solid rgba(244, 244, 239, 0.13);
      border-radius: 6px;
      background: rgba(7, 8, 8, 0.28);
      font-family: var(--font-readout);
    }

    .fact span {
      display: block;
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
    }

    .fact strong {
      display: block;
      margin-top: 3px;
      color: var(--text);
      font-size: 13px;
      overflow-wrap: anywhere;
    }

    .chart-shell {
      border: 1px solid var(--line);
      border-radius: 8px;
      background:
        linear-gradient(180deg, rgba(244, 244, 239, 0.07), rgba(7, 8, 8, 0.16)),
        #111315;
      padding: 10px;
      min-height: 280px;
    }

    .chart-head {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 8px;
      font-family: var(--font-readout);
      text-transform: uppercase;
    }

    .chart-head strong {
      overflow-wrap: anywhere;
    }

    .chart-head span {
      color: var(--muted);
      font-size: 12px;
      text-align: right;
    }

    .chart {
      width: 100%;
      min-height: 246px;
      display: block;
      overflow: visible;
    }

    .chart-grid {
      stroke: rgba(244, 244, 239, 0.13);
      stroke-width: 1;
    }

    .chart-axis,
    .chart-text {
      fill: var(--muted);
      font-family: var(--font-readout);
      font-size: 11px;
      text-transform: uppercase;
    }

    .chart-line {
      fill: none;
      stroke-width: 2.4;
      stroke-linejoin: round;
      stroke-linecap: round;
    }

    .chart-dot {
      stroke: #090a0c;
      stroke-width: 1.5;
    }

    .chart-legend {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-top: 6px;
      color: var(--muted);
      font-family: var(--font-readout);
      font-size: 12px;
      text-transform: uppercase;
    }

    .chart-legend span::before {
      content: "";
      display: inline-block;
      width: 18px;
      height: 2px;
      margin-right: 6px;
      vertical-align: middle;
      background: currentColor;
    }

    .runs {
      display: grid;
      gap: 10px;
    }

    .run-row {
      display: grid;
      grid-template-columns: 92px minmax(0, 1.3fr) minmax(0, 0.9fr) 112px 112px 112px;
      gap: 10px;
      align-items: center;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background:
        linear-gradient(120deg, rgba(246, 250, 255, 0.10), transparent 18% 100%),
        repeating-linear-gradient(110deg, rgba(255, 255, 255, 0.035) 0 1px, rgba(0, 0, 0, 0.05) 1px 5px),
        linear-gradient(145deg, #3d4652, #1c232c 48%, #0d1015);
    }

    .run-main {
      min-width: 0;
      overflow-wrap: anywhere;
    }

    .run-main strong,
    .experiment-row strong {
      display: flex;
      align-items: center;
      gap: 9px;
      min-width: 0;
    }

    .run-main strong .ui-icon,
    .experiment-row strong .ui-icon {
      width: 24px;
      height: 24px;
      border-radius: 8px;
    }

    .badge {
      display: inline-flex;
      justify-content: center;
      min-width: 86px;
      padding: 5px 9px;
      border-radius: 8px;
      font-size: 13px;
      font-weight: 700;
      color: var(--text);
      background: var(--wait-bg);
      border: 1px solid var(--line);
      font-family: var(--font-readout);
      text-transform: uppercase;
    }

    .badge.done { background: var(--ok-bg); color: var(--green); }
    .badge.active, .badge.started, .badge.created, .badge.writing { background: var(--warn-bg); color: var(--amber); }
    .badge.error { background: var(--bad-bg); color: var(--nasa-red); }

    .run-number {
      font-variant-numeric: tabular-nums;
      color: var(--muted);
      font-family: var(--font-readout);
    }

    .mono {
      font-family: var(--font-readout);
      font-size: 13px;
    }

    .log-line {
      min-height: 78px;
      padding: 12px;
      border-radius: 8px;
      border: 1px solid var(--line);
      background:
        linear-gradient(120deg, rgba(255, 255, 255, 0.08), transparent 18% 100%),
        repeating-linear-gradient(110deg, rgba(255, 255, 255, 0.025) 0 1px, rgba(0, 0, 0, 0.06) 1px 5px),
        linear-gradient(145deg, #1f2936, #080d14 46%, #05070b);
      color: #d8f4ff;
      overflow-wrap: anywhere;
      white-space: pre-wrap;
      box-shadow: inset 0 0 0 1px rgba(183, 198, 204, 0.06);
    }

    .config-list {
      display: grid;
      gap: 8px;
      margin: 0;
    }

    .config-list div {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      border-bottom: 1px solid var(--line);
      padding-bottom: 8px;
    }

    .config-list div:last-child { border-bottom: 0; padding-bottom: 0; }
    .config-list dt { color: var(--muted); }
    .config-list dt,
    .config-list dd {
      font-family: var(--font-readout);
      text-transform: uppercase;
    }
    .config-list dd { margin: 0; text-align: right; overflow-wrap: anywhere; color: #f8fbff; }

    .table-wrap {
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      min-width: 720px;
      background:
        repeating-linear-gradient(112deg, rgba(255, 255, 255, 0.035) 0 1px, rgba(0, 0, 0, 0.05) 1px 5px),
        linear-gradient(145deg, #3f4854, #1b222b 48%, #0d1015);
    }

    th, td {
      text-align: left;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      font-variant-numeric: tabular-nums;
    }

    th {
      color: var(--muted);
      font-size: 13px;
      font-weight: 700;
      background: var(--surface-2);
      font-family: var(--font-readout);
      text-transform: uppercase;
    }

    tr:last-child td { border-bottom: 0; }

    .experiment-list {
      display: grid;
      gap: 10px;
    }

    .experiment-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 110px 130px 130px;
      gap: 10px;
      align-items: center;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background:
        linear-gradient(120deg, rgba(246, 250, 255, 0.10), transparent 18% 100%),
        repeating-linear-gradient(110deg, rgba(255, 255, 255, 0.035) 0 1px, rgba(0, 0, 0, 0.05) 1px 5px),
        linear-gradient(145deg, #3d4652, #1c232c 48%, #0d1015);
    }

    .run-main strong,
    .experiment-row strong {
      font-family: var(--font-readout);
      text-transform: uppercase;
    }

    .empty {
      color: var(--muted);
      padding: 18px;
      border: 1px dashed var(--line);
      border-radius: 8px;
      background: var(--surface-2);
    }

    @media (max-width: 900px) {
      main { width: min(100% - 20px, 1180px); padding-top: 14px; }
      header, .layout { display: block; }
      header { padding: 14px; }
      .mission-brand { align-items: flex-start; }
      .nasa-mark { width: 54px; height: 54px; }
      .status-pill { margin-top: 12px; }
      .panel { margin-bottom: 12px; }
      .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .flight-dashboard { grid-template-columns: 1fr; }
      .intel-grid { grid-template-columns: 1fr; }
      .instrument-panel { min-height: 310px; }
      .gauge-value { font-size: 28px; }
      .instrument-readouts { grid-template-columns: 1fr; }
      .details-grid { grid-template-columns: 1fr; }
      .run-row { grid-template-columns: 1fr; align-items: stretch; }
      .experiment-row { grid-template-columns: 1fr; align-items: stretch; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div class="mission-brand">
        <img class="nasa-mark" src="https://www.nasa.gov/wp-content/uploads/2023/04/nasa-logo-web-rgb.png" alt="NASA logo">
        <div>
          <div class="kicker">DG-CSI Mission Console</div>
          <h1 id="experimentName">Telemetry Uplink</h1>
          <div class="subline" id="pathsLine">Loading...</div>
        </div>
      </div>
      <div class="status-pill"><span class="dot" id="freshDot"></span><span id="freshText">Checking log</span></div>
    </header>

    <section class="metrics top-metrics">
      <div class="metric">
        <div class="metric-top"><span class="ui-icon ui-complete"></span><div class="metric-label">Run packets</div></div>
        <div class="metric-value" id="completedValue">0/0</div>
        <div class="metric-note" id="completedNote">Waiting for metrics</div>
      </div>
      <div class="metric">
        <div class="metric-top"><span class="ui-icon ui-target"></span><div class="metric-label">Target lock</div></div>
        <div class="metric-value" id="activeTarget">--</div>
        <div class="metric-note" id="activeSeed">seed --</div>
      </div>
      <div class="metric">
        <div class="metric-top"><span class="ui-icon ui-stage"></span><div class="metric-label">Active phase</div></div>
        <div class="metric-value" id="stageValue">--</div>
        <div class="metric-note" id="epochValue">epoch unknown</div>
      </div>
    </section>

    <section class="layout">
      <div class="panel flight-speed-panel">
        <h2><span class="ui-icon ui-speed"></span>Flight Speed Panel</h2>
        <div class="flight-dashboard">
          <div class="instrument-panel gauge-panel" id="suiteGauge">
            <div class="instrument-title"><span>Whole realtime progress</span><span class="instrument-code">NAV-01</span></div>
            <div class="annunciator-row">
              <span>Runs <b id="navRuns">0/0</b></span>
              <span>Mode <b id="navMode">--</b></span>
              <span>Target <b id="navTarget">--</b></span>
              <span>Seed <b id="navSeed">--</b></span>
            </div>
            <div class="gauge-face">
              <div class="gauge-scale"><span>0</span><span>20</span><span>40</span><span>60</span><span>80</span><span>100</span></div>
              <div class="gauge-needle"></div>
              <div class="gauge-hub"></div>
              <div class="gauge-readout">
                <strong class="gauge-value" id="runPercent">0%</strong>
                <span class="gauge-caption">suite trajectory</span>
              </div>
            </div>
            <div class="instrument-note" id="trajectoryNote">continuous estimate</div>
          </div>

          <div class="instrument-panel gauge-panel" id="batchGauge">
            <div class="instrument-title"><span>Actual batch progress</span><span class="instrument-code">BCH-02</span></div>
            <div class="annunciator-row">
              <span>Stage <b id="batchStage">--</b></span>
              <span>Epoch <b id="batchEpoch">--</b></span>
              <span>Batch <b id="batchCount">--</b></span>
              <span>ETA <b id="batchEta">--</b></span>
            </div>
            <div class="gauge-face">
              <div class="gauge-scale"><span>0</span><span>20</span><span>40</span><span>60</span><span>80</span><span>100</span></div>
              <div class="gauge-needle"></div>
              <div class="gauge-hub"></div>
              <div class="gauge-readout">
                <strong class="gauge-value" id="loaderPercent">No signal</strong>
                <span class="gauge-caption" id="loaderDetail">awaiting stream</span>
              </div>
            </div>
            <div class="instrument-readouts">
              <div class="detail throughput-mini">
                <span>Throughput</span>
                <strong id="throughputMiniValue">--</strong>
                <small id="throughputMiniNote">ETA --</small>
              </div>
              <div class="detail"><span>Batch vector</span><strong id="loaderBatch">--</strong></div>
              <div class="detail"><span>Burn time</span><strong id="elapsedValue">--</strong></div>
              <div class="detail"><span>Signal age</span><strong id="lastUpdate">--</strong></div>
            </div>
          </div>
        </div>
      </div>

      <div class="panel span-2">
        <h2><span class="ui-icon ui-metrics"></span>Evaluation Telemetry</h2>
        <section class="metrics eval-metrics">
          <div class="metric">
            <div class="metric-top"><span class="ui-icon ui-target"></span><div class="metric-label">Mean accuracy</div></div>
            <div class="metric-value" id="metricAccuracy">--</div>
            <div class="metric-note" id="metricAccuracyNote">Waiting for completed runs</div>
          </div>
          <div class="metric">
            <div class="metric-top"><span class="ui-icon ui-metrics"></span><div class="metric-label">Macro F1</div></div>
            <div class="metric-value" id="metricF1">--</div>
            <div class="metric-note">Mean over completed runs</div>
          </div>
          <div class="metric">
            <div class="metric-top"><span class="ui-icon ui-progress"></span><div class="metric-label">Macro recall</div></div>
            <div class="metric-value" id="metricRecall">--</div>
            <div class="metric-note">Mean over completed runs</div>
          </div>
          <div class="metric">
            <div class="metric-top"><span class="ui-icon ui-progress"></span><div class="metric-label">Macro precision</div></div>
            <div class="metric-value" id="metricPrecision">--</div>
            <div class="metric-note">Mean over completed runs</div>
          </div>
          <div class="metric">
            <div class="metric-top"><span class="ui-icon ui-speed"></span><div class="metric-label">Top signal</div></div>
            <div class="metric-value" id="metricBest">--</div>
            <div class="metric-note" id="metricBestNote">No metrics yet</div>
          </div>
        </section>
        <div id="classMetrics"></div>
      </div>

      <div class="panel span-2">
        <h2><span class="ui-icon ui-config"></span>Dataset & Split Telemetry</h2>
        <div class="intel-grid">
          <section class="micro-panel">
            <h3>Dataset source</h3>
            <div class="fact-grid" id="datasetFacts"></div>
          </section>
          <section class="micro-panel">
            <h3>Active split</h3>
            <div class="fact-grid" id="splitFacts"></div>
          </section>
        </div>
        <div id="splitTable"></div>
      </div>

      <div class="panel span-2">
        <h2><span class="ui-icon ui-metrics"></span>Training Trajectory</h2>
        <div class="chart-shell">
          <div class="chart-head">
            <strong id="historyRun">No completed history yet</strong>
            <span id="historyNote">metrics.json is written when each run finishes</span>
          </div>
          <svg class="chart" id="historyChart" viewBox="0 0 760 260" role="img" aria-label="Training performance by epoch"></svg>
          <div class="chart-legend">
            <span style="color:#f4f4ef">Val accuracy</span>
            <span style="color:#9fbd9a">Val F1</span>
            <span style="color:#cfc06d">Val recall</span>
          </div>
        </div>
      </div>

      <div class="panel span-2">
        <h2><span class="ui-icon ui-config"></span>Ablation Modules</h2>
        <div class="experiment-list" id="experiments"></div>
      </div>

      <div class="panel span-2">
        <h2><span class="ui-icon ui-target"></span>Target Channels</h2>
        <div class="runs" id="runs"></div>
      </div>

      <div class="panel span-2">
        <h2><span class="ui-icon ui-log"></span>Uplink Stream</h2>
        <div class="log-line mono" id="latestLine">Waiting for log output...</div>
      </div>
    </section>
  </main>

  <script>
    const state = { lastPayload: null };

    function fmtPercent(value) {
      if (!Number.isFinite(value)) return "0%";
      return `${Math.max(0, Math.min(100, value)).toFixed(1)}%`;
    }

    function setText(id, value) {
      const element = document.getElementById(id);
      if (element) element.textContent = value;
    }

    function setGauge(id, value) {
      const element = document.getElementById(id);
      if (!element) return;
      const clamped = Math.max(0, Math.min(100, Number.isFinite(value) ? value : 0));
      const angle = -130 + clamped * 2.6;
      element.style.setProperty("--gauge-angle", `${angle}deg`);
    }

    function fmtNumber(value, digits = 4) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
      return Number(value).toFixed(digits);
    }

    function fmtAge(seconds) {
      if (seconds === null || seconds === undefined) return "not found";
      if (seconds < 2) return "just now";
      if (seconds < 60) return `${Math.round(seconds)}s ago`;
      if (seconds < 3600) return `${Math.round(seconds / 60)}m ago`;
      return `${Math.round(seconds / 3600)}h ago`;
    }

    function escapeHtml(value) {
      return String(value ?? "--")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }

    function shortList(values) {
      if (!values || !values.length) return "--";
      return values.join(", ");
    }

    function compactMap(map) {
      const entries = Object.entries(map || {});
      if (!entries.length) return "--";
      return entries.map(([key, value]) => `${key}:${value}`).join("  ");
    }

    function renderFacts(id, facts) {
      const element = document.getElementById(id);
      if (!element) return;
      element.innerHTML = facts.map(fact => `
        <div class="fact">
          <span>${escapeHtml(fact.label)}</span>
          <strong>${escapeHtml(fact.value)}</strong>
        </div>
      `).join("");
    }

    function renderDataSplit(payload) {
      const data = payload.dataset_info || {};
      const split = payload.split_info || {};
      renderFacts("datasetFacts", [
        { label: "Format", value: data.format || "--" },
        { label: "Environments", value: data.env_count ? `${data.env_count} (${shortList(data.envs)})` : "--" },
        { label: "Samples", value: data.dataset_size ?? "--" },
        { label: "Classes", value: data.class_count ?? "--" },
        { label: "Modalities", value: shortList(data.modalities) },
        { label: "Phase unwrap", value: data.unwrap_phase ? "enabled" : "disabled" },
      ]);
      renderFacts("splitFacts", [
        { label: "Mode", value: split.mode || payload.mode || "--" },
        { label: "Target", value: split.target_env || "--" },
        { label: "Source envs", value: shortList(split.source_envs) },
        { label: "Batch size", value: payload.config?.batch_size || "--" },
        { label: "Run batches", value: split.work?.run_batches ?? "--" },
        { label: "Train / Val / Test", value: `${split.train_count ?? "--"} / ${split.val_count ?? "--"} / ${split.test_count ?? "--"}` },
      ]);

      const splits = split.splits || [];
      const table = document.getElementById("splitTable");
      if (!table) return;
      if (!splits.length) {
        table.innerHTML = `<div class="empty">Split manifest has not been written yet.</div>`;
        return;
      }
      table.innerHTML = `
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Split</th>
                <th>Samples</th>
                <th>Environment counts</th>
                <th>Label counts</th>
              </tr>
            </thead>
            <tbody>
              ${splits.map(row => `
                <tr>
                  <td>${escapeHtml(row.name)}</td>
                  <td>${escapeHtml(row.count)}</td>
                  <td>${escapeHtml(compactMap(row.env_counts))}</td>
                  <td>${escapeHtml(compactMap(row.label_counts))}</td>
                </tr>
              `).join("")}
            </tbody>
          </table>
        </div>
      `;
    }

    function chartPath(points, xScale, yScale, key) {
      return points
        .filter(point => Number.isFinite(point[key]))
        .map((point, index) => `${index === 0 ? "M" : "L"} ${xScale(point.epoch).toFixed(1)} ${yScale(point[key]).toFixed(1)}`)
        .join(" ");
    }

    function renderTrainingChart(payload) {
      const history = payload.training_history || {};
      const svg = document.getElementById("historyChart");
      if (!svg) return;
      const points = history.points || [];
      setText("historyRun", history.run_label || "No completed history yet");
      setText("historyNote", history.note || "metrics.json is written when each run finishes");
      if (!points.length) {
        svg.innerHTML = `
          <rect x="0" y="0" width="760" height="260" fill="transparent"></rect>
          <text x="380" y="124" text-anchor="middle" class="chart-text">No epoch metrics on disk yet</text>
          <text x="380" y="146" text-anchor="middle" class="chart-text">The active trainer writes history after a run finishes</text>
        `;
        return;
      }

      const width = 760;
      const height = 260;
      const pad = { left: 48, right: 18, top: 18, bottom: 34 };
      const maxEpoch = Math.max(...points.map(point => point.epoch), 1);
      const xScale = epoch => pad.left + (epoch - 1) / Math.max(1, maxEpoch - 1) * (width - pad.left - pad.right);
      const yScale = value => pad.top + (1 - Math.max(0, Math.min(1, value))) * (height - pad.top - pad.bottom);
      const series = [
        { key: "val_accuracy", color: "#f4f4ef" },
        { key: "val_f1_macro", color: "#9fbd9a" },
        { key: "val_recall_macro", color: "#cfc06d" },
      ];
      const yTicks = [0, 0.25, 0.5, 0.75, 1];
      const xTicks = Array.from(new Set([1, Math.ceil(maxEpoch * 0.25), Math.ceil(maxEpoch * 0.5), Math.ceil(maxEpoch * 0.75), maxEpoch]));
      svg.innerHTML = `
        ${yTicks.map(value => `
          <line class="chart-grid" x1="${pad.left}" x2="${width - pad.right}" y1="${yScale(value).toFixed(1)}" y2="${yScale(value).toFixed(1)}"></line>
          <text class="chart-axis" x="10" y="${(yScale(value) + 4).toFixed(1)}">${Math.round(value * 100)}%</text>
        `).join("")}
        ${xTicks.map(epoch => `
          <line class="chart-grid" x1="${xScale(epoch).toFixed(1)}" x2="${xScale(epoch).toFixed(1)}" y1="${pad.top}" y2="${height - pad.bottom}"></line>
          <text class="chart-axis" text-anchor="middle" x="${xScale(epoch).toFixed(1)}" y="${height - 10}">E${epoch}</text>
        `).join("")}
        ${series.map(item => `<path class="chart-line" d="${chartPath(points, xScale, yScale, item.key)}" stroke="${item.color}"></path>`).join("")}
        ${series.map(item => {
          const valid = points.filter(point => Number.isFinite(point[item.key]));
          const last = valid[valid.length - 1];
          return last ? `<circle class="chart-dot" cx="${xScale(last.epoch).toFixed(1)}" cy="${yScale(last[item.key]).toFixed(1)}" r="3.5" fill="${item.color}"></circle>` : "";
        }).join("")}
      `;
    }

    function renderExperiments(payload) {
      const experiments = payload.experiments || [];
      if (!experiments.length) {
        document.getElementById("experiments").innerHTML = `<div class="empty">No configs found.</div>`;
        return;
      }
      document.getElementById("experiments").innerHTML = experiments.map(exp => {
      const pct = Number.isFinite(exp.realtime_percent)
        ? exp.realtime_percent
        : (exp.total_runs ? exp.completed_runs / exp.total_runs * 100 : 0);
        const metric = exp.mean_accuracy === null ? "--" : fmtNumber(exp.mean_accuracy);
        return `
          <div class="experiment-row">
            <div>
              <strong><span class="ui-icon ui-config"></span>${exp.name}</strong>
              <div class="metric-note">${exp.status}; ${fmtPercent(pct)} complete</div>
            </div>
            <div><span class="badge ${exp.status}">${exp.status}</span></div>
            <div class="run-number">${exp.completed_runs}/${exp.total_runs} runs</div>
            <div class="run-number">acc ${metric}</div>
          </div>
        `;
      }).join("");
    }

    function renderMetricSummary(payload) {
      const summary = payload.metric_summary || {};
      setText("metricAccuracy", summary.mean_accuracy === null ? "--" : fmtNumber(summary.mean_accuracy));
      setText("metricAccuracyNote", `${summary.completed_metric_runs || 0} completed metric run(s)`);
      setText("metricF1", summary.mean_f1_macro === null ? "--" : fmtNumber(summary.mean_f1_macro));
      const precision = summary.mean_precision_macro === null ? "--" : fmtNumber(summary.mean_precision_macro);
      const recall = summary.mean_recall_macro === null ? "--" : fmtNumber(summary.mean_recall_macro);
      setText("metricRecall", recall);
      setText("metricPrecision", precision);
      if (summary.best_run) {
        setText("metricBest", fmtNumber(summary.best_run.accuracy));
        setText("metricBestNote", `${summary.best_run.experiment} target ${summary.best_run.target_env}`);
      } else {
        setText("metricBest", "--");
        setText("metricBestNote", "No metrics yet");
      }

      const rows = payload.classwise_summary || [];
      if (!rows.length) {
        document.getElementById("classMetrics").innerHTML = `<div class="empty">Class-wise metrics will appear after the first confusion matrix is written.</div>`;
        return;
      }
      document.getElementById("classMetrics").innerHTML = `
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Class</th>
                <th>Support</th>
                <th>Class accuracy</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1</th>
              </tr>
            </thead>
            <tbody>
              ${rows.map(row => `
                <tr>
                  <td>${row.class_id}</td>
                  <td>${row.support}</td>
                  <td>${fmtNumber(row.accuracy)}</td>
                  <td>${fmtNumber(row.precision)}</td>
                  <td>${fmtNumber(row.recall)}</td>
                  <td>${fmtNumber(row.f1)}</td>
                </tr>
              `).join("")}
            </tbody>
          </table>
        </div>
      `;
    }

    function renderRuns(payload) {
      const runs = payload.runs;
      if (!runs.length) {
        document.getElementById("runs").innerHTML = `<div class="empty">No runs found.</div>`;
        return;
      }
      document.getElementById("runs").innerHTML = runs.map(run => {
        const accuracy = fmtNumber(run.accuracy);
        const recall = fmtNumber(run.recall_macro);
        const f1 = fmtNumber(run.f1_macro);
        const updated = run.updated_age_seconds === null ? "not written" : fmtAge(run.updated_age_seconds);
        return `
          <div class="run-row">
            <div><span class="badge ${run.status}">${run.status}</span></div>
            <div class="run-main">
              <strong><span class="ui-icon ui-target"></span>${run.experiment}</strong>
              <div class="metric-note">target ${run.target_env}; seed ${run.seed}</div>
            </div>
            <div class="run-main">
              <strong>${run.stage_label || run.status}</strong>
              <div class="metric-note">seed ${run.seed}; ${updated}</div>
            </div>
            <div class="run-number">accuracy ${accuracy}</div>
            <div class="run-number">recall ${recall}</div>
            <div class="run-number">f1 ${f1}</div>
          </div>
        `;
      }).join("");
    }

    function render(payload) {
      state.lastPayload = payload;
      setText("experimentName", payload.experiment_name);
      setText("pathsLine", `${payload.mode} - ${payload.output_root}`);

      const completed = payload.completed_runs;
      const total = payload.total_runs;
      const progress = payload.progress || {};
      const runPct = Number.isFinite(progress.percent)
        ? progress.percent
        : (total ? completed / total * 100 : 0);
      setText("completedValue", `${completed}/${total}`);
      setText("completedNote", total ? `${fmtPercent(runPct)} realtime; ${completed}/${total} finished` : "No target runs");
      setText("runPercent", `${fmtPercent(runPct)} realtime`);
      setText("trajectoryNote", total ? `${completed}/${total} packets; ${progress.calculation || "telemetry active"}` : "no target packets");
      setText("navRuns", `${completed}/${total}`);
      setText("navMode", payload.mode || "--");
      setGauge("suiteGauge", runPct);

      const active = payload.active_run;
      setText("activeTarget", active ? active.target_env : "--");
      setText("activeSeed", active ? `${active.experiment}; seed ${active.seed}; ${active.status}` : "waiting");
      setText("navTarget", active ? active.target_env : "--");
      setText("navSeed", active ? active.seed : "--");

      const live = payload.live_loader;
      if (live) {
        const pct = live.total ? live.current / live.total * 100 : live.percent;
        const throughput = live.rate || "--";
        const eta = live.remaining ? `ETA ${live.remaining}` : "ETA --";
        const epochLabel = live.epoch ? `${live.epoch}/${payload.config.epochs || "--"}` : "--";
        setText("stageValue", live.stage || "loader");
        setText("epochValue", live.epoch ? `epoch ${epochLabel}` : "epoch unknown");
        setText("batchStage", live.stage || "loader");
        setText("batchEpoch", epochLabel);
        setText("batchCount", `${live.current}/${live.total}`);
        setText("batchEta", live.remaining || "--");
        setText("throughputMiniValue", throughput);
        setText("throughputMiniNote", eta);
        setText("loaderPercent", fmtPercent(pct));
        setText("loaderDetail", `${live.current}/${live.total} batches`);
        setText("loaderBatch", `${live.current}/${live.total}`);
        setText("elapsedValue", live.elapsed || "--");
        setGauge("batchGauge", pct);
        setText("latestLine", live.raw_line || "No progress line yet");
      } else {
        setText("stageValue", "--");
        setText("epochValue", "epoch unknown");
        setText("batchStage", "--");
        setText("batchEpoch", "--");
        setText("batchCount", "--");
        setText("batchEta", "--");
        setText("throughputMiniValue", "--");
        setText("throughputMiniNote", "ETA --");
        setText("loaderPercent", "No signal");
        setText("loaderDetail", "awaiting stream");
        setText("loaderBatch", "--");
        setText("elapsedValue", "--");
        setGauge("batchGauge", 0);
        setText("latestLine", payload.log.latest_line || "Waiting for log output...");
      }

      setText("lastUpdate", fmtAge(payload.log.age_seconds));
      const freshDot = document.getElementById("freshDot");
      freshDot.classList.remove("fresh", "stale");
      if (payload.log.exists && payload.log.age_seconds !== null && payload.log.age_seconds < 30) {
        freshDot.classList.add("fresh");
        setText("freshText", `Log fresh - ${fmtAge(payload.log.age_seconds)}`);
      } else if (payload.log.exists) {
        freshDot.classList.add("stale");
        setText("freshText", `Log stale - ${fmtAge(payload.log.age_seconds)}`);
      } else {
        setText("freshText", "Log not found");
      }

      renderMetricSummary(payload);
      renderDataSplit(payload);
      renderTrainingChart(payload);
      renderExperiments(payload);
      renderRuns(payload);
    }

    async function refresh() {
      try {
        const response = await fetch("/api/status", { cache: "no-store" });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const payload = await response.json();
        render(payload);
      } catch (error) {
        setText("freshText", `Dashboard error - ${error.message}`);
        document.getElementById("freshDot").classList.add("stale");
      }
    }

    refresh();
    setInterval(refresh, 2000);
  </script>
</body>
</html>
"""


TIMING_RE = re.compile(r"(?P<elapsed>[^<]+)<(?P<remaining>[^,]+),\s*(?P<rate>.+)")
QUEUE_CONFIG_RE = re.compile(r'"(?P<path>configs/[^"]+\.json)"', re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve a read-only browser dashboard for PA-CSI DG experiment progress."
    )
    parser.add_argument("--config", help="Path to one JSON config used by the experiment.")
    parser.add_argument(
        "--queue-script",
        action="append",
        default=[],
        help="PowerShell queue script containing config paths. Can be passed more than once.",
    )
    parser.add_argument(
        "--no-existing-scan",
        action="store_true",
        help="Do not add configs discovered from existing output/log folders.",
    )
    parser.add_argument("--output-root", help="Override the output root if the run used --output-root.")
    parser.add_argument("--mode", help="Override the mode if the run used --mode.")
    parser.add_argument("--target-env", dest="target_env", help="Override target env if the run used --target-env.")
    parser.add_argument("--seeds", nargs="+", type=int, help="Override seed list if the run used --seeds.")
    parser.add_argument("--log-file", help="Optional stderr log to read for live batch progress.")
    parser.add_argument("--no-log", action="store_true", help="Disable live stderr-log parsing.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface for the dashboard.")
    parser.add_argument("--port", type=int, default=8765, help="Preferred dashboard port.")
    parser.add_argument("--open", action="store_true", help="Open the dashboard in the default browser.")
    return parser.parse_args()


def format_time(timestamp: Optional[float]) -> Optional[str]:
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


def age_seconds(timestamp: Optional[float]) -> Optional[float]:
    if timestamp is None:
        return None
    return max(0.0, time.time() - timestamp)


def read_json_file(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def classify_stage(total_batches: int, manifest: Optional[Mapping[str, Any]], batch_size: int) -> str:
    if not manifest:
        return "loader"
    stage_counts = {
        "train": manifest.get("train", {}).get("count"),
        "validation": manifest.get("val", {}).get("count"),
        "test": manifest.get("test", {}).get("count"),
    }
    for stage, count in stage_counts.items():
        if count is None:
            continue
        expected_batches = math.ceil(int(count) / max(1, batch_size))
        if expected_batches == total_batches:
            return stage
    return "loader"


def parse_timing(timing: str) -> Dict[str, Optional[str]]:
    match = TIMING_RE.search(timing)
    if not match:
        return {"elapsed": None, "remaining": None, "rate": timing}
    return {
        "elapsed": match.group("elapsed").strip(),
        "remaining": match.group("remaining").strip(),
        "rate": match.group("rate").strip(),
    }


def parse_tqdm_segments(log_path: Optional[Path]) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    if log_path is None or not log_path.exists():
        return None, []

    try:
        text = monitor.read_log_tail(log_path, max_bytes=4_000_000)
    except OSError:
        return None, []

    segments: List[Dict[str, Any]] = []
    latest: Optional[Dict[str, Any]] = None
    previous_current: Optional[int] = None
    previous_total: Optional[int] = None

    for raw_line in text.replace("\r", "\n").splitlines():
        line = monitor.ANSI_RE.sub("", raw_line).strip()
        match = monitor.TQDM_RE.search(line)
        if not match:
            continue

        current = int(match.group("current"))
        total = int(match.group("total"))
        percent = int(match.group("percent"))
        timing = match.group("timing").strip()

        if (
            not segments
            or previous_total != total
            or (previous_current is not None and current < previous_current)
        ):
            segments.append(
                {
                    "total": total,
                    "max_current": current,
                    "last_current": current,
                    "line_count": 1,
                }
            )
        else:
            segments[-1]["max_current"] = max(int(segments[-1]["max_current"]), current)
            segments[-1]["last_current"] = current
            segments[-1]["line_count"] = int(segments[-1]["line_count"]) + 1

        latest = {
            "current": current,
            "total": total,
            "percent": percent,
            "timing": timing,
            "raw_line": line,
        }
        previous_current = current
        previous_total = total

    return latest, segments


def parse_queue_config_paths(script_path: Path) -> List[Path]:
    try:
        text = script_path.read_text(encoding="utf-8")
    except OSError:
        return []
    return [monitor.resolve_path(match.group("path")) for match in QUEUE_CONFIG_RE.finditer(text)]


def mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def read_confusion_matrix(path: Path) -> Optional[List[List[int]]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = [[int(float(value)) for value in row] for row in csv.reader(handle) if row]
    except (OSError, ValueError):
        return None
    if not rows or any(len(row) != len(rows[0]) for row in rows):
        return None
    return rows


def add_confusion_matrices(matrices: List[List[List[int]]]) -> Optional[List[List[int]]]:
    if not matrices:
        return None
    size = max(len(matrix) for matrix in matrices)
    total = [[0 for _ in range(size)] for _ in range(size)]
    for matrix in matrices:
        for row_index, row in enumerate(matrix):
            for col_index, value in enumerate(row):
                total[row_index][col_index] += int(value)
    return total


def classwise_from_confusion(matrix: Optional[List[List[int]]]) -> List[Dict[str, Any]]:
    if matrix is None:
        return []
    size = len(matrix)
    col_sums = [sum(matrix[row][col] for row in range(size)) for col in range(size)]
    rows: List[Dict[str, Any]] = []
    for class_id, row in enumerate(matrix):
        support = sum(row)
        correct = row[class_id] if class_id < len(row) else 0
        predicted = col_sums[class_id] if class_id < len(col_sums) else 0
        recall = correct / support if support else 0.0
        precision = correct / predicted if predicted else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "class_id": class_id,
                "support": support,
                "accuracy": recall,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return rows


def expected_run_work(config: Mapping[str, Any], manifest: Optional[Mapping[str, Any]]) -> Dict[str, int]:
    training_config = config.get("training", {})
    batch_size = max(1, int(training_config.get("batch_size", 1)))
    epochs = max(1, int(training_config.get("epochs", 1)))

    def batches(section: str) -> int:
        if not manifest:
            return 0
        count = manifest.get(section, {}).get("count", 0)
        return int(math.ceil(int(count) / batch_size)) if count else 0

    train_batches = batches("train")
    val_batches = batches("val")
    test_batches = batches("test")
    epoch_batches = train_batches + val_batches
    return {
        "epochs": epochs,
        "train_batches": train_batches,
        "val_batches": val_batches,
        "test_batches": test_batches,
        "epoch_batches": epoch_batches,
        "run_batches": epochs * epoch_batches + test_batches,
        "segments_per_run": epochs * 2 + (1 if test_batches else 0),
    }


def logged_batch_work(segments: List[Mapping[str, Any]]) -> int:
    total = 0
    for segment in segments:
        total += min(int(segment.get("max_current", 0)), int(segment.get("total", 0)))
    return total


def run_metrics_from_files(path: Path, state: Mapping[str, Any]) -> Dict[str, Any]:
    metrics = read_json_file(path / "metrics.json") or {}
    test_metrics = metrics.get("test_metrics", {}) if isinstance(metrics, dict) else {}
    confusion = read_confusion_matrix(path / "confusion_matrix.csv")
    history = metrics.get("history", []) if isinstance(metrics, dict) else []
    if not isinstance(history, list):
        history = []
    return {
        "accuracy": test_metrics.get("accuracy", state.get("accuracy")),
        "precision_macro": test_metrics.get("precision_macro"),
        "recall_macro": test_metrics.get("recall_macro"),
        "f1_macro": test_metrics.get("f1_macro", state.get("f1_macro")),
        "best_val_accuracy": metrics.get("best_val_accuracy") if isinstance(metrics, dict) else None,
        "best_val_score": metrics.get("best_val_score") if isinstance(metrics, dict) else None,
        "history_len": len(history),
        "confusion": confusion,
        "classwise": classwise_from_confusion(confusion),
    }


class DashboardState:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.config_paths = self.collect_config_paths(args)
        if not self.config_paths:
            raise ValueError("Provide --config or --queue-script so the dashboard can find experiment configs.")
        self.experiments = [self.build_experiment(path) for path in self.config_paths]
        self.experiment_name = "Ablation Progress" if len(self.experiments) > 1 else self.experiments[0]["name"]
        self.mode = self.common_value("mode") or "mixed"
        self.output_root = self.common_value("output_root") or "multiple output roots"

    def collect_config_paths(self, args: argparse.Namespace) -> List[Path]:
        paths: List[Path] = []
        if args.config:
            paths.append(monitor.resolve_path(args.config))
        for script in args.queue_script:
            paths.extend(parse_queue_config_paths(monitor.resolve_path(script)))

        roots: List[Path] = []
        for path in paths:
            if not path.exists():
                continue
            config = monitor.load_config(path)
            root = monitor.resolve_path(args.output_root or str(config.get("output_root", "outputs")))
            roots.append(root)
        if args.output_root:
            roots.append(monitor.resolve_path(args.output_root))

        if not args.no_existing_scan:
            for root in list(dict.fromkeys(roots)):
                mode_names = [args.mode] if args.mode else ["dg_loeo", "random_split"]
                for mode_name in mode_names:
                    if not mode_name:
                        continue
                    mode_dir = root / mode_name
                    if mode_dir.exists():
                        for experiment_dir in mode_dir.iterdir():
                            config_path = REPO_ROOT / "configs" / f"{experiment_dir.name}.json"
                            if config_path.exists():
                                paths.append(config_path)
                log_dir = root / "run_logs"
                if log_dir.exists():
                    for log_path in log_dir.glob("*.stderr.log"):
                        config_path = REPO_ROOT / "configs" / f"{log_path.stem.replace('.stderr', '')}.json"
                        if config_path.exists():
                            paths.append(config_path)

        unique: List[Path] = []
        seen = set()
        for path in paths:
            resolved = path.resolve()
            if resolved in seen or not resolved.exists():
                continue
            seen.add(resolved)
            unique.append(resolved)
        return unique

    def build_experiment(self, config_path: Path) -> Dict[str, Any]:
        config = monitor.load_config(config_path)
        local_args = argparse.Namespace(**vars(self.args))
        local_args.config = str(config_path)
        output_root, mode, experiment_name, specs = monitor.build_run_specs(config, local_args)
        log_path = monitor.resolve_log_path(local_args, output_root, experiment_name)
        return {
            "config_path": config_path,
            "config": config,
            "output_root": output_root,
            "mode": mode,
            "name": experiment_name,
            "specs": specs,
            "log_path": log_path,
        }

    def common_value(self, key: str) -> Optional[str]:
        values = {str(experiment[key]) for experiment in self.experiments}
        if len(values) == 1:
            return next(iter(values))
        return None

    def run_path(self, experiment: Mapping[str, Any], target_env: str, seed: int) -> Path:
        return monitor.run_dir(
            Path(experiment["output_root"]),
            str(experiment["mode"]),
            str(experiment["name"]),
            target_env,
            seed,
        )

    def read_runs(self, experiment: Mapping[str, Any]) -> List[Dict[str, Any]]:
        runs: List[Dict[str, Any]] = []
        for target_env, seed in experiment["specs"]:
            path = self.run_path(experiment, target_env, seed)
            state = monitor.inspect_run(path)
            updated_at = state.get("updated_at")
            metrics = run_metrics_from_files(path, state)
            status = state["status"]
            stage_label = {
                "done": "metrics ready",
                "active": "checkpointing",
                "started": "split written",
                "created": "folder ready",
                "writing": "metrics writing",
                "pending": "not started",
            }.get(status, status)
            runs.append(
                {
                    "experiment": experiment["name"],
                    "target_env": target_env,
                    "seed": seed,
                    "status": status,
                    "stage_label": stage_label,
                    "accuracy": metrics["accuracy"],
                    "precision_macro": metrics["precision_macro"],
                    "recall_macro": metrics["recall_macro"],
                    "f1_macro": metrics["f1_macro"],
                    "best_val_accuracy": metrics["best_val_accuracy"],
                    "best_val_score": metrics["best_val_score"],
                    "history_len": metrics["history_len"],
                    "classwise": metrics["classwise"],
                    "confusion": metrics["confusion"],
                    "updated_at": format_time(updated_at),
                    "updated_age_seconds": age_seconds(updated_at),
                    "path": str(path),
                }
            )
        return runs

    def active_run(self, runs: List[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
        candidates = [run for run in runs if run["status"] in {"created", "started", "active", "writing"}]
        if not candidates:
            return None
        return min(candidates, key=lambda run: run.get("updated_age_seconds") if run.get("updated_age_seconds") is not None else float("inf"))

    def active_manifest(self, active: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
        if not active:
            return None
        return read_json_file(Path(str(active["path"])) / "split_manifest.json")

    def first_manifest(self, runs: List[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
        for run in runs:
            manifest = read_json_file(Path(str(run["path"])) / "split_manifest.json")
            if manifest:
                return manifest
        return None

    def log_status(self, experiment: Mapping[str, Any], latest: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        log_path = experiment["log_path"]
        exists = log_path is not None and log_path.exists()
        updated_at = log_path.stat().st_mtime if exists and log_path is not None else None
        return {
            "exists": exists,
            "path": str(log_path) if log_path else None,
            "size": log_path.stat().st_size if exists and log_path is not None else 0,
            "updated_at": format_time(updated_at),
            "age_seconds": age_seconds(updated_at),
            "latest_line": latest.get("raw_line") if latest else None,
        }

    def live_loader(
        self,
        experiment: Mapping[str, Any],
        latest: Optional[Mapping[str, Any]],
        segments: List[Mapping[str, Any]],
        manifest: Optional[Mapping[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if latest is None:
            return None

        training_config = experiment["config"].get("training", {})
        batch_size = int(training_config.get("batch_size", 1))
        total = int(latest["total"])
        stage = classify_stage(total, manifest, batch_size)
        segment_index = len(segments)
        stage_occurrence = sum(1 for segment in segments if int(segment["total"]) == total)
        timing = parse_timing(str(latest["timing"]))

        epoch: Optional[int] = None
        if stage in {"train", "validation"}:
            epoch = max(1, stage_occurrence)

        return {
            "current": int(latest["current"]),
            "total": total,
            "percent": int(latest["percent"]),
            "timing": latest["timing"],
            "elapsed": timing["elapsed"],
            "remaining": timing["remaining"],
            "rate": timing["rate"],
            "raw_line": latest["raw_line"],
            "stage": stage,
            "epoch": epoch,
            "segment_index": segment_index,
            "experiment": experiment["name"],
        }

    def realtime_experiment_progress(
        self,
        experiment: Mapping[str, Any],
        runs: List[Mapping[str, Any]],
        segments: List[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        total_runs = len(runs)
        completed_runs = sum(1 for run in runs if run["status"] == "done")
        if total_runs == 0:
            return {
                "realtime_runs": 0.0,
                "percent": 0.0,
                "active_run_fraction": 0.0,
                "calculation": "no runs",
            }
        if completed_runs >= total_runs:
            return {
                "realtime_runs": float(total_runs),
                "percent": 100.0,
                "active_run_fraction": 1.0,
                "calculation": "all runs completed",
            }

        manifest = self.active_manifest(self.active_run(runs)) or self.first_manifest(runs)
        work = expected_run_work(experiment["config"], manifest)
        run_batches = int(work["run_batches"])
        if not segments or run_batches <= 0:
            percent = completed_runs / total_runs * 100.0
            return {
                "realtime_runs": float(completed_runs),
                "percent": percent,
                "active_run_fraction": 0.0,
                "calculation": "waiting for live log",
            }

        logged_work = logged_batch_work(segments)
        active_work = max(0, logged_work - completed_runs * run_batches)
        active_fraction = min(1.0, active_work / run_batches)
        realtime_runs = min(float(total_runs), completed_runs + active_fraction)
        return {
            "realtime_runs": realtime_runs,
            "percent": realtime_runs / total_runs * 100.0,
            "active_run_fraction": active_fraction,
            "calculation": (
                f"{completed_runs} finished + {active_work}/{run_batches} "
                "batches in active run"
            ),
            "work": work,
        }

    def experiment_payload(self, experiment: Mapping[str, Any]) -> Dict[str, Any]:
        runs = self.read_runs(experiment)
        completed = sum(1 for run in runs if run["status"] == "done")
        latest, segments = parse_tqdm_segments(experiment["log_path"])
        log = self.log_status(experiment, latest)
        metric_runs = [run for run in runs if run.get("accuracy") is not None]
        if completed == len(runs) and runs:
            status = "done"
        elif log["exists"] and log["age_seconds"] is not None and log["age_seconds"] < 120:
            status = "active"
        elif any(run["status"] != "pending" for run in runs):
            status = "started"
        else:
            status = "pending"
        progress = self.realtime_experiment_progress(experiment, runs, segments)
        return {
            "name": experiment["name"],
            "mode": experiment["mode"],
            "output_root": str(experiment["output_root"]),
            "config_path": str(experiment["config_path"]),
            "total_runs": len(runs),
            "completed_runs": completed,
            "realtime_completed_runs": progress["realtime_runs"],
            "realtime_percent": progress["percent"],
            "progress_calculation": progress["calculation"],
            "status": status,
            "mean_accuracy": mean([float(run["accuracy"]) for run in metric_runs]),
            "runs": runs,
            "latest": latest,
            "segments": segments,
            "log": log,
            "experiment": experiment,
        }

    def metric_summary(self, runs: List[Mapping[str, Any]]) -> Dict[str, Any]:
        metric_runs = [run for run in runs if run.get("accuracy") is not None]
        best_run = max(metric_runs, key=lambda run: float(run["accuracy"]), default=None)
        return {
            "completed_metric_runs": len(metric_runs),
            "mean_accuracy": mean([float(run["accuracy"]) for run in metric_runs if run.get("accuracy") is not None]),
            "mean_f1_macro": mean([float(run["f1_macro"]) for run in metric_runs if run.get("f1_macro") is not None]),
            "mean_precision_macro": mean(
                [float(run["precision_macro"]) for run in metric_runs if run.get("precision_macro") is not None]
            ),
            "mean_recall_macro": mean(
                [float(run["recall_macro"]) for run in metric_runs if run.get("recall_macro") is not None]
            ),
            "best_run": {
                "experiment": best_run["experiment"],
                "target_env": best_run["target_env"],
                "seed": best_run["seed"],
                "accuracy": best_run["accuracy"],
                "f1_macro": best_run["f1_macro"],
            }
            if best_run
            else None,
        }

    def classwise_summary(self, runs: List[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        matrices = [run["confusion"] for run in runs if run.get("confusion") is not None]
        return classwise_from_confusion(add_confusion_matrices(matrices))

    def experiment_for_run(self, run: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
        if run:
            for experiment in self.experiments:
                if str(experiment["name"]) == str(run.get("experiment")):
                    return experiment
        return self.experiments[0]

    def dataset_split_info(
        self,
        active: Optional[Mapping[str, Any]],
        runs: List[Mapping[str, Any]],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        experiment = self.experiment_for_run(active)
        config = experiment["config"]
        data_config = config.get("data", {})
        env_files = data_config.get("env_files", {})
        envs = sorted(str(env) for env in env_files.keys()) if isinstance(env_files, dict) else []
        first_env_files = next(iter(env_files.values()), {}) if isinstance(env_files, dict) and env_files else {}
        modalities = sorted(str(key) for key in first_env_files.keys()) if isinstance(first_env_files, dict) else []
        manifest = self.active_manifest(active) or self.first_manifest(runs)

        label_ids = set()
        splits: List[Dict[str, Any]] = []
        if manifest:
            for key, label in (("train", "Train"), ("val", "Validation"), ("test", "Test")):
                section = manifest.get(key, {})
                label_counts = section.get("label_counts", {}) if isinstance(section, dict) else {}
                label_ids.update(str(class_id) for class_id in label_counts.keys())
                splits.append(
                    {
                        "name": label,
                        "count": section.get("count", 0) if isinstance(section, dict) else 0,
                        "env_counts": section.get("env_counts", {}) if isinstance(section, dict) else {},
                        "label_counts": label_counts,
                    }
                )

        work = expected_run_work(config, manifest) if manifest else {}
        dataset_info = {
            "format": data_config.get("format", "unknown"),
            "env_count": len(envs),
            "envs": envs,
            "modalities": modalities,
            "unwrap_phase": bool(data_config.get("unwrap_phase", False)),
            "dataset_size": manifest.get("dataset_size") if manifest else None,
            "class_count": len(label_ids) if label_ids else None,
        }
        split_info = {
            "mode": manifest.get("mode", experiment["mode"]) if manifest else experiment["mode"],
            "target_env": manifest.get("target_env") if manifest else active.get("target_env") if active else None,
            "source_envs": manifest.get("source_envs", []) if manifest else [],
            "train_count": splits[0]["count"] if len(splits) > 0 else None,
            "val_count": splits[1]["count"] if len(splits) > 1 else None,
            "test_count": splits[2]["count"] if len(splits) > 2 else None,
            "splits": splits,
            "work": work,
        }
        return dataset_info, split_info

    def training_history(self, runs: List[Mapping[str, Any]]) -> Dict[str, Any]:
        candidates: List[Tuple[float, Mapping[str, Any], Mapping[str, Any]]] = []
        for run in runs:
            metrics_path = Path(str(run["path"])) / "metrics.json"
            metrics = read_json_file(metrics_path)
            if not metrics:
                continue
            history = metrics.get("history", [])
            if not isinstance(history, list) or not history:
                continue
            try:
                updated_at = metrics_path.stat().st_mtime
            except OSError:
                updated_at = 0.0
            candidates.append((updated_at, run, metrics))

        if not candidates:
            return {
                "run_label": "No completed history yet",
                "note": "metrics.json is written when each run finishes",
                "points": [],
            }

        _, run, metrics = max(candidates, key=lambda item: item[0])
        points: List[Dict[str, Any]] = []
        for record in metrics.get("history", []):
            if not isinstance(record, dict):
                continue
            train = record.get("train", {}) if isinstance(record.get("train", {}), dict) else {}
            val = record.get("val", {}) if isinstance(record.get("val", {}), dict) else {}

            def maybe_float(section: Mapping[str, Any], key: str) -> Optional[float]:
                value = section.get(key)
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None

            points.append(
                {
                    "epoch": int(record.get("epoch", len(points) + 1)),
                    "train_accuracy": maybe_float(train, "accuracy"),
                    "val_accuracy": maybe_float(val, "accuracy"),
                    "val_f1_macro": maybe_float(val, "f1_macro"),
                    "val_recall_macro": maybe_float(val, "recall_macro"),
                    "val_precision_macro": maybe_float(val, "precision_macro"),
                    "train_loss": maybe_float(train, "loss_total"),
                    "val_loss": maybe_float(val, "loss_total"),
                }
            )

        best_val_accuracy = metrics.get("best_val_accuracy")
        note = f"{len(points)} epochs"
        if best_val_accuracy is not None:
            note += f"; best val acc {float(best_val_accuracy):.4f}"
        return {
            "run_label": f"{run['experiment']} target {run['target_env']} seed {run['seed']}",
            "note": note,
            "points": points,
        }

    def suite_progress(self, experiment_payloads: List[Mapping[str, Any]], total_runs: int) -> Dict[str, Any]:
        realtime_runs = sum(float(experiment.get("realtime_completed_runs", 0.0)) for experiment in experiment_payloads)
        percent = realtime_runs / total_runs * 100.0 if total_runs else 0.0
        active_calculations = [
            f"{experiment['name']}: {experiment.get('progress_calculation')}"
            for experiment in experiment_payloads
            if experiment.get("status") == "active"
        ]
        return {
            "realtime_runs": realtime_runs,
            "percent": percent,
            "calculation": "; ".join(active_calculations) if active_calculations else "no active log",
        }

    def payload(self) -> Dict[str, Any]:
        experiment_payloads = [self.experiment_payload(experiment) for experiment in self.experiments]
        runs = [run for experiment in experiment_payloads for run in experiment["runs"]]
        completed = sum(1 for run in runs if run["status"] == "done")
        progress = self.suite_progress(experiment_payloads, len(runs))

        active_experiment_payload = min(
            (
                experiment
                for experiment in experiment_payloads
                if experiment["log"]["exists"] and experiment["log"]["age_seconds"] is not None
            ),
            key=lambda experiment: experiment["log"]["age_seconds"],
            default=experiment_payloads[0],
        )
        active = self.active_run(runs)
        manifest = self.active_manifest(active)
        live_loader = self.live_loader(
            active_experiment_payload["experiment"],
            active_experiment_payload["latest"],
            active_experiment_payload["segments"],
            manifest,
        )
        dataset_info, split_info = self.dataset_split_info(active, runs)

        primary_config = self.experiments[0]["config"]
        training_config = primary_config.get("training", {})
        losses = primary_config.get("losses", {})
        enabled_losses = [
            name.replace("lambda_", "")
            for name in ("lambda_supcon",)
            if float(losses.get(name, 0.0)) > 0.0
        ]

        return {
            "experiment_name": self.experiment_name,
            "mode": self.mode,
            "output_root": str(self.output_root),
            "total_runs": len(runs),
            "completed_runs": completed,
            "progress": progress,
            "active_run": active,
            "runs": runs,
            "experiments": [
                {key: value for key, value in experiment.items() if key not in {"runs", "latest", "segments", "log", "experiment"}}
                for experiment in experiment_payloads
            ],
            "metric_summary": self.metric_summary(runs),
            "classwise_summary": self.classwise_summary(runs),
            "dataset_info": dataset_info,
            "split_info": split_info,
            "training_history": self.training_history(runs),
            "live_loader": live_loader,
            "log": active_experiment_payload["log"],
            "config": {
                "mode": self.mode,
                "config_count": len(self.experiments),
                "device": self.common_config_value("device") or "mixed",
                "epochs": int(training_config.get("epochs", 0)),
                "batch_size": int(training_config.get("batch_size", 0)),
                "selection_metric": training_config.get("selection_metric", "unknown"),
                "losses": ", ".join(enabled_losses) if enabled_losses else "varies by config",
            },
        }

    def common_config_value(self, key: str) -> Optional[str]:
        values = {str(experiment["config"].get(key, "unknown")) for experiment in self.experiments}
        if len(values) == 1:
            return next(iter(values))
        return None


def find_available_port(host: str, preferred_port: int) -> int:
    for port in range(preferred_port, preferred_port + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind((host, port))
            except OSError:
                continue
            return port
    raise OSError(f"No available port found from {preferred_port} to {preferred_port + 49}.")


def make_handler(state: DashboardState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def send_bytes(self, content: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/":
                self.send_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            if path == "/api/status":
                payload = json.dumps(state.payload()).encode("utf-8")
                self.send_bytes(payload, "application/json; charset=utf-8")
                return
            self.send_bytes(b"Not found", "text/plain; charset=utf-8", status=404)

    return Handler


def main() -> None:
    args = parse_args()
    state = DashboardState(args)
    port = find_available_port(args.host, args.port)
    server = ThreadingHTTPServer((args.host, port), make_handler(state))
    url = f"http://{args.host}:{port}/"

    print(f"Dashboard: {url}")
    print(f"Experiment: {state.experiment_name}")
    print(f"Configs: {len(state.experiments)}")
    if args.open:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
