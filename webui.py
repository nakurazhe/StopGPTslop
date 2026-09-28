#!/usr/bin/env python3
"""StopGPTslop — single-file web UI.

    python webui.py            # then open http://127.0.0.1:7860

Drop, browse or paste an image, adjust the strength, and drag the divider to compare
before and after. Language and theme both follow the operating system.

Notes:
- Uses only the standard library's http.server; images travel as base64 JSON, which
  avoids multipart parsing. No web framework needed.
- Changing the strength only re-runs the decoder: the encoder output is cached per
  image, which makes dragging the slider feel roughly four times more responsive.
- Model definitions come from modeling.py in this directory, so there is only ever
  one copy of them.
- The image pixels are never recoloured by the theme; only the surrounding workspace
  switches between a light neutral grey and the dark inspection background.
"""
import argparse
import base64
import hashlib
import io
import json
import os
import gc
import sys
import threading
import time
import webbrowser
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
TORPH_JS = os.path.join(HERE, "node_modules", "torph", "dist", "index.mjs")
sys.path.insert(0, HERE)
# Model definitions live in modeling.py so there is only one copy of them
from modeling import (ALIGN, build_fns, load_vae, load_refiner_model,
                      load_realesrgan_model)  # noqa: E402

STATE = {}
CACHE = {}
CACHE_ORDER = []
LOCK = threading.Lock()


def _autocast_context(device, dtype):
    if device.split(":")[0] == "cuda" and dtype in (torch.float16, torch.bfloat16):
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>StopGPTslop</title>
<style>
/* ---- tokens: light = warm paper, dark = darkroom (two ends of one temperature) ---- */
:root{
  --ground:#EDE7DA; --surface:#F7F3EA; --raised:#FFFDF8;
  --ink:#2B2A26; --ink-2:#5B564C; --muted:#8A8378; --line:#D8CFBD;
  --accent:#1F4E5F; --accent-ink:#FFFFFF; --accent-soft:#DCE7E9;
  --err-bg:#F4DCDC; --err-ink:#7C2222;
  --canvas:#1C1C1C;              /* viewing background, identical in both themes */
  --shadow:0 1px 2px rgba(43,42,38,.06), 0 8px 24px -12px rgba(43,42,38,.14);
}
@media (prefers-color-scheme:dark){
  :root{
    --ground:#211E1A; --surface:#2A2622; --raised:#332E28;
    --ink:#EAE4D7; --ink-2:#BDB5A6; --muted:#928A7C; --line:#3E3830;
    --accent:#6FB3AA; --accent-ink:#132220; --accent-soft:#25332F;
    --err-bg:#452020; --err-ink:#F6C9C9;
    --shadow:0 1px 2px rgba(0,0,0,.4), 0 10px 28px -14px rgba(0,0,0,.6);
  }
}
:root[data-theme="light"]{
  --ground:#EDE7DA; --surface:#F7F3EA; --raised:#FFFDF8;
  --ink:#2B2A26; --ink-2:#5B564C; --muted:#8A8378; --line:#D8CFBD;
  --accent:#1F4E5F; --accent-ink:#FFFFFF; --accent-soft:#DCE7E9;
  --err-bg:#F4DCDC; --err-ink:#7C2222;
  --shadow:0 1px 2px rgba(43,42,38,.06), 0 8px 24px -12px rgba(43,42,38,.14);
}
:root[data-theme="dark"]{
  --ground:#211E1A; --surface:#2A2622; --raised:#332E28;
  --ink:#EAE4D7; --ink-2:#BDB5A6; --muted:#928A7C; --line:#3E3830;
  --accent:#6FB3AA; --accent-ink:#132220; --accent-soft:#25332F;
  --err-bg:#452020; --err-ink:#F6C9C9;
  --shadow:0 1px 2px rgba(0,0,0,.4), 0 10px 28px -14px rgba(0,0,0,.6);
}
*{box-sizing:border-box}
html,body{height:100%;margin:0}
body{background:var(--ground);color:var(--ink);
  font:14px/1.55 system-ui,-apple-system,"Segoe UI","Noto Sans CJK SC","PingFang SC",sans-serif;
  -webkit-font-smoothing:antialiased}
.num{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
     font-variant-numeric:tabular-nums}
button{font:inherit;color:inherit;cursor:pointer;border:0;background:none}
button:focus-visible,input:focus-visible{outline:2px solid var(--accent);outline-offset:2px}

/* ---- shell ---- */
.app{display:flex;flex-direction:column;height:100dvh}
.top{display:flex;align-items:center;gap:12px;padding:0 16px;height:52px;flex:0 0 auto;
     background:var(--surface);border-bottom:1px solid var(--line)}
.mark{width:22px;height:22px;border-radius:6px;background:var(--accent);flex:0 0 auto;
      display:grid;place-items:center}
.mark i{display:block;width:9px;height:9px;border-radius:2px;
        background:linear-gradient(135deg,var(--accent-ink) 0 50%,transparent 50% 100%)}
.top h1{font-size:14px;font-weight:650;margin:0;letter-spacing:.01em}
.top .ver{color:var(--muted);font-size:11.5px;letter-spacing:.04em}
.grow{flex:1}
.icon{width:32px;height:32px;border-radius:8px;display:grid;place-items:center;
      color:var(--ink-2);font-size:12px;font-weight:600}
.icon:hover{background:var(--raised);color:var(--ink)}
.split{flex:1;display:grid;grid-template-columns:340px 1fr;min-height:0}

/* ---- left: controls ---- */
.panel{background:var(--surface);border-right:1px solid var(--line);overflow-y:auto;
       display:flex;flex-direction:column}
.grp{padding:16px;border-bottom:1px solid var(--line);display:flex;flex-direction:column;gap:10px}
.grp:last-child{border-bottom:0}
.eyebrow{font-size:10.5px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;
         color:var(--muted)}

#drop{border:1.5px dashed var(--line);border-radius:10px;background:var(--raised);
      padding:22px 14px;text-align:center;cursor:pointer;transition:border-color .15s,background .15s}
#drop:hover{border-color:var(--accent);background:var(--accent-soft)}
#drop .big{font-weight:600}
#drop .sm{color:var(--muted);font-size:12px;margin-top:4px}
kbd{font:inherit;font-size:11px;background:var(--surface);border:1px solid var(--line);
    border-radius:4px;padding:0 4px}
#thumb{display:none;gap:10px;align-items:center;padding:8px;border:1px solid var(--line);
       border-radius:10px;background:var(--raised);cursor:pointer}
#thumb.on{display:flex}
#thumb img{width:52px;height:52px;object-fit:cover;border-radius:6px;flex:0 0 auto;
           background:var(--canvas)}
#thumb .tn{font-size:12.5px;font-weight:550;overflow:hidden;text-overflow:ellipsis;
           white-space:nowrap}
#thumb .td{color:var(--muted);font-size:11.5px;margin-top:2px}

.aval{display:flex;align-items:baseline;gap:8px}
.aval b{font-size:26px;font-weight:600;letter-spacing:-.01em}
.aval span{color:var(--muted);font-size:12px}
input[type=range]{width:100%;accent-color:var(--accent);margin:0}
.presets{display:grid;grid-template-columns:repeat(2,1fr);gap:6px}
.presets button{border:1px solid var(--line);background:var(--raised);border-radius:8px;
                padding:7px 8px;font-size:12px;color:var(--ink-2);text-align:left;line-height:1.3}
.presets button:hover{border-color:var(--accent)}
.presets button.sel{background:var(--accent);border-color:var(--accent);color:var(--accent-ink)}
.presets b{display:block;font-size:12.5px;font-weight:650}
.hint{color:var(--muted);font-size:11.5px;margin:0}
.check{display:flex;align-items:flex-start;gap:9px;padding:9px 10px;border:1px solid var(--line);
       border-radius:8px;background:var(--raised);cursor:pointer;color:var(--ink-2)}
.check input{margin:3px 0 0;accent-color:var(--accent)}
.check b{display:block;color:var(--ink);font-size:12.5px}
.check small{display:block;color:var(--muted);font-size:11.5px;line-height:1.4;margin-top:2px}

.seg{display:flex;border:1px solid var(--line);border-radius:8px;overflow:hidden;
     background:var(--raised)}
.seg button{flex:1;padding:7px 0;font-size:12.5px;color:var(--ink-2)}
.seg button.sel{background:var(--accent);color:var(--accent-ink);font-weight:600}
.acts{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.acts .wide{grid-column:1/-1}
.pri{background:var(--accent);color:var(--accent-ink);border-radius:8px;padding:9px 14px;
     font-weight:600;font-size:13px}
.pri:disabled{opacity:.42;cursor:default}
.gho{border:1px solid var(--line);border-radius:8px;padding:9px 14px;font-size:13px;
     color:var(--ink-2);background:var(--raised)}
.gho:hover{color:var(--ink);border-color:var(--accent)}
.about{font-size:12px;color:var(--muted);line-height:1.6}
.about p{margin:0 0 8px}
.about p:last-child{margin:0}
.about code{font-family:ui-monospace,monospace;font-size:11.5px;color:var(--ink-2)}

/* ---- right: canvas ---- */
.stagewrap{position:relative;display:flex;flex-direction:column;min-height:0;
           background:var(--canvas)}
/* The whole window accepts drops; this overlay is only the visual cue. It must not
   capture pointer events, or the drop would never reach the document listener. */
#dropHint{position:absolute;inset:0;z-index:5;display:none;place-items:center;
  background:rgba(31,78,95,.34);   /* fallback for browsers without color-mix() */
  background:color-mix(in srgb,var(--accent) 34%,transparent);
  outline:2px dashed rgba(255,255,255,.72);outline-offset:-10px;
  color:#fff;font-size:15px;font-weight:600;letter-spacing:.02em;pointer-events:none}
body.dragging #dropHint{display:grid}
body.dragging #drop{border-color:var(--accent);background:var(--accent-soft)}
#wrap{flex:1;min-height:0;display:grid;place-items:center;padding:14px}
#wrap.fit{overflow:hidden}
#wrap.act{overflow:auto;place-items:start}
#cmp{position:relative;line-height:0;cursor:ew-resize;user-select:none;touch-action:none;
     box-shadow:0 6px 30px -8px rgba(0,0,0,.55)}
/* JS sizes #cmp so images only ever shrink, never enlarge: small images stay at 1:1,
   large ones scale down to fit. high-quality gives a better downscale; pixelated or
   crisp-edges would turn the very detail you are inspecting into jagged edges.
   pointer-events:none is required -- otherwise the browser starts its own image drag,
   which cancels the pointer sequence so pointerup never arrives and dragging sticks. */
#cmp img{display:block;width:100%;height:100%;pointer-events:none;
         image-rendering:high-quality;-webkit-user-drag:none;user-select:none}
#after{position:absolute;inset:0;clip-path:inset(0 0 0 50%);pointer-events:none}
#bar{position:absolute;top:0;bottom:0;left:50%;width:1px;background:rgba(255,255,255,.92);
     pointer-events:none;box-shadow:0 0 0 .5px rgba(0,0,0,.5)}
#bar::after{content:"";position:absolute;top:50%;left:50%;width:30px;height:30px;
     transform:translate(-50%,-50%);border-radius:50%;background:rgba(255,255,255,.95);
     box-shadow:0 2px 12px rgba(0,0,0,.55)}
#bar::before{content:"";position:absolute;top:50%;left:50%;width:9px;height:9px;
     transform:translate(-50%,-50%) rotate(45deg);border:1.5px solid var(--canvas);
     border-width:1.5px 0 0 1.5px;opacity:.75}
.tag{position:absolute;top:9px;padding:3px 8px;border-radius:5px;font-size:10.5px;
     letter-spacing:.06em;text-transform:uppercase;font-weight:650;
     background:rgba(0,0,0,.55);color:#fff;pointer-events:none;backdrop-filter:blur(6px)}
.tag.l{left:9px}.tag.r{right:9px}
#busy{position:absolute;inset:0;display:none;place-items:center;background:rgba(15,15,15,.55);
      color:#fff;font-size:12.5px;letter-spacing:.04em;backdrop-filter:blur(2px)}
#busy.on{display:grid}
.status{flex:0 0 auto;height:30px;display:flex;align-items:center;gap:14px;padding:0 14px;
        background:rgba(0,0,0,.34);color:rgba(255,255,255,.72);font-size:11.5px;
        border-top:1px solid rgba(255,255,255,.07)}
.empty{color:rgba(255,255,255,.4);font-size:13px;text-align:center;padding:40px}
#err{display:none;margin:0 14px 14px;padding:9px 12px;border-radius:8px;font-size:12.5px;
     background:var(--err-bg);color:var(--err-ink)}
#err.on{display:block}
@media (max-width:900px){
  .split{grid-template-columns:1fr;grid-template-rows:auto 1fr}
  .panel{border-right:0;border-bottom:1px solid var(--line)}
  #wrap{min-height:44vh}
}
/* ---- iOS-inspired interface: local CSS animations, no network dependency ---- */
:root{
  --ground:#f2f2f7;--surface:rgba(255,255,255,.76);--raised:#fff;
  --ink:#1c1c1e;--ink-2:#3a3a3c;--muted:#8e8e93;
  --line:rgba(60,60,67,.16);--accent:#007aff;--accent-ink:#fff;
  --accent-soft:rgba(0,122,255,.10);--err-bg:#fff0f0;--err-ink:#d70015;
  --canvas:#e9e9ee;--stage-status:rgba(255,255,255,.76);--stage-text:rgba(28,28,30,.62);
  --empty-text:rgba(28,28,30,.42);--shadow:0 1px 2px rgba(0,0,0,.04),0 12px 34px rgba(0,0,0,.08);
}
@media (prefers-color-scheme:dark){:root{
  --ground:#000;--surface:rgba(28,28,30,.78);--raised:#1c1c1e;
  --ink:#f5f5f7;--ink-2:#ebebf0;--muted:#98989d;
  --line:rgba(235,235,245,.14);--accent:#0a84ff;--accent-ink:#fff;
  --accent-soft:rgba(10,132,255,.16);--err-bg:#3b1418;--err-ink:#ff6961;
  --canvas:#0b0b0d;--stage-status:rgba(0,0,0,.42);--stage-text:rgba(255,255,255,.72);
  --empty-text:rgba(255,255,255,.46);--shadow:0 1px 2px rgba(0,0,0,.3),0 18px 44px rgba(0,0,0,.32);
}}
:root[data-theme="light"]{
  --ground:#f2f2f7;--surface:rgba(255,255,255,.76);--raised:#fff;
  --ink:#1c1c1e;--ink-2:#3a3a3c;--muted:#8e8e93;
  --line:rgba(60,60,67,.16);--accent:#007aff;--accent-ink:#fff;
  --accent-soft:rgba(0,122,255,.10);--err-bg:#fff0f0;--err-ink:#d70015;
  --canvas:#e9e9ee;--stage-status:rgba(255,255,255,.76);--stage-text:rgba(28,28,30,.62);
  --empty-text:rgba(28,28,30,.42);--shadow:0 1px 2px rgba(0,0,0,.04),0 12px 34px rgba(0,0,0,.08)
}
:root[data-theme="dark"]{
  --ground:#000;--surface:rgba(28,28,30,.78);--raised:#1c1c1e;
  --ink:#f5f5f7;--ink-2:#ebebf0;--muted:#98989d;
  --line:rgba(235,235,245,.14);--accent:#0a84ff;--accent-ink:#fff;
  --accent-soft:rgba(10,132,255,.16);--err-bg:#3b1418;--err-ink:#ff6961;
  --canvas:#0b0b0d;--stage-status:rgba(0,0,0,.42);--stage-text:rgba(255,255,255,.72);
  --empty-text:rgba(255,255,255,.46);--shadow:0 1px 2px rgba(0,0,0,.3),0 18px 44px rgba(0,0,0,.32)
}
html{background:var(--ground)}
body,button,input{font-family:Helvetica,Arial,sans-serif}
body{font-size:14px;line-height:1.45;background:
  radial-gradient(circle at 12% -10%,var(--accent-soft),transparent 28%),var(--ground)}
.num{font-family:Helvetica,Arial,sans-serif;font-variant-numeric:tabular-nums}
.app{padding:0 12px 12px}
.top{height:66px;padding:0 8px;background:transparent;border:0;gap:11px}
.mark{width:30px;height:30px;border-radius:9px;box-shadow:inset 0 1px 0 rgba(255,255,255,.28),0 5px 14px var(--accent-soft)}
.mark i{width:12px;height:12px;border-radius:3px}
.top h1{font-size:17px;font-weight:700;letter-spacing:-.022em}
.icon{width:36px;height:36px;border-radius:50%;background:var(--surface);border:1px solid var(--line);
  backdrop-filter:blur(20px) saturate(160%);-webkit-backdrop-filter:blur(20px) saturate(160%);
  transition:transform .35s cubic-bezier(.22,1,.36,1),background .2s,color .2s}
.icon:hover{background:var(--raised);transform:translateY(-1px)}
.icon:active{transform:scale(.92)}
.split{grid-template-columns:370px minmax(0,1fr);gap:12px;min-height:0}
.panel{background:transparent;border:0;padding-right:2px;scrollbar-width:thin;scrollbar-color:var(--line) transparent}
.grp{margin:0 0 10px;padding:17px;border:1px solid var(--line);border-radius:20px;
  background:var(--surface);box-shadow:var(--shadow);backdrop-filter:blur(28px) saturate(170%);
  -webkit-backdrop-filter:blur(28px) saturate(170%);gap:11px;
  animation:cardIn .6s cubic-bezier(.22,1,.36,1) both}
.grp:nth-child(2){animation-delay:.035s}.grp:nth-child(3){animation-delay:.07s}
.grp:nth-child(4){animation-delay:.105s}.grp:nth-child(5){animation-delay:.14s}
.grp:nth-child(6){animation-delay:.175s}
.eyebrow{font-size:12px;font-weight:700;letter-spacing:-.005em;text-transform:none;color:var(--ink)}
#drop{padding:24px 14px;border:1px dashed var(--line);border-radius:16px;background:var(--accent-soft);
  transition:transform .4s cubic-bezier(.22,1,.36,1),border-color .2s,background .2s}
#drop:hover{border-color:var(--accent);background:var(--accent-soft);transform:scale(1.012)}
#drop:active{transform:scale(.985)}
#drop .big{font-size:14px;font-weight:700}#drop .sm{font-size:12px;margin-top:5px}
kbd{border-radius:5px;padding:1px 5px;background:var(--raised);box-shadow:0 1px 2px rgba(0,0,0,.08)}
#thumb{border-radius:14px;padding:9px;background:var(--raised);transition:transform .25s}
#thumb:hover{transform:translateY(-1px)}#thumb img{border-radius:10px}
.aval b{font-size:30px;font-weight:700;letter-spacing:-.035em}.aval span{font-size:12px}
input[type=range]{height:24px;appearance:none;-webkit-appearance:none;background:transparent;cursor:pointer}
input[type=range]::-webkit-slider-runnable-track{height:5px;border-radius:99px;
  background:linear-gradient(90deg,var(--accent) 0 var(--fill,50%),rgba(118,118,128,.18) var(--fill,50%) 100%)}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:22px;height:22px;margin-top:-8.5px;border-radius:50%;
  background:#fff;border:.5px solid rgba(0,0,0,.10);box-shadow:0 2px 7px rgba(0,0,0,.26);transition:transform .18s}
input[type=range]:active::-webkit-slider-thumb{transform:scale(1.14)}
.presets{gap:7px}.presets button{border:0;border-radius:12px;padding:9px 10px;background:var(--accent-soft);
  color:var(--ink-2);text-align:center;transition:transform .3s cubic-bezier(.22,1,.36,1),background .2s,color .2s}
.presets button:hover{border:0;transform:translateY(-1px)}.presets button:active{transform:scale(.96)}
.presets button.sel{background:var(--accent);color:#fff;box-shadow:0 5px 15px var(--accent-soft)}
.presets b{font-size:13px}.hint{display:flex;justify-content:space-between;align-items:center;color:var(--ink-2);font-size:13px}
.hint b{color:var(--muted);font-weight:500}
.seg{padding:3px;border:0;border-radius:12px;overflow:visible;background:rgba(118,118,128,.12)}
.seg button{padding:7px 4px;border-radius:9px;font-size:13px;transition:background .25s,transform .25s,color .25s}
.seg button:active{transform:scale(.96)}
.seg button.sel{background:var(--raised);color:var(--ink);font-weight:600;box-shadow:0 1px 4px rgba(0,0,0,.16)}
.acts{gap:9px}.pri,.gho{min-height:42px;border-radius:13px;padding:10px 14px;font-weight:600;
  transition:transform .3s cubic-bezier(.22,1,.36,1),filter .2s,opacity .2s}
.pri{background:var(--accent);color:#fff;box-shadow:0 7px 18px var(--accent-soft)}
.gho{border:0;background:rgba(118,118,128,.12);color:var(--ink)}
.pri:not(:disabled):hover,.gho:hover{filter:brightness(1.05);border:0}.pri:not(:disabled):active,.gho:active{transform:scale(.975)}
.stagewrap{border:1px solid var(--line);border-radius:24px;overflow:hidden;box-shadow:var(--shadow);background:var(--canvas);
  transition:background .45s cubic-bezier(.19,1,.22,1),border-color .3s,box-shadow .3s}
#wrap{padding:20px}.empty{color:var(--empty-text);font-size:14px;transition:color .3s}
#cmp{border-radius:16px;overflow:hidden;box-shadow:0 14px 42px rgba(0,0,0,.5)}
#cmp.reveal{animation:imageIn .6s cubic-bezier(.22,1,.36,1) both}
#bar{width:2px}.tag{top:12px;padding:5px 9px;border-radius:8px;font-size:10px;background:rgba(20,20,22,.58)}
.tag.l{left:12px}.tag.r{right:12px}
#busy{gap:10px;font-weight:600;backdrop-filter:blur(7px);-webkit-backdrop-filter:blur(7px)}
#busy::before{content:"";width:22px;height:22px;border-radius:50%;border:2px solid rgba(255,255,255,.3);border-top-color:#fff;animation:spin .8s linear infinite}
.status{height:34px;padding:0 16px;background:var(--stage-status);color:var(--stage-text);font-size:11px;
  border-color:var(--line);transition:background .3s,color .3s}
#err{border-radius:12px;margin:0 14px 14px;padding:11px 13px}
@keyframes cardIn{from{opacity:0;transform:translateY(10px) scale(.985)}to{opacity:1;transform:none}}
@keyframes imageIn{from{opacity:0;transform:scale(.985)}to{opacity:1;transform:none}}
@keyframes spin{to{transform:rotate(360deg)}}
@media (max-width:900px){
  .app{padding:0 8px 8px}.top{height:58px}.split{grid-template-columns:1fr;grid-template-rows:auto 1fr;gap:8px}
  .panel{max-height:48vh;padding:0}.grp{border-radius:17px;margin-bottom:8px}.stagewrap{border-radius:19px}
}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style></head><body>
<div class="app">
  <div class="top">
    <span class="mark"><i></i></span>
    <h1 data-i="appName"></h1>
    <span class="grow"></span>
    <button class="icon num" id="lang" title="Language">EN</button>
    <button class="icon" id="theme" title="Theme">◑</button>
  </div>

  <div class="split">
    <aside class="panel">
      <div class="grp">
        <span class="eyebrow" data-i="secInput"></span>
        <div id="drop">
          <div class="big" data-i="dropBig"></div>
          <div class="sm" id="dropSm"></div>
          <input type="file" id="file" accept="image/*" hidden>
        </div>
        <div id="thumb">
          <img id="tImg" alt="">
          <div style="min-width:0">
            <div class="tn" id="tName"></div>
            <div class="td num" id="tDim"></div>
          </div>
        </div>
      </div>

      <div class="grp">
        <span class="eyebrow" data-i="secAlpha"></span>
        <div class="aval"><b class="num" id="aval">1.00</b><span data-i="alphaUnit"></span></div>
        <input type="range" id="a" min="0" max="2" step="0.05" value="1.0">
        <div class="presets" id="presets"></div>
      </div>

      <div class="grp">
        <span class="eyebrow" data-i="secView"></span>
        <div class="seg" id="seg">
          <button data-fit="1" class="sel" data-i="viewFit"></button>
          <button data-fit="0" data-i="viewActual"></button>
        </div>
      </div>

      <div class="grp">
        <span class="eyebrow" data-i="secMicro"></span>
        <label class="hint"><span data-i="microLabel"></span> <b class="num" id="microVal">0.55</b></label>
        <input type="range" id="micro" min="0" max="1" step="0.05" value="0.55">
      </div>

      <div class="grp">
        <span class="eyebrow" data-i="secRestore"></span>
        <label class="hint"><span data-i="detailLabel"></span> <b class="num" id="detailVal">0.00</b></label>
        <input type="range" id="detail" min="0" max="1" step="0.05" value="0">
        <label class="hint"><span data-i="casLabel"></span> <b class="num" id="casVal">0.00</b></label>
        <input type="range" id="cas" min="0" max="1" step="0.05" value="0">
        <label class="hint"><span data-i="grainLabel"></span> <b class="num" id="grainVal">0.000</b></label>
        <input type="range" id="grain" min="0" max="0.03" step="0.001" value="0">
      </div>

      <div class="grp">
        <span class="eyebrow" data-i="secUpscale"></span>
        <div class="seg" id="srMode">
          <button data-sr="off" class="sel" data-i="srOff"></button>
          <button data-sr="1x" data-i="sr1x"></button>
          <button data-sr="2x" data-i="sr2x"></button>
        </div>
        <label class="hint"><span data-i="srBlendLabel"></span> <b class="num" id="srBlendVal">0.35</b></label>
        <input type="range" id="srBlend" min="0" max="1" step="0.05" value="0.35">
      </div>

      <div class="grp">
        <div class="acts">
          <button class="pri wide" id="dl" disabled data-i="download"></button>
          <button class="gho" id="dlcmp" disabled data-i="downloadCmp"></button>
          <button class="gho" id="reset" data-i="clear"></button>
        </div>
      </div>

    </aside>

    <div class="stagewrap">
      <div id="dropHint" data-i="dropHere"></div>
      <div id="wrap" class="fit">
        <div class="empty" id="emptyMsg" data-i="emptyHint"></div>
        <div id="cmp" style="display:none">
          <img id="before" alt="" draggable="false">
          <div id="after"><img id="afterimg" alt="" draggable="false"></div>
          <div id="bar"></div>
          <div class="tag l" data-i="tagBefore"></div>
          <div class="tag r" data-i="tagAfter"></div>
          <div id="busy" data-i="working"></div>
        </div>
      </div>
      <div class="status num" id="status"></div>
      <div id="err"></div>
    </div>
  </div>
</div>
<script>
const $=s=>document.querySelector(s);
const I18N={
 zh:{appName:"StopGPTslop",
     secInput:"输入",dropBig:"拖入图片",
     dropSm:'点击选择 · <kbd>Ctrl</kbd>+<kbd>V</kbd> 粘贴',dropHere:"松手即导入",
     secAlpha:"清理力度",alphaUnit:"alpha",
     p25:"保守",p50:"平衡",p75:"强力",p100:"激进",
     secView:"视图",viewFit:"适应窗口",viewActual:"1:1",
     download:"下载结果",downloadCmp:"下载对比图",clear:"清空",
     errExport:"导出失败：",
     tagBefore:"原图",tagAfter:"处理后",working:"处理中…",
     emptyHint:"左侧拖入或粘贴图片后，在此处拖动中缝对比",
     dispScale:"显示",cached:"复用缓存",errPrefix:"处理失败：",errType:"请提供图片文件",
     errRead:"读取文件失败"},
 ru:{appName:"StopGPTslop",
     secInput:"Исходное изображение",dropBig:"Перетащите изображение",
     dropSm:'или нажмите для выбора · <kbd>Ctrl</kbd>+<kbd>V</kbd> для вставки',dropHere:"Отпустите для загрузки",
     secAlpha:"Сила очистки",alphaUnit:"alpha",
     p25:"Бережно",p50:"Сбалансированно",p75:"Сильно",p100:"Агрессивно",
     secView:"Просмотр",viewFit:"Вместить",viewActual:"1:1",
     download:"Скачать PNG",downloadCmp:"Скачать сравнение",clear:"Очистить",
     errExport:"Ошибка экспорта: ",
     tagBefore:"До",tagAfter:"После",working:"Обработка…",
     emptyHint:"Перетащите или вставьте изображение слева, затем двигайте разделитель для сравнения",
     dispScale:"масштаб",cached:"из кеша",errPrefix:"Ошибка: ",errType:"Выберите файл изображения",
     errRead:"Не удалось прочитать файл"},
 en:{appName:"StopGPTslop",
     secInput:"Input",dropBig:"Drop an image",
     dropSm:'or click to browse · <kbd>Ctrl</kbd>+<kbd>V</kbd> to paste',dropHere:"Drop to load",
     secAlpha:"Cleanup strength",alphaUnit:"alpha",
     p25:"Conservative",p50:"Balanced",p75:"Strong",p100:"Aggressive",
     secView:"View",viewFit:"Fit",viewActual:"1:1",
     download:"Download",downloadCmp:"Download comparison",clear:"Clear",
     errExport:"Export failed: ",
     tagBefore:"Before",tagAfter:"After",working:"Working…",
     emptyHint:"Drop or paste an image on the left, then drag the divider here to compare",
     dispScale:"scale",cached:"cached",errPrefix:"Failed: ",errType:"Please provide an image file",
     errRead:"Could not read the file"}};
Object.assign(I18N.ru,{
  secMicro:"\u041c\u0435\u043b\u043a\u0438\u0439 \u043f\u0430\u0442\u0442\u0435\u0440\u043d",microLabel:"\u041f\u043e\u0434\u0430\u0432\u043b\u0435\u043d\u0438\u0435 \u043c\u0438\u043a\u0440\u043e\u0442\u0435\u043a\u0441\u0442\u0443\u0440\u044b",
  secRestore:"\u0412\u043e\u0441\u0441\u0442\u0430\u043d\u043e\u0432\u043b\u0435\u043d\u0438\u0435 \u0434\u0435\u0442\u0430\u043b\u0435\u0439",detailLabel:"\u0427\u0430\u0441\u0442\u043e\u0442\u043d\u0430\u044f \u0440\u0435\u0437\u043a\u043e\u0441\u0442\u044c",casLabel:"CAS-\u0440\u0435\u0437\u043a\u043e\u0441\u0442\u044c",grainLabel:"\u0417\u0435\u0440\u043d\u043e",
  secUpscale:"Real-ESRGAN",srOff:"\u0412\u044b\u043a\u043b.",sr1x:"\u0412\u043e\u0441\u0441\u0442. 1\u00d7",sr2x:"\u0410\u043f\u0441\u043a\u0435\u0439\u043b 2\u00d7",srBlendLabel:"\u0412\u043a\u043b\u0430\u0434 AI"
});
Object.assign(I18N.en,{
  secMicro:"Fine pattern",microLabel:"Micro-texture suppression",
  secRestore:"Detail recovery",detailLabel:"Frequency detail",casLabel:"CAS sharpness",grainLabel:"Grain",
  secUpscale:"Real-ESRGAN",srOff:"Off",sr1x:"Restore 1x",sr2x:"Upscale 2x",srBlendLabel:"AI contribution"
});
Object.assign(I18N.zh,{
  secMicro:"\u7ec6\u5c0f\u7eb9\u7406",microLabel:"\u5fae\u7eb9\u7406\u6291\u5236",
  secRestore:"\u7ec6\u8282\u6062\u590d",detailLabel:"\u9891\u7387\u7ec6\u8282",casLabel:"CAS \u9510\u5316",grainLabel:"\u9897\u7c92",
  secUpscale:"Real-ESRGAN",srOff:"\u5173",sr1x:"\u6062\u590d 1\u00d7",sr2x:"\u653e\u5927 2\u00d7",srBlendLabel:"AI \u6743\u91cd"
});
const PRESETS=[[0.25,"p25"],[0.5,"p50"],[0.75,"p75"],[1.0,"p100"]];

const navLang=(navigator.language||"en").toLowerCase();
let lang=navLang.startsWith("zh")?"zh":navLang.startsWith("ru")?"ru":"en";
lang=localStorage.getItem("stopGPTslop.lang")||lang;
const t=k=>(I18N[lang][k]??k);
function paint(){
  document.documentElement.lang=lang;
  document.title=t("appName");
  document.querySelectorAll("[data-i]").forEach(el=>el.textContent=t(el.dataset.i));
  $("#dropSm").innerHTML=t("dropSm");
  $("#lang").textContent=lang==="ru"?"EN":lang==="en"?"中":"RU";
  $("#presets").innerHTML=PRESETS.map(([v,k])=>
    `<button data-a="${v}"><b class="num">${v.toFixed(2)}</b>${t(k)}</button>`).join("");
  $("#presets").querySelectorAll("button").forEach(b=>b.onclick=()=>{
    slider.value=b.dataset.a;onAlpha();});
  markPresets();updStatus();
}
$("#lang").onclick=()=>{lang=lang==="ru"?"en":lang==="en"?"zh":"ru";
  localStorage.setItem("stopGPTslop.lang",lang);paint();};

const isDark=()=>document.documentElement.dataset.theme
  ? document.documentElement.dataset.theme==="dark"
  : matchMedia("(prefers-color-scheme:dark)").matches;
const savedTheme=localStorage.getItem("stopGPTslop.theme");
if(savedTheme)document.documentElement.dataset.theme=savedTheme;
const paintTheme=()=>{$("#theme").textContent=isDark()?"☀":"☾";};
paintTheme();
$("#theme").onclick=()=>{const d=!isDark();
  document.documentElement.dataset.theme=d?"dark":"light";
  localStorage.setItem("stopGPTslop.theme",d?"dark":"light");paintTheme();};

const drop=$("#drop"),file=$("#file"),slider=$("#a"),aval=$("#aval"),wrap=$("#wrap"),
      cmp=$("#cmp"),before=$("#before"),afterimg=$("#afterimg"),after=$("#after"),bar=$("#bar"),
      busy=$("#busy"),err=$("#err"),status=$("#status"),emptyMsg=$("#emptyMsg"),
      thumb=$("#thumb"),dl=$("#dl"),dlcmp=$("#dlcmp"),
      micro=$("#micro"),detail=$("#detail"),cas=$("#cas"),grain=$("#grain"),srBlend=$("#srBlend");
let srcData=null,key=null,outData=null,fname="",seq=0,pending=false,fit=true,dispScale=1,
    lastMs=null,lastCached=false,outW=null,outH=null,srMode="off",rerun=false;

/* Torph performs the real digit-by-digit morph. Until its local ES module is ready,
   values still update normally, so a missing optional animation can never break UI. */
let TorphTextMorph=null;
const morphs=new Map();
function morphValue(el,value){
  const text=String(value);
  if(!TorphTextMorph){el.textContent=text;return;}
  let morph=morphs.get(el);
  if(!morph){
    el.textContent="";
    morph=new TorphTextMorph({element:el,locale:lang,
      ease:{stiffness:200,damping:20,mass:1},numbers:true,scale:true});
    morphs.set(el,morph);
  }
  morph.update(text);
}
import("/vendor/torph.mjs").then(({TextMorph})=>{
  TorphTextMorph=TextMorph;
  [aval,$("#microVal"),$("#detailVal"),$("#casVal"),$("#grainVal"),$("#srBlendVal")]
    .forEach(el=>morphValue(el,el.textContent));
}).catch(error=>console.warn("Torph unavailable; using static values",error));

const rangeFill=el=>{
  const pct=(+el.value-+el.min)/(+el.max-+el.min)*100;
  el.style.setProperty("--fill",`${pct}%`);
};
document.querySelectorAll('input[type="range"]').forEach(range=>{
  rangeFill(range);
  range.addEventListener("input",()=>{
    rangeFill(range);
    range.animate([
      {transform:"scaleX(1)"},{transform:"scaleX(.992)"},
      {transform:"scaleX(1.006)"},{transform:"scaleX(1)"}
    ],{duration:360,easing:"cubic-bezier(.19,1,.22,1)"});
  });
});
document.addEventListener("pointerdown",event=>{
  const button=event.target.closest("button");
  if(!button||button.disabled)return;
  button.animate([
    {transform:"scale(1)"},{transform:"scale(.94)"},
    {transform:"scale(1.025)"},{transform:"scale(.995)"},{transform:"scale(1)"}
  ],{duration:430,easing:"cubic-bezier(.19,1,.22,1)"});
});

const fail=m=>{err.textContent=m;err.classList.add("on");busy.classList.remove("on");};
const clearErr=()=>err.classList.remove("on");
function markPresets(){
  document.querySelectorAll("#presets button").forEach(b=>
    b.classList.toggle("sel",Math.abs(+b.dataset.a-+slider.value)<1e-9));
}
function updStatus(){
  const iw=before.naturalWidth,ih=before.naturalHeight;
  if(!iw){status.textContent="";return;}
  let s=`${iw}×${ih}  ${(iw*ih/1e6).toFixed(2)}MP  ${t("dispScale")} ${Math.round(dispScale*100)}%`;
  if(outW&&outH&&(outW!==iw||outH!==ih))s+=`  → ${outW}×${outH}`;
  if(lastMs!=null)s+=`  ${lastMs}ms`+(lastCached?`  ${t("cached")}`:"");
  status.textContent=s;
}

/* Capping the scale at 1 means images only shrink. Enlarging would just magnify the
   very defects being judged. The canvas is a fixed-height flex item, so its own client
   size is the available area. */
function layout(){
  const iw=before.naturalWidth,ih=before.naturalHeight;
  if(!iw)return;
  wrap.className=fit?"fit":"act";
  let w=iw,h=ih;
  if(fit){
    const pad=28;
    const s=Math.min(1,(wrap.clientWidth-pad)/iw,(wrap.clientHeight-pad)/ih);
    w=Math.max(1,Math.round(iw*s));h=Math.max(1,Math.round(ih*s));
  }
  dispScale=w/iw;
  cmp.style.width=w+"px";cmp.style.height=h+"px";
  updStatus();
}
// ResizeObserver catches more than window.resize: an error bar appearing, the panel
// collapsing, or a narrow layout stacking all resize the canvas without a window event.
new ResizeObserver(()=>layout()).observe(wrap);
window.addEventListener("resize",layout);
before.addEventListener("load",()=>{
  thumb.classList.add("on");$("#tImg").src=srcData;
  $("#tName").textContent=fname||"—";
  $("#tDim").textContent=`${before.naturalWidth}×${before.naturalHeight}`;
  layout();
});
$("#seg").querySelectorAll("button").forEach(b=>b.onclick=()=>{
  fit=b.dataset.fit==="1";
  $("#seg").querySelectorAll("button").forEach(x=>x.classList.toggle("sel",x===b));
  layout();
});
$("#srMode").querySelectorAll("button").forEach(b=>b.onclick=()=>{
  srMode=b.dataset.sr;
  $("#srMode").querySelectorAll("button").forEach(x=>x.classList.toggle("sel",x===b));
  run();
});

function loadFile(f){
  if(!f||!f.type.startsWith("image/"))return fail(t("errType"));
  clearErr();fname=f.name||"";
  const r=new FileReader();
  r.onload=()=>{srcData=r.result;key=null;lastMs=null;
    before.src=srcData;afterimg.src=srcData;
    emptyMsg.style.display="none";cmp.style.display="";
    cmp.classList.remove("reveal");requestAnimationFrame(()=>cmp.classList.add("reveal"));
    setPos(50);run();};
  r.onerror=()=>fail(t("errRead"));
  r.readAsDataURL(f);
}
drop.onclick=()=>file.click();
thumb.onclick=()=>file.click();
file.onchange=e=>{loadFile(e.target.files[0]);file.value="";};

/* Drops are accepted anywhere in the window -- nobody should have to aim at a small
   box. Track an enter count rather than a boolean: dragleave also fires when moving
   between child elements, and a boolean would make the highlight flicker. */
const hasFiles=e=>Array.from(e.dataTransfer?.types||[]).includes("Files");
let dragDepth=0;
const setDragging=on=>document.body.classList.toggle("dragging",on);
document.addEventListener("dragenter",e=>{
  if(!hasFiles(e))return; e.preventDefault(); dragDepth++; setDragging(true);});
document.addEventListener("dragover",e=>{
  if(hasFiles(e))e.preventDefault();});      // without this, drop never fires
document.addEventListener("dragleave",e=>{
  if(!hasFiles(e))return; dragDepth=Math.max(0,dragDepth-1); if(!dragDepth)setDragging(false);});
document.addEventListener("drop",e=>{
  if(!hasFiles(e))return;
  e.preventDefault(); dragDepth=0; setDragging(false);
  loadFile(e.dataTransfer.files[0]);});
document.addEventListener("paste",e=>{
  for(const it of (e.clipboardData||{}).items||[])
    if(it.type.startsWith("image/")){loadFile(it.getAsFile());return;}
});

const setPos=p=>{p=Math.max(0,Math.min(100,p));
  after.style.clipPath=`inset(0 0 0 ${p}%)`;bar.style.left=p+"%";};
const at=e=>{const r=cmp.getBoundingClientRect();return (e.clientX-r.left)/r.width*100;};
let drag=false;
cmp.addEventListener("pointerdown",e=>{
  e.preventDefault();                 // stop the browser's own image drag
  drag=true;try{cmp.setPointerCapture(e.pointerId);}catch(_){}
  setPos(at(e));});
cmp.addEventListener("pointermove",e=>{if(drag)setPos(at(e));});
// Missing either cancel or lostpointercapture leaves the divider stuck to the cursor
["pointerup","pointercancel","lostpointercapture"].forEach(k=>
  cmp.addEventListener(k,()=>{drag=false;}));
cmp.addEventListener("dragstart",e=>e.preventDefault());

function onAlpha(){
  morphValue(aval,(+slider.value).toFixed(2));rangeFill(slider);markPresets();
  clearTimeout(slider._t);slider._t=setTimeout(run,240);   // debounce while dragging
}
slider.oninput=onAlpha;
function onRestore(){
  morphValue($("#microVal"),(+micro.value).toFixed(2));
  morphValue($("#detailVal"),(+detail.value).toFixed(2));
  morphValue($("#casVal"),(+cas.value).toFixed(2));
  morphValue($("#grainVal"),(+grain.value).toFixed(3));
  morphValue($("#srBlendVal"),(+srBlend.value).toFixed(2));
  [micro,detail,cas,grain,srBlend].forEach(rangeFill);
  clearTimeout(detail._t);detail._t=setTimeout(run,240);
}
[micro,detail,cas,grain,srBlend].forEach(x=>x.oninput=onRestore);

async function run(){
  if(!srcData)return;
  if(pending){rerun=true;return;}
  pending=true;const my=++seq;busy.classList.add("on");clearErr();
  try{
    const r=await fetch("/api/process",{method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({image:srcData,alpha:+slider.value,key,
                           micro:+micro.value,detail:+detail.value,cas:+cas.value,grain:+grain.value,
                           sr_mode:srMode,sr_blend:+srBlend.value})});
    const j=await r.json();
    if(!r.ok||j.error)throw new Error(j.error||("HTTP "+r.status));
    key=j.key;outData=j.result;afterimg.src=outData;
    lastMs=j.ms;lastCached=j.cached;outW=j.w;outH=j.h;
    dl.disabled=dlcmp.disabled=false;updStatus();
  }catch(e){fail(t("errPrefix")+e.message);}
  finally{
    busy.classList.remove("on");pending=false;
    if(rerun){rerun=false;run();}
  }
}
const baseName=()=>fname.replace(/\.[^.]+$/,"")||"image";
const saveBlobUrl=(href,name,revoke)=>{
  const a=document.createElement("a");a.href=href;a.download=name;a.click();
  if(revoke)setTimeout(()=>URL.revokeObjectURL(href),1000);
};
dl.onclick=()=>{
  if(!outData)return;
  const fx=(+micro.value||+detail.value||+cas.value||+grain.value)?"_fx":"";
  const sr=srMode==="off"?"":`_sr${srMode}`;
  saveBlobUrl(outData,`${baseName()}_clean_a${(+slider.value).toFixed(2)}${fx}${sr}.png`,false);
};
/* Compose the pair at full resolution, not at the on-screen size: the point of the
   export is to inspect detail, and exporting the scaled view would throw that away.
   Same layout as modeling.py --side-by-side, so the two can be viewed together. */
dlcmp.onclick=()=>{
  if(!outData||!before.naturalWidth)return;
  const iw=afterimg.naturalWidth||before.naturalWidth;
  const ih=afterimg.naturalHeight||before.naturalHeight,gap=6;
  try{
    const c=document.createElement("canvas");
    c.width=iw*2+gap;c.height=ih;
    const x=c.getContext("2d");
    if(!x)throw new Error("canvas 2d unavailable");
    x.fillStyle="#fff";x.fillRect(0,0,c.width,c.height);
    x.drawImage(before,0,0,iw,ih);
    x.drawImage(afterimg,iw+gap,0,iw,ih);
    c.toBlob(b=>{
      if(!b)return fail(t("errExport")+`canvas ${c.width}x${c.height}`);
      saveBlobUrl(URL.createObjectURL(b),
                  `${baseName()}_clean_compare_a${(+slider.value).toFixed(2)}.png`,true);
    },"image/png");
  }catch(e){fail(t("errExport")+e.message);}
};
$("#reset").onclick=()=>{
  srcData=outData=key=null;fname="";lastMs=outW=outH=null;fit=true;dl.disabled=dlcmp.disabled=true;
  srMode="off";
  micro.value="0.55";detail.value="0";cas.value="0";grain.value="0";srBlend.value="0.35";
  onRestore();
  thumb.classList.remove("on");cmp.style.display="none";emptyMsg.style.display="";
  $("#seg").querySelectorAll("button").forEach(x=>
    x.classList.toggle("sel",x.dataset.fit==="1"));
  $("#srMode").querySelectorAll("button").forEach(x=>
    x.classList.toggle("sel",x.dataset.sr==="off"));
  status.textContent="";clearErr();
};
paint();
</script></body></html>
"""


def _encode_z(arr_u8):
    """Run the encoder and the refiner once per image."""
    dev, dtype = STATE["device"], STATE["dtype"]
    H, W = arr_u8.shape[:2]
    x = torch.from_numpy(arr_u8.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(dev)
    ph, pw = (-H) % ALIGN, (-W) % ALIGN
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    with torch.no_grad(), _autocast_context(dev, dtype):
        z = STATE["enc"](x * 2 - 1)
        dz = STATE["R"](z)
    return z, dz, H, W


def _decode(z, dz, alpha, H, W):
    dev, dtype = STATE["device"], STATE["dtype"]
    with torch.no_grad(), _autocast_context(dev, dtype):
        y = STATE["dec"](z + alpha * dz)
    y = ((y.float().clamp(-1, 1) + 1) / 2)[0, :, :H, :W]
    return (y.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)


def _realesrgan(arr_u8, target_scale, blend):
    """Run compact Real-ESRGAN in padded tiles and resize each tile to 1x/2x."""
    if target_scale not in (1, 2):
        return arr_u8
    blend = float(np.clip(blend, 0.0, 1.0))
    if blend == 0.0:
        if target_scale == 1:
            return arr_u8
        h, w = arr_u8.shape[:2]
        return np.asarray(Image.fromarray(arr_u8).resize(
            (w * target_scale, h * target_scale), Image.Resampling.LANCZOS
        ))

    model, device, dtype = STATE["sr_model"], STATE["device"], STATE["sr_dtype"]
    tile, pad = STATE["sr_tile"], 16
    h, w = arr_u8.shape[:2]
    restored = Image.new("RGB", (w * target_scale, h * target_scale))
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            x1, y1 = min(x0 + tile, w), min(y0 + tile, h)
            px0, py0 = max(0, x0 - pad), max(0, y0 - pad)
            px1, py1 = min(w, x1 + pad), min(h, y1 + pad)
            patch = np.ascontiguousarray(arr_u8[py0:py1, px0:px1])
            tensor = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0)
            tensor = tensor.to(device=device, dtype=dtype).div_(255.0)
            with torch.inference_mode():
                pred = model(tensor).float().clamp_(0, 1)[0]
            pred_u8 = (pred.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
            patch_w, patch_h = (px1 - px0) * target_scale, (py1 - py0) * target_scale
            reduced = Image.fromarray(pred_u8).resize(
                (patch_w, patch_h), Image.Resampling.LANCZOS
            )
            crop = reduced.crop(((x0 - px0) * target_scale,
                                 (y0 - py0) * target_scale,
                                 (x1 - px0) * target_scale,
                                 (y1 - py0) * target_scale))
            restored.paste(crop, (x0 * target_scale, y0 * target_scale))

    ai = np.asarray(restored).astype(np.float32)
    base = np.asarray(Image.fromarray(arr_u8).resize(
        restored.size, Image.Resampling.LANCZOS
    )).astype(np.float32)
    return np.clip(base + blend * (ai - base), 0, 255).round().astype(np.uint8)


def _smoothstep(low, high, value):
    """Hermite threshold without the hard rims a binary texture mask creates."""
    scaled = np.clip((value - low) / max(high - low, 1e-6), 0.0, 1.0)
    return scaled * scaled * (3.0 - 2.0 * scaled)


def _blur_positive_field(field, radius, ceiling):
    """Blur a non-negative float field through Pillow's fast 8-bit Gaussian path."""
    encoded = np.clip(field * (255.0 / ceiling), 0, 255).round().astype(np.uint8)
    blurred = np.asarray(
        Image.fromarray(encoded).filter(ImageFilter.GaussianBlur(radius)),
        dtype=np.float32,
    )
    return blurred * (ceiling / 255.0)


def _suppress_micro_pattern(arr_u8, amount):
    """Attenuate repeated 1-6 px texture while protecting larger image contours.

    The latent refiner has an effective pixel stride of eight, so very fine grids can
    survive it.  This pixel-space pass measures fine-band energy locally, distinguishes
    it from broader edges, and only blends textured regions toward a small Gaussian
    base.  It deliberately exposes a strength control: fabric, hair and AI micro-patterns
    overlap spectrally, so no fixed threshold is correct for every image.
    """
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount == 0.0:
        return arr_u8

    image = arr_u8.astype(np.float32) / 255.0
    pil = Image.fromarray(arr_u8)
    fine_base = np.asarray(
        pil.filter(ImageFilter.GaussianBlur(0.75)), dtype=np.float32
    ) / 255.0
    small_base = np.asarray(
        pil.filter(ImageFilter.GaussianBlur(1.65)), dtype=np.float32
    ) / 255.0
    broad_base = np.asarray(
        pil.filter(ImageFilter.GaussianBlur(3.4)), dtype=np.float32
    ) / 255.0

    fine = image - fine_base
    small = fine_base - small_base
    coeff = np.array([0.2126, 0.7152, 0.0722], np.float32)
    fine_luma = fine @ coeff
    small_luma = small @ coeff

    # A repeated micro-pattern has persistent local band energy. Isolated contour
    # pixels also have high energy, but the broader-gradient protection below removes
    # them from the mask.
    energy = np.abs(fine_luma) + 0.55 * np.abs(small_luma)
    local_energy = _blur_positive_field(energy, 2.2, 0.14)
    texture = _smoothstep(0.006, 0.030, local_energy)

    small_luma_base = small_base @ coeff
    grad_y, grad_x = np.gradient(small_luma_base)
    gradient = np.hypot(grad_x, grad_y)
    broad_structure = np.abs((small_base - broad_base) @ coeff)
    protection = _smoothstep(0.014, 0.080, gradient + 0.45 * broad_structure)

    mask = texture * (1.0 - 0.88 * protection)
    mask = _blur_positive_field(mask, 0.8, 1.0)
    correction = 0.90 * fine + 0.38 * small
    output = image - amount * mask[..., None] * correction
    return np.clip(output * 255.0, 0, 255).round().astype(np.uint8)


def _frequency_and_cas(arr_u8, detail, cas):
    """Luminance frequency split followed by contrast-adaptive sharpening."""
    detail, cas = float(np.clip(detail, 0, 1)), float(np.clip(cas, 0, 1))
    image = arr_u8.astype(np.float32) / 255.0
    if detail > 0:
        pil = Image.fromarray(arr_u8)
        small = np.asarray(pil.filter(ImageFilter.GaussianBlur(0.65))).astype(np.float32) / 255.0
        large = np.asarray(pil.filter(ImageFilter.GaussianBlur(1.8))).astype(np.float32) / 255.0
        coeff = np.array([0.2126, 0.7152, 0.0722], np.float32)
        fine = ((image - small) * coeff).sum(axis=2)
        medium = ((small - large) * coeff).sum(axis=2)
        image = np.clip(image + detail * (0.85 * fine + 0.35 * medium)[..., None], 0, 1)

    if cas > 0:
        coeff = np.array([0.2126, 0.7152, 0.0722], np.float32)
        luma = (image * coeff).sum(axis=2)
        p = np.pad(luma, 1, mode="edge")
        north, south = p[:-2, 1:-1], p[2:, 1:-1]
        west, east = p[1:-1, :-2], p[1:-1, 2:]
        local_min = np.minimum.reduce((north, south, west, east, luma))
        local_max = np.maximum.reduce((north, south, west, east, luma))
        amplify = np.sqrt(np.clip(
            np.minimum(local_min, 1.0 - local_max) / np.maximum(local_max, 1e-4), 0, 1
        ))
        weight = amplify * (-1.0 / 6.5)
        rgb = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode="edge")
        cross = (rgb[:-2, 1:-1] + rgb[2:, 1:-1] +
                 rgb[1:-1, :-2] + rgb[1:-1, 2:])
        sharpened = (image + weight[..., None] * cross) / (1 + 4 * weight[..., None])
        image = np.clip(image + cas * (sharpened - image), 0, 1)
    return (image * 255).round().astype(np.uint8)


def _add_grain(arr_u8, amount, seed):
    amount = float(np.clip(amount, 0, 0.04))
    if amount == 0:
        return arr_u8
    image = arr_u8.astype(np.float32) / 255.0
    luma = image @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, luma.shape).astype(np.float32)
    # Film-like monochrome grain: present everywhere, slightly stronger in shadows.
    adaptive = 0.55 + 0.45 * (1.0 - luma)
    image = np.clip(image + noise[..., None] * adaptive[..., None] * amount, 0, 1)
    return (image * 255).round().astype(np.uint8)


def process(img_b64, alpha, key, micro=0.0, detail=0.0, cas=0.0, grain=0.0,
            sr_mode="off", sr_blend=0.35):
    raw = base64.b64decode(img_b64.split(",", 1)[-1])
    arr = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
    k = key or hashlib.md5(raw).hexdigest()
    t0 = time.time()
    with LOCK:                                  # serialise GPU work across requests
        hit = k in CACHE
        if not hit:
            z, dz, H, W = _encode_z(arr)
            # Cache the uncorrected VAE round-trip. Subtracting it from the
            # corrected decode isolates the learned cleanup delta, so preservation
            # mode keeps the original pixels instead of replacing them with a
            # slightly softer VAE reconstruction.
            baseline = _decode(z, dz, 0.0, H, W)
            CACHE[k] = (z, dz, H, W, baseline)
            CACHE_ORDER.append(k)
            while len(CACHE_ORDER) > STATE["cache_n"]:
                CACHE.pop(CACHE_ORDER.pop(0), None)
        z, dz, H, W, baseline = CACHE[k]
        decoded = _decode(z, dz, alpha, H, W)
        delta = decoded.astype(np.int16) - baseline.astype(np.int16)
        out = np.clip(arr.astype(np.int16) + delta, 0, 255).astype(np.uint8)
        if sr_mode not in {"off", "1x", "2x"}:
            raise ValueError("unknown Real-ESRGAN mode")
        if sr_mode != "off":
            out = _realesrgan(out, 1 if sr_mode == "1x" else 2, sr_blend)
        out = _suppress_micro_pattern(out, micro)
        out = _frequency_and_cas(out, detail, cas)
        out = _add_grain(out, grain, int(k[:16], 16))
    buf = io.BytesIO()
    # PNG avoids adding lossy compression artifacts or softness to an image whose
    # fine detail is the entire reason for using the cleaner.
    Image.fromarray(out).save(buf, format="PNG", compress_level=1)
    # Large VAE and SR tiles leave sizable temporary CUDA/CPU arenas behind. They
    # are useful for batch throughput but harmful in this interactive 8 GB setup,
    # where the next slider change otherwise appears to hang near the VRAM limit.
    if STATE["device"].split(":")[0] == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    out_h, out_w = out.shape[:2]
    return dict(result="data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
                key=k, w=out_w, h=out_h, ms=int((time.time() - t0) * 1000), cached=hit)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/vendor/torph.mjs":
            try:
                with open(TORPH_JS, "rb") as src:
                    return self._send(200, src.read(), "text/javascript; charset=utf-8")
            except OSError:
                return self._send(404, b"not found", "text/plain")
        if path not in ("/", "/index.html"):
            return self._send(404, b"not found", "text/plain")
        self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")

    def do_POST(self):
        if self.path != "/api/process":
            return self._send(404, b'{"error":"not found"}', "application/json")
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n))
            out = process(req["image"], float(req.get("alpha", 1.0)), req.get("key"),
                          float(req.get("micro", 0.0)), float(req.get("detail", 0.0)),
                          float(req.get("cas", 0.0)),
                          float(req.get("grain", 0.0)), req.get("sr_mode", "off"),
                          float(req.get("sr_blend", 0.35)))
            body = json.dumps(out).encode()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return self._send(500, json.dumps({"error": "out of GPU memory"}).encode(),
                              "application/json")
        except Exception as e:
            return self._send(500, json.dumps({"error": str(e)}).encode(), "application/json")
        self._send(200, body, "application/json")

    def log_message(self, fmt, *a):     # quieten per-request logging; keep non-2xx
        if not str(a[1] if len(a) > 1 else "").startswith("2"):
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % a))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--refiner-weight",
                   default=os.path.join(HERE, "weights", "flux2_refiner.pt"))
    p.add_argument("--realesrgan-weight", default=os.path.join(
        HERE, "weights", "realesrgan", "realesr-general-wdn-x4v3.pth"))
    p.add_argument("--sr-tile", type=int, default=256,
                   help="Real-ESRGAN input tile size; lower values use less VRAM")
    p.add_argument("--vae", default="black-forest-labs/FLUX.2-VAE")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--fp32", action="store_true")
    p.add_argument("--no-tile-vae", action="store_true",
                   help="disable VAE tiling")
    p.add_argument("--cache-n", type=int, default=4, help="how many images to keep encoded, so strength changes skip re-encoding")
    p.add_argument("--no-compile", action="store_true",
                   help="disable compilation. It is on by default: the one-off cost is repaid "
                        "quickly for a long-running server, at the price of a slower "
                        "first image")
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()

    dtype = torch.float32 if args.fp32 else torch.bfloat16
    t0 = time.time()
    print("Loading model ...", flush=True)
    vae = load_vae(args.vae, args.device)
    R, refiner_meta = load_refiner_model(args.refiner_weight, args.device)
    sr_model, sr_dtype = load_realesrgan_model(args.realesrgan_weight, args.device)
    if not args.no_tile_vae and hasattr(vae, "enable_tiling"):
        vae.enable_tiling()
    enc, dec = build_fns(vae, not args.no_compile)
    STATE.update(vae=vae, R=R,
                 sr_model=sr_model, sr_dtype=sr_dtype, sr_tile=max(64, args.sr_tile),
                 enc=enc, dec=dec, device=args.device, dtype=dtype,
                 cache_n=max(1, args.cache_n))
    # Bind before announcing readiness. The other order prints a URL and only then
    # fails on an occupied port, sending users to whatever else is listening there.
    try:
        httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError as e:
        sys.exit(f"Cannot bind {args.host}:{args.port} -- {e}\nTry another --port")
    url = f"http://{args.host}:{args.port}"
    print(f"Ready in {time.time()-t0:.1f}s | {args.device} "
          f"{'fp32' if args.fp32 else 'bf16'} | "
          f"compiled={'no' if args.no_compile else 'yes'}\n{url}", flush=True)
    if not args.no_compile:
        print("  The first image will take longer while the model is compiled", flush=True)
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped", flush=True)
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
