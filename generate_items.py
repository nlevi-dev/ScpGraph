import base64
import gzip
import os
import re
import pandas as pd

df = pd.read_csv("914_outputs_expanded.csv")[["Input", "admin_only"]].drop_duplicates("Input")

os.makedirs("index/items", exist_ok=True)

def slug(name):
    return "graph_full_" + re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")

for _, r in df.iterrows():
    name = r["Input"]
    svg_path = f"graphs/{slug(name)}.svg"
    if not os.path.exists(svg_path):
        print(f"Missing {svg_path}, skipping")
        continue

    with open(svg_path, "rb") as f_in:
        gz_b64 = base64.b64encode(gzip.compress(f_in.read())).decode()

    html = ("<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"UTF-8\">"
            f"<title>SCP-914: {name}</title>"
            "<style>body{font-family:sans-serif;margin:1rem}a{color:#2980b9}"
            "svg{width:1280px;height:auto}</style></head><body>"
            f"<p><a href=\"../index.html\">&#8592; Back to index</a></p>"
            f"<h1>{name}</h1><div id=g></div>"
            f"<script>var d='{gz_b64}';"
            "var b=Uint8Array.from(atob(d),c=>c.charCodeAt(0));"
            "new Response(new Blob([b]).stream().pipeThrough(new DecompressionStream('gzip'))).text()"
            ".then(t=>{document.getElementById('g').innerHTML=t;"
            "var s=document.querySelector('svg');"
            "function r(){s.style.width=Math.max(window.innerWidth*window.devicePixelRatio,window.innerWidth)-32+'px'}"
            "window.addEventListener('resize',r);screen.orientation.addEventListener('change',r);r();})"
            "</script></body></html>")
    out = f"index/items/{slug(name)}.html"
    with open(out, "w") as f:
        f.write(html)
    print(f"Saved {out}")
