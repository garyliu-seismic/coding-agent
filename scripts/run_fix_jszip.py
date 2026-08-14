"""Run the agent in `run` mode to fix the JSZip CDN/npm inconsistency."""
from __future__ import annotations

import os

os.chdir(r"C:\Users\garyl\coding-agent")

from coding_agent.cli import main

TASK = """Fix the JSZip dependency inconsistency so the npm package is bundled instead of the CDN.
1) In src/services/previewService.ts, add:  import JSZip from "jszip";
   at the top with the other imports, and delete these two lines inside the previewDocument function:
   - const JSZip = (window as any).JSZip;
   - if (!JSZip) throw new Error('JSZip not loaded.');
2) In src/taskpane.html, remove the jszip CDN <script> tag (the line containing https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js).
3) Run npm run build and confirm webpack compiles successfully.
Then summarize what you changed and call finish."""

raise SystemExit(
    main(["--root", r"C:\Users\garyl\hello", "run", TASK])
)
