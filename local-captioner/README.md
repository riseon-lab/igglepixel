# IgglePixel Local Captioner

Run the local captioner with the Python backend when using Video Miner:

```bash
cd local-captioner
python3 -m pip install -r requirements.txt
python3 server.py
```

Open:

```text
http://127.0.0.1:8778/
```

The Video Miner tab extracts frames with OpenCV, ranks them through the configured local vision endpoint, and can run a conservative detail-preserving enhancement pass. Append mode lets multiple videos build one candidate pool. Ranking skips already-ranked frames unless Force re-rank is enabled; enhancement skips already-enhanced frames unless Force enhance is enabled.

To use an external enhancer later, start the server with `LOCAL_CAPTIONER_ENHANCE_CMD` containing `{input}` and `{output}` placeholders.
