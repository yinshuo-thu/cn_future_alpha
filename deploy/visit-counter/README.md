# Visit counter for the cn-future-alpha report

A tiny, stdlib-only page-visit counter shown in the bottom-right corner of
<https://autoalpha.cn/cn_future_alpha/>. No external dependencies.

## Pieces

- **`counter.py`** — a `http.server` service on `127.0.0.1:8766`. Persists the
  count in `/opt/cnfa-counter/count.txt` (atomic, `flock`).
  - `GET /cn_future_alpha/visits`     → peek (no increment)
  - `GET /cn_future_alpha/visits/hit` → +1, returns the new count
  - response: `{"count": N}`
- **`cnfa-counter.service`** — systemd unit (runs as `www-data`).
- **`nginx-location.conf`** — the nginx `location` to proxy `/cn_future_alpha/visits`
  to the service (add it to the `autoalpha.cn` server block).
- **Page side** (in `summary_src.html`, between the `VISIT-COUNTER` comment
  markers): a fixed bottom-right badge + a small script that calls the endpoint on
  load and shows the count. It counts a visit at most once per 6h per browser
  (localStorage), hides itself if the backend is unreachable, and is `display:none`
  in print. `tools/build_report_export.py` strips the whole block from the PDF
  exports, so the submission PDFs are unaffected.

## Install on the server

```bash
sudo mkdir -p /opt/cnfa-counter
sudo cp counter.py /opt/cnfa-counter/counter.py
echo 0 | sudo tee /opt/cnfa-counter/count.txt
sudo chown -R www-data:www-data /opt/cnfa-counter

sudo cp cnfa-counter.service /etc/systemd/system/cnfa-counter.service
sudo systemctl daemon-reload
sudo systemctl enable --now cnfa-counter

# add nginx-location.conf into the autoalpha.cn server block, then:
sudo nginx -t && sudo systemctl reload nginx
```

Reset the count any time with:
`echo 0 | sudo tee /opt/cnfa-counter/count.txt && sudo systemctl restart cnfa-counter`
