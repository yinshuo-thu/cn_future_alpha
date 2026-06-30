# Visit counter for the cn-future-alpha report

A tiny, stdlib-only **unique-IP** visit counter shown inline in the footer of
<https://autoalpha.cn/cn_future_alpha/>. No external dependencies.

## Pieces

- **`counter.py`** — a `http.server` service on `127.0.0.1:8766`. Counts each
  client IP **at most once** (the real IP comes from nginx's `X-Real-IP`). IPs are
  stored hashed (sha256, salted) in `/opt/cnfa-counter/ips.txt`; the count is the
  number of distinct IPs (mirrored to `count.txt` for inspection).
  - `GET /cn_future_alpha/visits`     → peek (current unique count, no change)
  - `GET /cn_future_alpha/visits/hit` → register this IP if new; returns the count
  - response: `{"count": N}`
- **`cnfa-counter.service`** — systemd unit (runs as `www-data`).
- **`nginx-location.conf`** — the nginx `location` to proxy `/cn_future_alpha/visits`
  to the service (add it to the `autoalpha.cn` server block).
- **Page side** (in `summary_src.html`, between the `VISIT-COUNTER` comment
  markers): a small muted-gray "N visits" shown inline in the footer, on the same
  line as the thank-you note, plus a script that calls `/hit` on load (the server
  dedups by IP). It hides itself if the backend is unreachable and is
  `display:none` in print. `tools/build_report_export.py` strips the whole block
  from the PDF exports, so the submission PDFs are unaffected.

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
`sudo truncate -s0 /opt/cnfa-counter/ips.txt && echo 0 | sudo tee /opt/cnfa-counter/count.txt && sudo systemctl restart cnfa-counter`
