# Schema catalogue — UNSW-NB15 vs TON_IoT Network

**Phase 1 deliverable.** Direct input to Phase 2 (feature alignment). Built from the three
delivered CSVs plus the official upstream feature-description artifacts under
`data/raw/reference/`. Companion machine-readable file: `reports/schema_catalogue.csv`
(one row per delivered column, 45 + 44 = 89 rows).

Every number below is measured from the delivered files, not quoted from documentation.
Stats are on **UNSW-NB15 training set** (175,341 rows) and the full **TON_IoT
`Train_Test_Network.csv`** (211,043 rows); the UNSW testing set (82,332 rows) is
**structurally identical** — same 45 column names in the same order, same pandas dtypes —
so it needs no separate catalogue. Where its *values* diverge (categorical vocabularies) that
is called out explicitly.

No raw records appear in this file. Categorical vocabularies, numeric ranges and counts only;
IP addresses, DNS queries, HTTP URIs, user agents and certificate subjects are reported as
cardinality only, per both datasets' redistribution terms.

All three CSVs carry a UTF-8 BOM — every read uses `encoding="utf-8-sig"`.

## 1. Aligned mapping table (the eight `FEATURE_MAP` concepts)

Ranges are min–max on the stated split. **Verdict** is the Phase 2 action.

| concept | UNSW-NB15 column · dtype · range | TON_IoT column · dtype · range | unit / semantic mismatch | verdict |
| --- | --- | --- | --- | --- |
| `flow_duration` | `dur` · float64 · 0–59.999989 s | `duration` · float64 · 0–93,516.93 s | Same unit (seconds) but **incomparable support**: UNSW is hard-capped at 60 s by the capture design; TON_IoT runs to 26 hours. 1.5% of UNSW rows vs **28.4%** of TON_IoT rows are exactly 0. | **needs normalization** — log1p, and clip/winsorize the TON_IoT tail before z-scoring, or the UNSW-train-fitted scaler maps almost all TON_IoT rows into one bin. |
| `protocol` | `proto` · object · 133 levels | `proto` · object · 3 levels (`tcp`,`udp`,`icmp`) | Severe cardinality asymmetry. TON_IoT's 3 levels are all present in UNSW, but UNSW's 133 Argus protocol names cover only **81.7%** of its own train rows with `{tcp,udp,icmp}`; 31 levels have <100 rows. UNSW `icmp` is 15 rows in train and **0 in test**. | **needs collapsing** — one-hot `{tcp, udp, icmp, other}` only. Anything finer is a UNSW-only vocabulary the model cannot exercise cross-era. |
| `service` | `service` · object · 13 levels (`-` = 53.7%) | `service` · object · 9 levels (`-` = 62.6%) | Both use literal `-` for 'not detected', and it is the modal value on both sides. Only 5 levels are shared (`-`,`dns`,`ftp`,`http`,`ssl`) but they cover **93.3%** of UNSW train, 95.2% of UNSW test and **99.8%** of TON_IoT. `ftp-data` (UNSW) and `smb;gssapi` (TON_IoT, a multi-valued cell) are era/domain artifacts. | **needs collapsing** — keep `{-, dns, http, ftp, ssl, other}`; split TON_IoT's `;`-joined cells to their first token first. Treat `-` as its own level, not as an imputed NaN. |
| `conn_state` | `state` · object · 9 levels train / 11 across both splits | `conn_state` · object · 13 levels | **Vocabularies are strictly disjoint — zero shared tokens.** UNSW ships Argus states (`INT`,`FIN`,`CON`,`REQ`,`RST`,`ECO`,`ACC`,`CLO`,`PAR`,`URN`, plus one junk `no`); TON_IoT ships Zeek `conn_state` (`S0`,`S1`,`S2`,`S3`,`SF`,`SH`,`SHR`,`REJ`,`RSTO`,`RSTOS0`,`RSTR`, `RSTRH`,`OTH`). UNSW's train/test vocabularies also differ (train-only `ECO`,`PAR`,`URN`,`no`; test-only `ACC`,`CLO`). | **needs collapsing** — a hand-written 3-way map (completed / reset / no-response) is the only way this feature survives; see §4.5 for the proposed map. Dropping it is defensible. |
| `src_bytes` | `sbytes` · int64 · 28–12,965,233 B | **`src_ip_bytes`** · int64 · 0–6,522,626 B  *(currently `src_bytes` in the code)* | **The coded pairing is wrong.** UNSW `sbytes` is IP-level (min = 28 B = 20 B IP + 8 B UDP header; never 0). TON_IoT `src_bytes` is Zeek *payload* bytes and is **0 on 65.5%** of rows. `src_ip_bytes` is total IP bytes and is 0 on only 8.1%. See §4.6. | **needs normalization, after repairing the pairing** — repoint to `src_ip_bytes`, then log1p both sides. |
| `dst_bytes` | `dbytes` · int64 · 0–14,655,550 B | **`dst_ip_bytes`** · int64 · 0–86,395,523 B  *(currently `dst_bytes` in the code)* | Same defect as `src_bytes`: TON_IoT `dst_bytes` is payload and 0 on 70.6% of rows; `dst_ip_bytes` is IP-level and 0 on 39.4%, which matches UNSW `dbytes`'s 48.1% zero rate (both are 'destination never replied'). | **needs normalization, after repairing the pairing** — repoint to `dst_ip_bytes`, then log1p. |
| `src_pkts` | `spkts` · int64 · 1–9,616 | `src_pkts` · int64 · 0–24,623 | Same notion (count of source-originated packets), same unit. Only edge difference: UNSW's floor is 1, TON_IoT allows 0 (8.1% of rows — Zeek logs responder-only/`OTH` flows). | **keep as-is** (log1p for the heavy tail). The cleanest of the eight pairings. |
| `dst_pkts` | `dpkts` · int64 · 0–10,974 | `dst_pkts` · int64 · 0–121,942 | Same notion and unit; zero rates are comparable (UNSW 48.1%, TON_IoT 39.4%). | **keep as-is** (log1p). |

**Summary of verdicts:** 2 keep as-is (`src_pkts`, `dst_pkts`), 1 needs normalization
(`flow_duration`), 2 need collapsing (`protocol`, `service`), 1 needs collapsing *or* dropping
(`conn_state`), and 2 need the **TON_IoT column repointed** before anything else
(`src_bytes`, `dst_bytes`). No pairing is a clean drop.

Plus the two `DERIVED_FEATURES`, computed identically on both sides from the mapped
ingredients (see §4.7 for the zero-duration guard):

| derived | UNSW-NB15 | TON_IoT |
| --- | --- | --- |
| `bytes_per_sec` | `(sbytes + dbytes) / dur` | `(src_ip_bytes + dst_ip_bytes) / duration` |
| `pkts_per_sec` | `(spkts + dpkts) / dur` | `(src_pkts + dst_pkts) / duration` |

## 2. Full column tables

`n_missing` counts the sentinel named in `missing_token`. `top_share` is the share of rows held
by the single most common value — the near-zero-variance indicator. Full machine-readable
version, including `description_source` per column, is in `reports/schema_catalogue.csv`.

### 2.1 UNSW-NB15 — 45 columns (training set, 175,341 rows)

| # | column | dtype | semantic | n_unique | n_missing (token) | top_share | disposition | meaning |
| ---: | --- | --- | --- | ---: | --- | ---: | --- | --- |
| 1 | `id` | int64 | identity | 175,341 | — | 0.000 | drop-identity | Row index of the partitioned set; 1..N, unique per row. Not a traffic feature. |
| 2 | `dur` | float64 | numeric-continuous | 74,039 | — | 0.131 | mapped | Record total duration (seconds); span from first to last packet of the flow. |
| 3 | `proto` | object | categorical | 133 | — | 0.456 | mapped | Transaction protocol (Argus protocol name). |
| 4 | `service` | object | categorical | 13 | 94,168 ('-') | 0.537 | mapped | Application service inferred by Bro/Zeek; '-' when no service was identified. |
| 5 | `state` | object | categorical | 9 | — | 0.469 | mapped | Argus transaction state and its dependent protocol (ACC, CLO, CON, ECO, ECR, FIN, INT, MAS, PAR, REQ, RST, TST, TXD, URH, URN, '-'). NOT Zeek conn_state codes. |
| 6 | `spkts` | int64 | numeric-count | 480 | — | 0.486 | mapped | Source to destination packet count. |
| 7 | `dpkts` | int64 | numeric-count | 443 | — | 0.481 | mapped | Destination to source packet count. |
| 8 | `sbytes` | int64 | numeric-count | 7,214 | — | 0.223 | mapped | Source to destination transaction bytes. Empirically IP-level (min=28 = 20B IP + 8B UDP header), i.e. headers included, not payload-only. |
| 9 | `dbytes` | int64 | numeric-count | 6,660 | — | 0.481 | mapped | Destination to source transaction bytes; same IP-level notion as sbytes. |
| 10 | `rate` | float64 | numeric-continuous | 76,991 | — | 0.131 | drop-unmapped | Packets per second over the flow span. Empirically (spkts+dpkts-1)/dur (99.82% of dur>0 rows match within 0.1%). Argus (n-1)-interval convention. |
| 11 | `sttl` | int64 | numeric-count | 11 | — | 0.654 | drop-unmapped | Source to destination time to live value. |
| 12 | `dttl` | int64 | numeric-count | 6 | — | 0.481 | drop-unmapped | Destination to source time to live value. |
| 13 | `sload` | float64 | numeric-continuous | 80,885 | — | 0.068 | drop-unmapped | Source bits per second. Empirically sbytes*8*(spkts-1)/(spkts*dur), i.e. bits/sec with the Argus (n-1)/n packet correction (99.98% match within 1%). |
| 14 | `dload` | float64 | numeric-continuous | 77,474 | — | 0.481 | drop-unmapped | Destination bits per second; same (n-1)/n convention as sload. |
| 15 | `sloss` | int64 | numeric-count | 409 | — | 0.544 | drop-unmapped | Source packets retransmitted or dropped. |
| 16 | `dloss` | int64 | numeric-count | 370 | — | 0.551 | drop-unmapped | Destination packets retransmitted or dropped. |
| 17 | `sinpkt` | float64 | numeric-continuous | 76,161 | — | 0.149 | drop-unmapped | Source interpacket arrival time (mSec). |
| 18 | `dinpkt` | float64 | numeric-continuous | 74,245 | — | 0.481 | drop-unmapped | Destination interpacket arrival time (mSec). |
| 19 | `sjit` | float64 | numeric-continuous | 77,532 | — | 0.505 | drop-unmapped | Source jitter (mSec). |
| 20 | `djit` | float64 | numeric-continuous | 76,831 | — | 0.535 | drop-unmapped | Destination jitter (mSec). |
| 21 | `swin` | int64 | numeric-count | 13 | — | 0.544 | drop-unmapped | Source TCP window advertisement value. |
| 22 | `stcpb` | int64 | numeric-count | 75,265 | — | 0.549 | drop-unmapped | Source TCP base sequence number. |
| 23 | `dtcpb` | int64 | numeric-count | 75,089 | — | 0.549 | drop-unmapped | Destination TCP base sequence number. |
| 24 | `dwin` | int64 | numeric-count | 7 | — | 0.549 | drop-unmapped | Destination TCP window advertisement value. |
| 25 | `tcprtt` | float64 | numeric-continuous | 43,319 | — | 0.549 | drop-unmapped | TCP connection setup round-trip time; sum of synack and ackdat. |
| 26 | `synack` | float64 | numeric-continuous | 40,142 | — | 0.549 | drop-unmapped | TCP setup time between the SYN and the SYN_ACK packets. |
| 27 | `ackdat` | float64 | numeric-continuous | 37,708 | — | 0.549 | drop-unmapped | TCP setup time between the SYN_ACK and the ACK packets. |
| 28 | `smean` | int64 | numeric-count | 1,357 | — | 0.232 | drop-unmapped | Mean packet size transmitted by the source. Empirically round(sbytes/spkts) on 99.06% of rows. |
| 29 | `dmean` | int64 | numeric-count | 1,328 | — | 0.481 | drop-unmapped | Mean packet size transmitted by the destination. |
| 30 | `trans_depth` | int64 | numeric-count | 11 | — | 0.898 | drop-unmapped | Pipelined depth into the connection of the http request/response transaction. |
| 31 | `response_body_len` | int64 | numeric-count | 2,386 | — | 0.936 | drop-unmapped | Uncompressed content size of data transferred from the server's http service. |
| 32 | `ct_srv_src` | int64 | numeric-count | 52 | — | 0.207 | drop-unmapped | No. of connections sharing the same service and source address in the last 100 connections. |
| 33 | `ct_state_ttl` | int64 | numeric-count | 5 | — | 0.468 | drop-unmapped | Bucket index for each state according to a specific range of source/destination TTL values. |
| 34 | `ct_dst_ltm` | int64 | numeric-count | 50 | — | 0.322 | drop-unmapped | No. of connections of the same destination address in the last 100 connections. |
| 35 | `ct_src_dport_ltm` | int64 | numeric-count | 47 | — | 0.529 | drop-unmapped | No. of connections of the same source address and destination port in the last 100 connections. |
| 36 | `ct_dst_sport_ltm` | int64 | numeric-count | 32 | — | 0.625 | drop-unmapped | No. of connections of the same destination address and source port in the last 100 connections. |
| 37 | `ct_dst_src_ltm` | int64 | numeric-count | 54 | — | 0.268 | drop-unmapped | No. of connections of the same source and destination address in the last 100 connections. |
| 38 | `is_ftp_login` | int64 | numeric-count | 4 | — | 0.985 | drop-unmapped | Documented as binary (1 if the ftp session was authenticated), but the delivered column takes 0/1/2/4 and is byte-identical to ct_ftp_cmd. |
| 39 | `ct_ftp_cmd` | int64 | numeric-count | 4 | — | 0.985 | drop-unmapped | No. of flows that has a command in an ftp session. Byte-identical to is_ftp_login in both delivered splits. |
| 40 | `ct_flw_http_mthd` | int64 | numeric-count | 11 | — | 0.898 | drop-unmapped | No. of flows that has methods such as Get and Post in http service. |
| 41 | `ct_src_ltm` | int64 | numeric-count | 50 | — | 0.244 | drop-unmapped | No. of connections of the same source address in the last 100 connections. |
| 42 | `ct_srv_dst` | int64 | numeric-count | 52 | — | 0.241 | drop-unmapped | No. of connections sharing the same service and destination address in the last 100 connections. |
| 43 | `is_sm_ips_ports` | int64 | binary-flag | 2 | — | 0.984 | drop-unmapped | 1 if source and destination IP addresses are equal and port numbers are equal, else 0. |
| 44 | `attack_cat` | object | label | 10 | — | 0.319 | label | Attack category name; 'Normal' for benign records. Delivered vocabulary uses 'Backdoor' (singular) where the docs say 'Backdoors'. |
| 45 | `label` | int64 | label | 2 | — | 0.681 | label | 0 for normal, 1 for attack. |

### 2.2 TON_IoT Network — 44 columns (211,043 rows)

| # | column | dtype | semantic | n_unique | n_missing (token) | top_share | disposition | meaning |
| ---: | --- | --- | --- | ---: | --- | ---: | --- | --- |
| 1 | `src_ip` | object | identity | 51 | — | 0.292 | drop-identity | Source IP address originating the flow. |
| 2 | `src_port` | int64 | identity | 26,628 | — | 0.017 | drop-identity | Originating endpoint's TCP/UDP source port. |
| 3 | `dst_ip` | object | identity | 753 | — | 0.226 | drop-identity | Destination IP address responding to the flow. |
| 4 | `dst_port` | int64 | identity | 2,039 | — | 0.335 | drop-identity | Responding endpoint's TCP/UDP destination port. |
| 5 | `proto` | object | categorical | 3 | — | 0.800 | mapped | Transport-layer protocol of the flow connection. |
| 6 | `service` | object | categorical | 9 | 132,032 ('-') | 0.626 | mapped | Dynamically detected application protocol (Zeek service); '-' when undetected. |
| 7 | `duration` | float64 | numeric-continuous | 68,570 | — | 0.284 | mapped | Flow duration in seconds: time of last packet seen minus time of first packet seen. |
| 8 | `src_bytes` | int64 | numeric-count | 2,199 | — | 0.655 | derived-input | Source PAYLOAD bytes, derived from TCP sequence numbers. Zero for 65.5% of rows (handshake-only / scan / ICMP flows carry no payload). NOT the counterpart of UNSW sbytes - see src_ip_bytes. |
| 9 | `dst_bytes` | int64 | numeric-count | 2,338 | — | 0.706 | derived-input | Destination PAYLOAD bytes from TCP sequence numbers; zero for 70.6% of rows. NOT the counterpart of UNSW dbytes - see dst_ip_bytes. |
| 10 | `conn_state` | object | categorical | 13 | — | 0.246 | mapped | Zeek connection state (S0, S1, S2, S3, SF, SH, SHR, REJ, RSTO, RSTOS0, RSTR, RSTRH, OTH). Disjoint vocabulary from UNSW state. |
| 11 | `missed_bytes` | int64 | numeric-count | 694 | — | 0.986 | unused | Number of missing bytes in content gaps; 0 on 98.6% of rows. |
| 12 | `src_pkts` | int64 | numeric-count | 274 | — | 0.480 | mapped | Number of packets originated by the source. |
| 13 | `src_ip_bytes` | int64 | numeric-count | 3,648 | — | 0.093 | mapped | Total IP bytes from the source (sum of IP total-length). The IP-level notion that matches UNSW sbytes; exceeds src_bytes on 99.5% of rows. |
| 14 | `dst_pkts` | int64 | numeric-count | 203 | — | 0.394 | mapped | Number of packets originated by the destination. |
| 15 | `dst_ip_bytes` | int64 | numeric-count | 3,304 | — | 0.394 | mapped | Total IP bytes from the destination; IP-level counterpart of UNSW dbytes. |
| 16 | `dns_query` | object | free-text | 726 | 176,198 ('-') | 0.835 | drop-unmapped | Domain name subject of the DNS query. |
| 17 | `dns_qclass` | int64 | categorical | 3 | 176,275 (0 (not-applicable sentinel)) | 0.835 | drop-unmapped | DNS query class value; 0 used as the not-applicable sentinel (83.5%). |
| 18 | `dns_qtype` | int64 | categorical | 12 | 176,275 (0 (not-applicable sentinel)) | 0.835 | drop-unmapped | DNS query type value; 0 used as the not-applicable sentinel (83.5%). |
| 19 | `dns_rcode` | int64 | categorical | 4 | 202,344 (0 (not-applicable sentinel)) | 0.959 | drop-unmapped | DNS response code; 0 conflates 'NOERROR' with 'not a DNS flow'. |
| 20 | `dns_AA` | object | binary-flag | 3 | 176,030 ('-') | 0.834 | drop-unmapped | DNS authoritative-answer flag (T/F), '-' when not a DNS flow. |
| 21 | `dns_RD` | object | binary-flag | 3 | 176,030 ('-') | 0.834 | drop-unmapped | DNS recursion-desired flag (T/F), '-' when not a DNS flow. |
| 22 | `dns_RA` | object | binary-flag | 3 | 176,030 ('-') | 0.834 | drop-unmapped | DNS recursion-available flag (T/F), '-' when not a DNS flow. |
| 23 | `dns_rejected` | object | binary-flag | 3 | 176,030 ('-') | 0.834 | drop-unmapped | DNS query rejected by the server (T/F), '-' when not a DNS flow. |
| 24 | `ssl_version` | object | categorical | 4 | 210,642 ('-') | 0.998 | unused | SSL/TLS version offered by the server. |
| 25 | `ssl_cipher` | object | categorical | 6 | 210,642 ('-') | 0.998 | unused | SSL cipher suite chosen by the server. |
| 26 | `ssl_resumed` | object | binary-flag | 3 | 210,642 ('-') | 0.998 | unused | SSL session-resumption flag (T/F). |
| 27 | `ssl_established` | object | binary-flag | 3 | 210,642 ('-') | 0.998 | unused | SSL connection-established flag (T/F). |
| 28 | `ssl_subject` | object | free-text | 6 | 211,032 ('-') | 1.000 | unused | Subject of the X.509 certificate offered by the server. |
| 29 | `ssl_issuer` | object | free-text | 5 | 211,032 ('-') | 1.000 | unused | Certificate-authority owner/originator of the SSL certificate. |
| 30 | `http_trans_depth` | object | numeric-count | 11 | 210,740 ('-') | 0.999 | unused | Pipelined depth into the HTTP connection. Object dtype because '-' is the not-applicable sentinel. |
| 31 | `http_method` | object | categorical | 4 | 210,756 ('-') | 0.999 | unused | HTTP request method (GET, POST, HEAD). |
| 32 | `http_uri` | object | free-text | 86 | 210,756 ('-') | 0.999 | unused | URI used in the HTTP request. |
| 33 | `http_version` | object | categorical | 2 | 210,745 ('-') | 0.999 | unused | HTTP version used. |
| 34 | `http_request_body_len` | int64 | numeric-count | 6 | 211,027 (0 (not-applicable sentinel)) | 1.000 | unused | Uncompressed content size transferred from the HTTP client. |
| 35 | `http_response_body_len` | int64 | numeric-count | 75 | 210,766 (0 (not-applicable sentinel)) | 0.999 | unused | Uncompressed content size transferred from the HTTP server. |
| 36 | `http_status_code` | int64 | categorical | 8 | 210,745 (0 (not-applicable sentinel)) | 0.999 | unused | HTTP status code returned by the server; 0 is the not-applicable sentinel. |
| 37 | `http_user_agent` | object | free-text | 35 | 210,756 ('-') | 0.999 | unused | HTTP User-Agent header value. Doc types this 'Number'; it is a string. |
| 38 | `http_orig_mime_types` | object | categorical | 3 | 211,027 ('-') | 1.000 | unused | Ordered vector of MIME types from the source system. |
| 39 | `http_resp_mime_types` | object | categorical | 10 | 210,839 ('-') | 0.999 | unused | Ordered vector of MIME types from the destination system. |
| 40 | `weird_name` | object | categorical | 11 | 210,687 ('-') | 0.998 | unused | Name of the Zeek protocol anomaly/violation observed. |
| 41 | `weird_addl` | object | free-text | 3 | 210,886 ('-') | 0.999 | unused | Additional information associated with the protocol anomaly. |
| 42 | `weird_notice` | object | binary-flag | 2 | 210,687 ('-') | 0.998 | unused | Whether the violation/anomaly was turned into a Zeek notice. |
| 43 | `label` | int64 | label | 2 | — | 0.763 | label | 0 indicates normal, 1 indicates attack. |
| 44 | `type` | object | label | 10 | — | 0.237 | label | Attack category tag ('normal' plus nine attack families). |

## 3. Categorical vocabularies (full value sets with counts)

### 3.1 `proto` — UNSW-NB15 (133 levels) vs TON_IoT (3 levels)

TON_IoT's entire protocol vocabulary, and UNSW's top 12 (the rest of UNSW's tail is a long
list of Argus protocol names, each <1% of rows; 31 levels have <100 train rows):

| level | UNSW train n | UNSW train % | UNSW test n | TON_IoT n | TON_IoT % |
| --- | ---: | ---: | ---: | ---: | ---: |
| `tcp` | 79,946 | 45.59 | 43,095 | 168,747 | 79.96 |
| `udp` | 63,283 | 36.09 | 29,418 | 42,015 | 19.91 |
| `unas` | 12,084 | 6.89 | 3,515 | 0 | 0.00 |
| `arp` | 2,859 | 1.63 | 987 | 0 | 0.00 |
| `ospf` | 2,595 | 1.48 | 676 | 0 | 0.00 |
| `sctp` | 1,150 | 0.66 | 324 | 0 | 0.00 |
| `any` | 300 | 0.17 | 96 | 0 | 0.00 |
| `gre` | 225 | 0.13 | 88 | 0 | 0.00 |
| `sun-nd` | 201 | 0.11 | 54 | 0 | 0.00 |
| `ipv6` | 201 | 0.11 | 61 | 0 | 0.00 |
| `mobile` | 201 | 0.11 | 52 | 0 | 0.00 |
| `swipe` | 201 | 0.11 | 52 | 0 | 0.00 |
| `icmp` | 15 | 0.01 | 0 | 281 | 0.13 |
| *121 further UNSW levels* | 12,095 | 6.90 | 3,914 | — | — |

Coverage of `{tcp, udp, icmp}`: **81.69%** of UNSW train, 88.07% of UNSW test, **100%** of
TON_IoT. UNSW `icmp` is 15 rows in train and **0 in test**, so an `icmp` one-hot column is
effectively untrainable in-distribution but fires on 281 TON_IoT rows.

### 3.2 `service` — both datasets, complete vocabularies

| level | UNSW train n | UNSW train % | UNSW test n | UNSW test % | TON_IoT n | TON_IoT % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `-` | 94,168 | 53.71 | 47,153 | 57.27 | 132,032 | 62.56 |
| `dns` | 47,294 | 26.97 | 21,367 | 25.95 | 39,446 | 18.69 |
| `http` | 18,724 | 10.68 | 8,287 | 10.07 | 37,029 | 17.55 |
| `smtp` | 5,058 | 2.88 | 1,851 | 2.25 | — | — |
| `ftp-data` | 3,995 | 2.28 | 1,396 | 1.70 | — | — |
| `ftp` | 3,428 | 1.96 | 1,552 | 1.89 | 1,065 | 0.50 |
| `ssh` | 1,302 | 0.74 | 204 | 0.25 | — | — |
| `pop3` | 1,105 | 0.63 | 423 | 0.51 | — | — |
| `ssl` | 56 | 0.03 | 30 | 0.04 | 1,025 | 0.49 |
| `gssapi` | — | — | — | — | 184 | 0.09 |
| `dce_rpc` | — | — | — | — | 136 | 0.06 |
| `smb` | — | — | — | — | 108 | 0.05 |
| `dhcp` | 94 | 0.05 | 26 | 0.03 | — | — |
| `snmp` | 80 | 0.05 | 29 | 0.04 | — | — |
| `irc` | 25 | 0.01 | 5 | 0.01 | — | — |
| `smb;gssapi` | — | — | — | — | 18 | 0.01 |
| `radius` | 12 | 0.01 | 9 | 0.01 | — | — |

Shared levels `{-, dns, ftp, http, ssl}` cover 93.34% / 95.21% / 99.79% of the three splits.
UNSW-only: `dhcp`, `ftp-data`, `irc`, `pop3`, `radius`, `smtp`, `snmp`, `ssh`. TON_IoT-only:
`dce_rpc`, `gssapi`, `smb`, `smb;gssapi` — note `smb;gssapi` is a **multi-valued cell**
(Zeek joins concurrently-detected services with `;`), so a naive one-hot creates a phantom level.

### 3.3 UNSW `state` vs TON_IoT `conn_state` — the two raw vocabularies side by side

**These share no tokens at all.** Left table is Argus; right table is Zeek. Phase 2 must supply
an explicit collapse map or drop the concept — there is nothing to align lexically.

| UNSW `state` (Argus) | train n | test n | | TON_IoT `conn_state` (Zeek) | n | % |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| `FIN` | 77,825 | 39,339 | | `S0` | 51,937 | 24.61 |
| `INT` | 82,275 | 34,163 | | `SF` | 50,210 | 23.79 |
| `CON` | 13,152 | 6,982 | | `REJ` | 44,852 | 21.25 |
| `REQ` | 1,991 | 1,842 | | `OTH` | 23,332 | 11.06 |
| `RST` | 83 | 1 | | `SH` | 12,014 | 5.69 |
| `ECO` | 12 | 0 | | `S1` | 10,771 | 5.10 |
| `ACC` | 0 | 4 | | `S3` | 6,557 | 3.11 |
| `URN` | 1 | 0 | | `SHR` | 5,629 | 2.67 |
| `CLO` | 0 | 1 | | `RSTR` | 1,989 | 0.94 |
| `PAR` | 1 | 0 | | `RSTRH` | 1,690 | 0.80 |
| `no` | 1 | 0 | | `RSTO` | 1,309 | 0.62 |
|  | |  | | `S2` | 627 | 0.30 |
|  | |  | | `RSTOS0` | 126 | 0.06 |

UNSW `state` carries three separate problems beyond the vocabulary gap: a **junk level** `no`
(1 row), a **train/test vocabulary mismatch** (`ECO`/`PAR`/`URN`/`no` are train-only; `ACC`/`CLO`
are test-only, 5 rows total), and a modal level `INT` (46.9%) that means 'no state recorded'.

### 3.4 Labels

Binary `label` agrees in encoding across both datasets (0 = normal, 1 = attack) and is a strict
function of the multiclass column on both sides (verified by cross-tab: no row has a `normal`/
`Normal` category with `label = 1`, or vice versa). `BINARY_LABEL_COL` in `schema_map.py` is
correct as written.

| split | normal (0) | attack (1) | normal share |
| --- | ---: | ---: | ---: |
| UNSW train | 56,000 | 119,341 | 31.9% |
| UNSW test | 37,000 | 45,332 | 44.9% |
| TON_IoT | 50,000 | 161,043 | 23.7% |

### 3.5 `attack_cat` (UNSW) vs `type` (TON_IoT) — full value counts

| UNSW `attack_cat` | train n | test n | | TON_IoT `type` | n |
| --- | ---: | ---: | --- | --- | ---: |
| `Normal` | 56,000 | 37,000 | | `normal` | 50,000 |
| `Generic` | 40,000 | 18,871 | | `backdoor` | 20,000 |
| `Exploits` | 33,393 | 11,132 | | `ddos` | 20,000 |
| `Fuzzers` | 18,184 | 6,062 | | `dos` | 20,000 |
| `DoS` | 12,264 | 4,089 | | `injection` | 20,000 |
| `Reconnaissance` | 10,491 | 3,496 | | `password` | 20,000 |
| `Analysis` | 2,000 | 677 | | `ransomware` | 20,000 |
| `Backdoor` | 1,746 | 583 | | `scanning` | 20,000 |
| `Shellcode` | 1,133 | 378 | | `xss` | 20,000 |
| `Worms` | 130 | 44 | | `mitm` | 1,043 |

## 4. Findings for Phase 2

### 4.1 The three documentation gaps, confirmed

**(a) `NUSW-NB15_features.csv` describes the 49-column full dataset, not the 45-column
partitioned set — confirmed, and the reconciliation is exact.** 11 documented names are not
delivered and 7 delivered names are not in the description file:

| | names |
| --- | --- |
| Documented, **not delivered** (11) | `srcip`, `sport`, `dstip`, `dsport`, `Stime`, `Ltime` — genuinely absent · `smeansz`, `dmeansz`, `res_bdy_len`, `Sintpkt`, `Dintpkt` — **present under new names** |
| Delivered, **not documented by name** (7) | `smean`, `dmean`, `response_body_len`, `sinpkt`, `dinpkt` — renames of the above · **`id`, `rate` — genuinely undocumented** |

Net: 49 documented − 6 truly absent + 2 genuinely new = **45**. ✓

Two corrections to how this gap was previously described:

- **The `ct_*` columns are *not* undocumented.** All ten delivered `ct_*` columns appear in
  `NUSW-NB15_features.csv` (rows 37–47). The one real defect is that `ct_src_ltm` is spelled
  **`ct_src_ ltm`** upstream — an embedded space — so an exact-string join silently misses it.
  `smean`/`dmean`/`response_body_len` are likewise documented, as `smeansz`/`dmeansz`/`res_bdy_len`.
  Only **`id` and `rate`** have no documentation entry at all.
- **`The UNSW-NB15 description.pdf` cannot resolve them.** It is a **single page of prose** —
  provenance (IXIA PerfectStorm, 100 GB pcap), the nine attack families, the 2,540,044-record
  total, the 175,341/82,332 partition sizes — with **no per-column list**. `ReadMe.pdf` is a
  one-page folder inventory plus licence. So for UNSW there is exactly **one** per-column
  artifact, `NUSW-NB15_features.csv`, and `id`/`rate` had to be resolved empirically:

| column | resolved meaning | evidence |
| --- | --- | --- |
| `id` | 1-based row index of the partitioned file (1…175,341 in train; unique per row) | 175,341 distinct values over 175,341 rows, contiguous range |
| `rate` | **packets per second** = `(spkts + dpkts − 1) / dur` | **99.82%** of the 172,684 `dur > 0` rows match within 0.1%. Competing forms fail: `(spkts+dpkts)/dur` matches 0.61%, `(spkts+dpkts−2)/dur` 0.61%, `(sbytes+dbytes)/dur` 0.00% |

Both are `description_source = empirical` in the CSV; every other column cites
`features_csv:<upstream name>` with the rename or case change noted inline.

`rate` is worth flagging for a second reason: it is a **near-duplicate of the promised
`pkts_per_sec` derived feature**, differing only by the Argus `(n−1)` interval convention.
It is still `drop-unmapped` — TON_IoT has no rate column, so the feature must be derived on both
sides anyway — but nobody should 'discover' `rate` later and think the derivation is redundant.

**(b) `DROP_COLUMNS` is largely dead — confirmed against the real headers.** 7 of the 12 named
columns exist in neither delivered file:

| `DROP_COLUMNS` entry | in UNSW 45? | in TON_IoT 44? |
| --- | --- | --- |
| `id` | **yes** | no |
| `srcip` | no | no |
| `sport` | no | no |
| `dstip` | no | no |
| `dsport` | no | no |
| `src_ip` | no | **yes** |
| `src_port` | no | **yes** |
| `dst_ip` | no | **yes** |
| `dst_port` | no | **yes** |
| `ts` | no | no |
| `stime` | no | no |
| `ltime` | no | no |

`srcip`, `sport`, `dstip`, `dsport` are the *full*-dataset UNSW names (never in the partitioned
set); `stime`/`ltime` were dropped when the partition was built; `ts` is TON_IoT's timestamp,
documented as feature #1 of 46 in `Network Features-Description.pdf` but **not present** in the
delivered 44-column file. Recommended corrected tuple:

```python
DROP_COLUMNS: tuple[str, ...] = (
    "id",                                        # UNSW row index, 1..N, unique per row
    "src_ip", "src_port", "dst_ip", "dst_port",  # TON_IoT identity columns
)
```

The stale names are harmless if Phase 2 drops by intersection, but actively dangerous if it
asserts membership — and the comment they carry ("timestamps") misleads a reader into thinking
a temporal leakage vector was handled when there is none left to handle.

The live risk is real and is worse on the TON_IoT side than the drop-list implies: TON_IoT has
only **51 distinct `src_ip`** and 753 distinct `dst_ip`
across 211,043 rows, so source IP alone is close to a label lookup table for the target era.
UNSW `id` is even starker: strictly increasing and, because the partition was built by
concatenating per-class blocks, monotonically informative about `attack_cat`.

**(c) `sload`/`dload` are lower-case in the partitioned set — confirmed.** The delivered header
has `sload`, `dload`, `spkts`, `dpkts`, `sjit`, `djit`, `label`; the description file has
`Sload`, `Dload`, `Spkts`, `Dpkts`, `Sjit`, `Djit`, `Label`. The whole delivered header is
lower-case. `CLAUDE.md` and the implementation plan both write `Sload`/`Dload`; any code copied
from those docs raises `KeyError`. (This is separate from the BOM issue, which corrupts only the
*first* column name.)

### 4.2 Missing-value encoding, with counts

**Neither dataset contains a single `NaN`** — `pandas` parses all 89 columns without a null.
Missingness is entirely sentinel-encoded, differently on each side.

**TON_IoT — literal `-` in object columns.** 22 of the 44 columns use it. Sorted by severity:

| column | `-` count | share | usable? |
| --- | ---: | ---: | --- |
| `ssl_subject` | 211,032 | 99.99% | **no** — <0.2% informative |
| `ssl_issuer` | 211,032 | 99.99% | **no** — <0.2% informative |
| `http_orig_mime_types` | 211,027 | 99.99% | **no** — <0.2% informative |
| `weird_addl` | 210,886 | 99.93% | **no** — <0.2% informative |
| `http_resp_mime_types` | 210,839 | 99.90% | **no** — <0.2% informative |
| `http_method` | 210,756 | 99.86% | **no** — <0.2% informative |
| `http_uri` | 210,756 | 99.86% | **no** — <0.2% informative |
| `http_user_agent` | 210,756 | 99.86% | **no** — <0.2% informative |
| `http_version` | 210,745 | 99.86% | **no** — <0.2% informative |
| `http_trans_depth` | 210,740 | 99.86% | **no** — <0.2% informative |
| `weird_name` | 210,687 | 99.83% | **no** — <0.2% informative |
| `weird_notice` | 210,687 | 99.83% | **no** — <0.2% informative |
| `ssl_version` | 210,642 | 99.81% | **no** — <0.2% informative |
| `ssl_cipher` | 210,642 | 99.81% | **no** — <0.2% informative |
| `ssl_resumed` | 210,642 | 99.81% | **no** — <0.2% informative |
| `ssl_established` | 210,642 | 99.81% | **no** — <0.2% informative |
| `dns_query` | 176,198 | 83.49% | marginal — 16.5% informative |
| `dns_AA` | 176,030 | 83.41% | marginal — 16.5% informative |
| `dns_RD` | 176,030 | 83.41% | marginal — 16.5% informative |
| `dns_RA` | 176,030 | 83.41% | marginal — 16.5% informative |
| `dns_rejected` | 176,030 | 83.41% | marginal — 16.5% informative |
| `service` | 132,032 | 62.56% | **yes** — `-` is a meaningful level ('not detected') |

So: **`service` is the only TON_IoT `-`-bearing column with enough signal to keep**, and there
`-` is a genuine category ('Zeek detected no application protocol'), matching UNSW's use of `-`
in its own `service`. The `dns_*` block is 83.4% empty (16.6% informative — DNS flows only); the
entire `ssl_*`, `http_*` and `weird_*` blocks are **99.8–100% empty**. `ssl_subject` and
`ssl_issuer` have **11 informative rows each** out of 211,043; `http_orig_mime_types` has 16.

**TON_IoT — `0` as a not-applicable sentinel in numeric columns.** Six numeric columns overload
0 to mean 'this was not an HTTP/DNS flow', which is *not* distinguishable from a real zero:

| column | rows == 0 | share | note |
| --- | ---: | ---: | --- |
| `dns_qclass` | 176,275 | 83.53% | matches the 83.5% non-DNS rate |
| `dns_qtype` | 176,275 | 83.53% | matches the 83.5% non-DNS rate |
| `dns_rcode` | 202,344 | 95.88% | **conflates DNS `NOERROR` with 'not DNS'** |
| `http_request_body_len` | 211,027 | 99.99% | matches the 99.9% non-HTTP rate |
| `http_response_body_len` | 210,766 | 99.87% | matches the 99.9% non-HTTP rate |
| `http_status_code` | 210,745 | 99.86% | matches the 99.9% non-HTTP rate |

None of the six reaches a model — the three `http_*` ones are `unused` (near-zero-variance) and
the three `dns_*` ones are `drop-unmapped` (no UNSW counterpart) — but the overloading is the
reason not to treat them as ordinary counts if the subspace is ever widened.

**UNSW — `-` in `service` only; `0` elsewhere is a real zero, not a sentinel.**
`service == '-'` on 94,168 train rows (53.7%) and
47,153 test rows (57.3%) — keep as a level. `state` has no `-` in
either delivered split (the description file lists one), but it does carry a junk `no` level
(1 row). Elsewhere UNSW's zeros are semantically meaningful and must **not** be imputed:

| column | rows == 0 | meaning |
| --- | ---: | --- |
| `dbytes` | 84,282 | destination never replied — genuine 0, and `dpkts`/`dttl`/`dload`/`dmean` agree on the same 84,282-row set |
| `dpkts` | 84,282 | same 'no reply' set |
| `dur` | 2,657 | sub-microsecond / single-packet flow, not missing |
| `sttl` | 3,162 | 3,162 rows with no source TTL observed |
| `swin` | 95,395 | non-TCP flow (no window advertisement) |
| `tcprtt` | 96,300 | non-TCP or no completed handshake |

The `d*` columns are mutually consistent: `dbytes == 0`, `dpkts == 0` and `dmean == 0` on exactly
the **same 84,282 rows** (48.07%). That is a structural 'no reply' pattern, and imputing it would
destroy the strongest legitimate signal in the shared subspace.

### 4.3 Constant and near-constant columns

**No column in either dataset is strictly constant** (`constant = False` for all 89 rows).
Near-zero-variance — single modal value holding ≥98% of rows — is the real problem, and it is
almost entirely a TON_IoT phenomenon:

| dataset | column | top_share | note |
| --- | --- | ---: | --- |
| unsw | `is_ftp_login` | 0.9854 | **byte-identical to each other** in both splits |
| unsw | `ct_ftp_cmd` | 0.9854 | **byte-identical to each other** in both splits |
| unsw | `is_sm_ips_ports` | 0.9842 | binary flag, 1.6% positive |
| toniot | `missed_bytes` | 0.9860 | only 2,945 non-zero rows |
| toniot | `ssl_version` | 0.9981 | protocol-specific block, see §4.2 |
| toniot | `ssl_cipher` | 0.9981 | protocol-specific block, see §4.2 |
| toniot | `ssl_resumed` | 0.9981 | protocol-specific block, see §4.2 |
| toniot | `ssl_established` | 0.9981 | protocol-specific block, see §4.2 |
| toniot | `ssl_subject` | 0.9999 | protocol-specific block, see §4.2 |
| toniot | `ssl_issuer` | 0.9999 | protocol-specific block, see §4.2 |
| toniot | `http_trans_depth` | 0.9986 | protocol-specific block, see §4.2 |
| toniot | `http_method` | 0.9986 | protocol-specific block, see §4.2 |
| toniot | `http_uri` | 0.9986 | protocol-specific block, see §4.2 |
| toniot | `http_version` | 0.9986 | protocol-specific block, see §4.2 |
| toniot | `http_request_body_len` | 0.9999 | protocol-specific block, see §4.2 |
| toniot | `http_response_body_len` | 0.9987 | protocol-specific block, see §4.2 |
| toniot | `http_status_code` | 0.9986 | protocol-specific block, see §4.2 |
| toniot | `http_user_agent` | 0.9986 | protocol-specific block, see §4.2 |
| toniot | `http_orig_mime_types` | 0.9999 | protocol-specific block, see §4.2 |
| toniot | `http_resp_mime_types` | 0.9990 | protocol-specific block, see §4.2 |
| toniot | `weird_name` | 0.9983 | protocol-specific block, see §4.2 |
| toniot | `weird_addl` | 0.9993 | protocol-specific block, see §4.2 |
| toniot | `weird_notice` | 0.9983 | protocol-specific block, see §4.2 |

**20 of TON_IoT's 44 columns (45%) are near-zero-variance** and carry `phase2_disposition =
`unused``. None of them has a UNSW counterpart, so this costs the shared subspace nothing — but
it is worth stating in the report that TON_IoT's 44 columns are not 44 features' worth of
information. Its effective flow-level width is roughly **15 columns**.

Two UNSW redundancies worth recording even though both columns are `drop-unmapped`:
`is_ftp_login` and `ct_ftp_cmd` are **byte-identical** (perfectly collinear), and `trans_depth`
equals `ct_flw_http_mthd` on 99.29% of rows. Also, `is_ftp_login` is documented as `Binary` but
actually takes values `{0, 1, 2, 4}` — a documentation error, not a parsing artifact.

### 4.4 `SHARED_FAMILIES` — proposed full alignment

`schema_map.py:53` is half-filled, and the half that is filled contains a **silent bug**: it maps
`"Backdoors"`, but the delivered `attack_cat` vocabulary uses **`Backdoor`** (singular). The plural
form appears in `NUSW-NB15_features.csv` and `UNSW-NB15_LIST_EVENTS.csv` but never in the
partitioned CSVs, so that entry currently matches **zero rows** and fails silently rather than
raising.

Only **three attack families (plus the benign class) have a genuine counterpart on both sides**.
Proposed complete mapping:

| shared family | UNSW `attack_cat` | UNSW train n | TON_IoT `type` | TON_IoT n | note |
| --- | --- | ---: | --- | ---: | --- |
| `normal` | `Normal` | 56,000 | `normal` | 50,000 | benign class |
| `dos` | `DoS` | 12,264 | `dos` | 20,000 | direct |
| `scanning` | `Reconnaissance` | 10,491 | `scanning` | 20,000 | direct (synonyms) |
| `backdoor` | `Backdoor` | 1,746 | `backdoor` | 20,000 | direct — **note the singular** |

**`ddos` is not a shared family.** `README.md` lists shared families as "`DoS`, `DDoS`,
`backdoor`, `scanning/reconnaissance`" — but **UNSW-NB15 has no DDoS class at all**
(`attack_cat` has exactly the ten levels in §3.5, and `UNSW-NB15_LIST_EVENTS.csv` lists no DDoS
subcategory either). `ddos` exists only on the TON_IoT side, so it cannot appear in a per-family
*cross-era* comparison. `README.md` needs that item removed. Folding `ddos` into `dos` would be a
silent redefinition of the DoS result and should not be done.

Families with **no counterpart**, which must be excluded from the per-family cross-era analysis:

| side | family | n | why it has no counterpart |
| --- | --- | ---: | --- |
| UNSW | `Exploits` | 33,393 | broad IXIA vulnerability-exploitation class; TON_IoT splits this behaviour across `injection`/`xss`/`password` with different generators |
| UNSW | `Generic` | 40,000 | block-cipher/generic-signature attacks, 99% from one IXIA `Generic,IXIA` subcategory — an artifact of the 2015 generator |
| UNSW | `Fuzzers` | 18,184 | protocol fuzzing; absent from TON_IoT |
| UNSW | `Analysis` | 2,000 | port scan / spam / HTML-file probing mix; overlaps `scanning` only partially, so mapping it would double-count |
| UNSW | `Shellcode` | 1,133 | OS-specific shellcode; absent from TON_IoT |
| UNSW | `Worms` | 130 | absent from TON_IoT; also too small to score |
| TON_IoT | `ddos` | 20,000 | absent from UNSW |
| TON_IoT | `injection` | 20,000 | web-app injection; closest UNSW analogue is `Exploits`, but the mapping is not defensible |
| TON_IoT | `password` | 20,000 | credential brute-force; no UNSW class |
| TON_IoT | `ransomware` | 20,000 | no 2015 counterpart — this is the class the drift story is about |
| TON_IoT | `xss` | 20,000 | no UNSW class |
| TON_IoT | `mitm` | 1,043 | no UNSW class; also the one class that misses its documented count |

Recommended `SHARED_FAMILIES`, with every delivered level named explicitly so an unmapped level
is a `KeyError` rather than a silent drop:

```python
SHARED_FAMILIES: dict[str, dict[str, str]] = {
    "unsw": {
        "Normal": "normal",
        "DoS": "dos",
        "Backdoor": "backdoor",        # delivered vocabulary is SINGULAR
        "Reconnaissance": "scanning",
        # no shared counterpart -> excluded from per-family analysis
        "Exploits": None, "Generic": None, "Fuzzers": None,
        "Analysis": None, "Shellcode": None, "Worms": None,
    },
    "toniot": {
        "normal": "normal",
        "dos": "dos",
        "backdoor": "backdoor",
        "scanning": "scanning",
        # no shared counterpart
        "ddos": None, "injection": None, "password": None,
        "ransomware": None, "xss": None, "mitm": None,
    },
}
```

Consequence for RQ1's per-family breakdown: it covers **24,501 of 119,341 UNSW attack rows
(20.5%)** and **60,000 of 161,043 TON_IoT attack rows (37.3%)**. The headline binary result uses
all rows; only the per-family table is restricted. That restriction is already promised in
`README.md`'s Limitations section, but the 20.5% figure should appear in the report so the
narrowness is explicit.

### 4.5 `state` ↔ `conn_state` — proposed coarse collapse

Given fully disjoint vocabularies (§3.3), the only defensible options are a hand-written
3-way collapse or dropping the feature. Proposed collapse, grounded in the Argus state
definitions in `NUSW-NB15_features.csv` and the Zeek `conn_state` definitions in
`bro_log_vars.pdf`:

| coarse level | UNSW `state` | UNSW train coverage | TON_IoT `conn_state` | TON_IoT coverage |
| --- | --- | ---: | --- | ---: |
| `completed` | `FIN`, `CLO` | 44.38% | `SF`, `S1`, `S2`, `S3` | 32.30% |
| `reset` | `RST` | 0.05% | `RSTO`, `RSTR`, `RSTOS0`, `RSTRH`, `REJ` | 23.68% |
| `no_response` | `INT`, `REQ`, `ACC`, `CON` | 55.56% | `S0`, `SH`, `SHR` | 32.97% |
| `other` | `ECO`, `PAR`, `URN`, `no` | 0.01% | `OTH` | 11.06% |

The map is exhaustive on both sides — every delivered level is assigned, so no row falls through.

The collapse is *lossy and asymmetric* — `reset` is 0.05% of UNSW but 23.7% of TON_IoT, and
`other` is 0.01% vs 11.1%. A model that never saw a reset in training gets a one-hot column that
fires on a quarter of the target era. **Recommendation: keep the collapsed feature, but run the
cross-era evaluation once with and once without it**, and report both. If the delta is large, the
feature is measuring the Argus↔Zeek instrumentation change rather than concept drift, and should
be dropped — which is exactly the confound the project is trying not to bundle in.

`CON` is placed in `no_response` deliberately: Argus `CON` means 'connected, no state
transition observed', which is closer to Zeek `S1`/`OTH` than to `FIN`. It is the single
judgement call in this table (13,152 UNSW train rows, 7.5%) and should be revisited if the
with/without test above shows sensitivity.

### 4.6 Bytes semantics — numerically, the coded pairing is wrong

`FEATURE_MAP` pairs `sbytes ↔ src_bytes`. The measurements say that pairs an **IP-level** count
against a **payload** count. Distribution comparison:

| statistic | UNSW `sbytes` | TON `src_bytes` (payload) | TON `src_ip_bytes` (IP) |
| --- | ---: | ---: | ---: |
| min | 28 | 0 | 0 |
| 25th pct | 114 | 0 | 48 |
| median | 430 | 0 | 82 |
| 75th pct | 1,418 | 130 | 415 |
| 99th pct | 74,125 | 3,016 | 4,800 |
| max | 12,965,233 | 3,890,855,126 | 6,522,626 |
| share == 0 | 0.00% | 65.46% | 8.10% |

Four independent lines of evidence, all pointing the same way:

1. **UNSW `sbytes` has a floor of 28 bytes and is never 0** (`smean` min is also 28). 28 = 20 B
   IPv4 header + 8 B UDP header — the smallest possible IP datagram carrying zero payload. A
   payload counter could not have that floor. UNSW `sbytes` is therefore IP-level.
2. **TON `src_bytes` is 0 on 65.5% of rows** (138,156 rows), including 121,062 rows that have
   `src_pkts > 0` — packets were sent but Zeek recorded no payload (SYN scans, rejected
   connections, ICMP). Pairing it with a column whose minimum is 28 forces a spurious
   distribution shift of exactly the kind RQ1 is trying to measure.
3. **The per-packet difference is exactly a header.** Median `(src_ip_bytes − src_bytes) /
   src_pkts` by protocol: **udp = 28 B**, **tcp = 48 B**, **icmp = 106 B** — 20 B IP + 8 B UDP,
   20 B IP + 20–28 B TCP (with options), and ICMP's full body (Zeek reports ICMP payload as 0).
   `src_ip_bytes ≥ src_bytes` on **99.53%** of rows.
4. **Per-packet magnitudes line up only for the IP pairing.** Median bytes-per-packet:

| protocol | UNSW `sbytes/spkts` | TON `src_ip_bytes/src_pkts` | TON `src_bytes/src_pkts` |
| --- | ---: | ---: | ---: |
| tcp | 80.4 | 52.0 | 0.0 |
| udp | 57.0 | 82.0 | 49.0 |
| icmp | 84.0 | 106.0 | 0.0 |

**Recommendation: repoint the TON_IoT side of `src_bytes`/`dst_bytes` to `src_ip_bytes`/
`dst_ip_bytes`, and put `src_bytes`/`dst_bytes` on the drop list instead.** That is the exact
opposite of what `README.md` and the `DROP_COLUMNS` comment currently say ("drop the unmapped
`*_ip_bytes` columns"). The instruction to "pick one notion and apply it consistently" is right;
the notion to pick is **total IP bytes**, because that is the only one UNSW offers.

Secondary reason to prefer `*_ip_bytes`: `src_bytes` carries 67 rows
above 10^8 bytes, topping out at 3,890,855,126 (3.9 GB in a single flow) — Zeek's
sequence-number-derived payload estimate breaks on wrapped or spoofed sequence numbers, and 985
rows (0.47%) even report more payload than total IP bytes. `src_ip_bytes` maxes at
6,522,626, three orders of magnitude lower and physically plausible. Feeding the
payload column into a UNSW-train-fitted z-scorer would map those 67 rows to standard scores in
the thousands.

### 4.7 `duration == 0` — the divide-by-zero guard, and a trap inside it

| dataset | column | rows == 0 | share | smallest non-zero |
| --- | --- | ---: | ---: | --- |
| UNSW train | `dur` | 2,657 | 1.52% | 1e-06 s |
| UNSW test | `dur` | 950 | 1.15% | 1e-06 s |
| TON_IoT | `duration` | 60,013 | 28.44% | 1e-06 s |

Both columns are clean `float64` with no non-numeric tokens, so the guard is the only issue —
but **the zero rate differs by 19×** (1.52% vs 28.44%), which makes the choice of guard a
cross-era design decision rather than a formality.

**The trap: `duration == 0` points in opposite directions in the two eras.**

| dataset | rows with duration == 0 | normal | attack | normal share |
| --- | ---: | ---: | ---: | ---: |
| UNSW train | 2,657 | 2,630 | 27 | **99.0%** |
| TON_IoT | 60,013 | 12,701 | 47,312 | **21.2%** |

Against base rates of 31.9% normal (UNSW train) and 23.7% normal (TON_IoT): a zero-duration flow
is a near-certain **normal** marker in 2015 (99.0%) and a mild **attack** marker in 2019–2020
(21.2% normal). Whatever sentinel `bytes_per_sec`/`pkts_per_sec` take on those rows becomes a
learned normal-class signal in training and fires on 60,013 target rows with the opposite
meaning. This is a genuine drift finding, but it also means the guard must not smuggle it in
accidentally.

**Recommendation:** guard with `np.where(dur > 0, x / dur, 0.0)` — an explicit 0, not `NaN`,
not `1e-6` (which would produce astronomically large rates) — and add a separate explicit
`zero_duration` binary flag so the effect is a named feature rather than an artifact hidden
inside two rate columns. Then it can be reported, or ablated, deliberately.

### 4.8 `sload` is bits per second — with an Argus correction nobody expects

Verified on the 172,360 rows with `dur > 0` and `sload > 0`:

| candidate formula | median `sload / formula` | rows within 1% | rows within 5% |
| --- | ---: | ---: | ---: |
| `sbytes / dur` (bytes/sec) | 6.0000 | 0.00% | 0.00% |
| `sbytes * 8 / dur` (naive bits/sec) | 0.7500 | 2.72% | 18.87% |
| `sbytes * 8 * (spkts-1) / (spkts * dur)` | 1.0000 | 99.98% | 100.00% |

**`sload = sbytes × 8 × (spkts − 1) / (spkts × dur)`** — 99.98% of rows within 1%. The factor 8
confirms bits, not bytes. The `(n−1)/n` factor is Argus measuring load over the *n−1 inter-packet
intervals* that `dur` actually spans, which is the same convention `rate` uses. Equivalently
`sload = (sbytes − smean) * 8 / dur`, which matches to the same 99.98%. `dload` behaves
identically.

Per-packet-count check, isolating the correction (median ratio of `sload` to naive `sbytes*8/dur`):

| `spkts` | rows | observed ratio | predicted `(n−1)/n` |
| ---: | ---: | ---: | ---: |
| 2 | 85,169 | 0.5000 | 0.5000 |
| 3 | 71 | 0.6668 | 0.6667 |
| 4 | 4,700 | 0.7500 | 0.7500 |
| 10 | 27,200 | 0.9007 | 0.9000 |

**Why this matters:** the median UNSW flow has `spkts = 2`, so the naive `sbytes*8/dur` is
**2× larger** than `sload` on a typical row, and the median ratio across all rows is 0.75 — not
1.0. Anyone who later computes `bytes_per_sec` and 'cross-checks' it against `sload` will find a
discrepancy that is neither the bits/bytes factor of 8 nor a bug, but the compounding of both
effects. **Do not cross-check the derived rates against `sload`/`dload` at all** — they are
`drop-unmapped` (TON_IoT has no rate column) and the derivation is defined independently on both
sides. `CLAUDE.md` and `README.md` describe `Sload`/`Dload` as 'bits per second', which is right
as far as it goes but omits the `(n−1)/n` term; the plan should say so.

### 4.9 TTL — re-confirmed absent from TON_IoT

Substring scan of all 44 delivered TON_IoT column names for `ttl` and `hop`: **zero matches**.
The complete list of TTL-bearing columns across both datasets is UNSW-only:

| dataset | TTL columns |
| --- | --- |
| UNSW-NB15 | `sttl` (11 distinct values, 0–255), `dttl` (6 distinct, 0–254), `ct_state_ttl` (5 distinct, 0–6) |
| TON_IoT | **none** |

This confirms the deviation recorded in `data/README.md` from the header itself rather than from
the Zeek-schema argument. The 44 columns are Zeek `conn.log` / `dns.log` / `ssl.log` / `http.log` /
`weird.log` fields (all 44 appear in `Network Features-Description.pdf`), and none of those logs
exports a per-flow IP TTL. TTL is recoverable only by reprocessing the raw captures — out of
scope for this timeline. All three UNSW TTL columns are `drop-unmapped`.

One note for the report's framing: `sttl` in UNSW train takes only **11 distinct values** and its
modal value covers 65.4% of rows, which is consistent with the generator-fingerprint argument in
`README.md` — a real network would show a far broader initial-TTL distribution.

### 4.10 Other cross-era hazards found while cataloguing

- **UNSW train and test have different categorical vocabularies.** `state`: `ECO`/`PAR`/`URN`/`no`
  are train-only, `ACC`/`CLO` are test-only. `proto`: 133 levels in train, 131 in test, but every
  test level appears in train. `service`: same 13 levels both sides. So the `RARE_BUCKET = "other"`
  mechanism is needed for the *in-distribution* split too, not just for TON_IoT — a one-hot
  encoder fitted on UNSW-train alone will hit unseen `state` levels on UNSW-test. Fit the encoder
  with `handle_unknown="infrequent_if_exist"` (or bucket before encoding) so this is impossible.
- **`service` cells can be multi-valued on the TON_IoT side.** `smb;gssapi` (18 rows) is Zeek
  joining two concurrently-detected services. Split on `;` and take the first token before
  bucketing, or it becomes a phantom category.
- **UNSW `dur` is capped at 60 s**, an artifact of the capture design (max observed 59.999989 s).
  TON_IoT's max is 93,516.9 s. Any duration-derived feature inherits this ceiling, and z-scoring
  with UNSW-train statistics maps most of TON_IoT's upper tail off-scale. Log1p then clip.
- **UNSW `icmp` has 15 train rows and 0 test rows**, but TON_IoT has 281. A `proto=icmp` one-hot
  is effectively untrained yet active cross-era.
- **`missed_bytes` is the only TON_IoT flow-level column that is both near-constant and
  unmapped**; it is not a leakage risk, just dead weight. Listed `unused`.

## 5. Recommended changes to `src/schema_map.py` (for Phase 2 to apply)

This catalogue does not modify `src/`. Five changes, in priority order:

1. **`FEATURE_MAP`: repoint the byte concepts** — `"src_bytes": ("sbytes", "src_ip_bytes")` and
   `"dst_bytes": ("dbytes", "dst_ip_bytes")`. The dictionary *keys* stay the same; only the
   TON_IoT column changes. (§4.6)
2. **`SHARED_FAMILIES`: `"Backdoors"` → `"Backdoor"`** — currently matches zero rows, silently.
   Then complete both sub-dicts per §4.4, with every delivered level named explicitly.
3. **`DROP_COLUMNS`: cut the 7 non-existent names**, keep `id` + the four TON_IoT identity
   columns, and move `src_bytes`/`dst_bytes` onto the drop list in place of `*_ip_bytes`. (§4.1b,
   §4.6)
4. **Add the `state ↔ conn_state` collapse map** from §4.5 as an explicit module-level dict, plus
   a `zero_duration` flag alongside `DERIVED_FEATURES` (§4.7).
5. **Fix the docs** (four separate corrections):
   - `CLAUDE.md`/`README.md`/the implementation plan write `Sload`/`Dload`; the delivered columns
     are `sload`/`dload` (§4.1c).
   - `README.md`'s instruction to "drop … unmapped `*_ip_bytes`" is backwards — the
     `*_ip_bytes` columns are the mappable ones (§4.6).
   - `README.md` lists `DDoS` as a shared attack family; UNSW-NB15 has no DDoS class (§4.4).
   - The `sload` unit note should mention the `(n−1)/n` term, not just "bits per second" (§4.8).

---

*Generated for Phase 1 from `data/raw/` and `data/raw/reference/`. Companion:
`reports/schema_catalogue.csv`.*
